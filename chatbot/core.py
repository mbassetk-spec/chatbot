"""Chatbot engine: retrieval, grounding, orchestration and validation.

This module contains no HTTP and no UI code. It talks to the host SaaS through
three narrow interfaces that the host must implement (see ``main.create_app``):

* ``ReportSource``       - permission-aware report search / access checks
* ``ModelGateway``       - the SaaS operator's server-side AI gateway
* ``ConversationStore``  - optional persistence (an in-memory default is given)

Nothing here invents report data. If an adapter is not wired up, the
``Unconfigured*`` placeholders fail with a clear 503 instead of answering.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import os
from openai import AsyncOpenAI
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence
from  chatbot import prompts
log = logging.getLogger("employee_chat")

MAX_MESSAGE_CHARS = 2000

class ChatError(Exception):
    code = "internal_error"
    status = 500
    user_message = "Something went wrong. Please try again."

    def __init__(self, user_message: Optional[str] = None):
        super().__init__(user_message or self.user_message)
        if user_message:
            self.user_message = user_message


class InvalidRequest(ChatError):
    code, status = "invalid_request", 400
    user_message = "That request could not be processed."


class Unauthenticated(ChatError):
    code, status = "unauthenticated", 401
    user_message = "Your session has expired. Please sign in again."


class Forbidden(ChatError):
    code, status = "forbidden", 403
    user_message = "You do not have access to this feature."


class ConversationNotFound(ChatError):
    code, status = "conversation_not_found", 404
    user_message = "Conversation not found."


class RateLimited(ChatError):
    code, status = "rate_limited", 429
    user_message = "You are sending messages too quickly. Please wait a moment."


class SourceUnavailable(ChatError):
    code, status = "reports_unavailable", 503
    user_message = "The reports service is temporarily unavailable. Please try again shortly."


class ModelUnavailable(ChatError):
    code, status = "assistant_unavailable", 503
    user_message = "The assistant is temporarily unavailable. Please try again shortly."


class ModelTimeout(ChatError):
    code, status = "assistant_timeout", 504
    user_message = "The assistant took too long to respond. Please try again."


class InvalidModelOutput(ChatError):
    code, status = "assistant_error", 502
    user_message = "The assistant could not produce a reliable answer. Please try again."


class IntegrationNotConfigured(ChatError):
    code, status = "integration_not_configured", 503
    user_message = "This feature has not been fully set up by your administrator."


@dataclass(frozen=True)
class EmployeeContext:
    """Identity derived from the host's verified authentication. Never from the client."""

    employee_id: str
    tenant_id: str
    roles: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ReportChunk:
    """One authorized piece of report content returned by the host.

    Only ``report_id``, ``title`` and ``content`` are required. Optional
    metadata is passed on only if the host actually supplies it.
    """

    report_id: str
    title: str
    content: str
    category: Optional[str] = None
    department: Optional[str] = None
    period: Optional[str] = None
    version: Optional[str] = None
    updated_at: Optional[str] = None
    section: Optional[str] = None
    is_latest: Optional[bool] = None


@dataclass(frozen=True)
class Citation:
    report_id: str
    title: str
    period: Optional[str] = None
    section: Optional[str] = None
    version: Optional[str] = None
    updated_at: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StoredMessage:
    role: str  # "user" | "assistant"
    content: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = False
    ts: float = field(default_factory=time.time)


@dataclass
class Conversation:
    id: str
    messages: list[StoredMessage] = field(default_factory=list)




class ReportSource(Protocol):
    """Adapter over the SaaS backend's approved, permission-aware report APIs.

    Implementations MUST enforce tenant and report permissions for ``ctx``
    themselves (the browser is never trusted) and must not return anything the
    employee could not open in the dashboard.
    """

    async def search(
        self, ctx: EmployeeContext, query: str, *, limit: int
    ) -> Sequence[ReportChunk]: ...

    async def can_access(self, ctx: EmployeeContext, report_id: str) -> bool: ...

    async def report_url(self, ctx: EmployeeContext, report_id: str) -> Optional[str]:
        """Same-origin dashboard path for the report, or None."""
        ...


class ModelGateway(Protocol):
    """The operator's server-side model service. Credentials stay inside it."""

    async def generate(
        self, *, system: str, messages: list[dict[str, str]], timeout_s: float
    ) -> str: ...


class ConversationStore(Protocol):
    async def create(self, ctx: EmployeeContext) -> Conversation: ...
    async def get(self, ctx: EmployeeContext, conversation_id: str) -> Optional[Conversation]: ...
    async def append(
        self, ctx: EmployeeContext, conversation_id: str, messages: Sequence[StoredMessage]
    ) -> None: ...


class UnconfiguredReportSource:
    async def search(self, ctx, query, *, limit):
        raise IntegrationNotConfigured()

    async def can_access(self, ctx, report_id):
        raise IntegrationNotConfigured()

    async def report_url(self, ctx, report_id):
        raise IntegrationNotConfigured()


class UnconfiguredModelGateway:
    async def generate(self, *, system, messages, timeout_s):
        raise IntegrationNotConfigured()

class OpenAICompatibleModelGateway:
    """
    Connects to any provider exposing an OpenAI-compatible
    Chat Completions API.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.getenv("CHAT_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None

        # Create the client only when the first request arrives.
        self._client: Optional[AsyncOpenAI] = None

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        timeout_s: float,
    ) -> str:

        if not self.api_key:
            raise IntegrationNotConfigured(
                "OPENAI_API_KEY is not configured."
            )

        if not self.model:
            raise IntegrationNotConfigured(
                "CHAT_MODEL is not configured."
            )

        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                *messages,
            ],
            timeout=timeout_s,
        )

        content = response.choices[0].message.content

        if not isinstance(content, str) or not content.strip():
            raise InvalidModelOutput()

        return content.strip()
class InMemoryConversationStore:
    """Process-local, bounded, TTL-limited store.

    Suitable for a single worker and for development. For multiple workers or
    retention requirements, implement ``ConversationStore`` on the host's
    existing storage. Conversations are keyed by tenant + employee + id, so one
    employee can never read another's.
    """

    def __init__(self, ttl_s: float = 3600, max_conversations: int = 5000, max_messages: int = 40):
        self._data: dict[tuple[str, str, str], tuple[float, Conversation]] = {}
        self._ttl, self._max, self._max_msgs = ttl_s, max_conversations, max_messages

    def _purge(self) -> None:
        now = time.time()
        for k in [k for k, (t, _) in self._data.items() if now - t > self._ttl]:
            del self._data[k]
        while len(self._data) > self._max:
            del self._data[min(self._data, key=lambda k: self._data[k][0])]

    async def create(self, ctx: EmployeeContext) -> Conversation:
        self._purge()
        conv = Conversation(id=uuid.uuid4().hex)
        self._data[(ctx.tenant_id, ctx.employee_id, conv.id)] = (time.time(), conv)
        return conv

    async def get(self, ctx: EmployeeContext, conversation_id: str) -> Optional[Conversation]:
        self._purge()
        item = self._data.get((ctx.tenant_id, ctx.employee_id, conversation_id))
        return item[1] if item else None

    async def append(self, ctx, conversation_id, messages) -> None:
        key = (ctx.tenant_id, ctx.employee_id, conversation_id)
        item = self._data.get(key)
        if not item:
            raise ConversationNotFound()
        conv = item[1]
        conv.messages.extend(messages)
        del conv.messages[: -self._max_msgs]
        self._data[key] = (time.time(), conv)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_message(raw: Any) -> str:
    if not isinstance(raw, str):
        raise InvalidRequest("Please enter a message.")
    text = unicodedata.normalize("NFKC", raw)
    text = _CONTROL_CHARS.sub("", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text:
        raise InvalidRequest("Please enter a message.")
    if len(text) > MAX_MESSAGE_CHARS:
        raise InvalidRequest(f"Messages are limited to {MAX_MESSAGE_CHARS} characters.")
    return text


def build_retrieval_query(message: str, history: Sequence[StoredMessage]) -> str:
    """Resolve short follow-ups ("compare it with the previous month") by
    prepending the employee's previous question. Only the employee's own words
    are used, never earlier model output."""
    if history and len(message.split()) <= 12:
        prev = next((m.content for m in reversed(history) if m.role == "user"), "")
        if prev:
            return f"{prev}\n{message}"[:600]
    return message


# --------------------------------------------------------------------------- #
# Model output parsing
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_model_output(raw: str) -> tuple[bool, str, list[str]]:
    if not isinstance(raw, str):
        raise InvalidModelOutput()
    text = _FENCE.sub("", raw.strip())
    start = text.find("{")
    if start < 0:
        raise InvalidModelOutput()
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        raise InvalidModelOutput() from None
    answerable, answer, used = obj.get("answerable"), obj.get("answer"), obj.get("used_sources", [])
    if not isinstance(answerable, bool) or not isinstance(answer, str):
        raise InvalidModelOutput()
    if not isinstance(used, list) or not all(isinstance(u, str) for u in used):
        raise InvalidModelOutput()
    return answerable, answer.strip()[:4000], used


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class ChatService:
    def __init__(
        self,
        source: ReportSource,
        model: ModelGateway,
        store: ConversationStore,
        *,
        max_sources: int = 6,
        max_chunk_chars: int = 3000,
        max_evidence_chars: int = 12000,
        search_timeout_s: float = 10.0,
        model_timeout_s: float = 30.0,
        history_turns: int = 6,
    ):
        self.source, self.model, self.store = source, model, store
        self.max_sources = max_sources
        self.max_chunk_chars = max_chunk_chars
        self.max_evidence_chars = max_evidence_chars
        self.search_timeout_s = search_timeout_s
        self.model_timeout_s = model_timeout_s
        self.history_turns = history_turns

    # ---- public API ------------------------------------------------------- #

    async def handle_message(
        self, ctx: EmployeeContext, conversation_id: Optional[str], message: Any
    ) -> dict[str, Any]:
        started = time.monotonic()
        status, report_ids = "error", []
        try:
            text = normalize_message(message)
            conv = await self._load_or_create(ctx, conversation_id)

            chunks = await self._retrieve(ctx, build_retrieval_query(text, conv.messages))
            report_ids = sorted({c.report_id for c in chunks})

            if not chunks:
                result = await self._finish(ctx, conv, text, prompts.NO_INFO_MESSAGE, [], False)
                status = "no_evidence"
                return result

            labelled = {f"S{i}": c for i, c in enumerate(chunks, 1)}
            history = await self._usable_history(ctx, conv)
            raw = await self._call_model(text, labelled, history)
            answerable, answer, used = parse_model_output(raw)

            used_chunks = [labelled[u] for u in dict.fromkeys(used) if u in labelled]
            if not answerable or not used_chunks:
                answer, used_chunks = prompts.NO_INFO_MESSAGE, []
                status = "no_evidence"
            else:
                status = "answered"

            citations = await self._build_citations(ctx, used_chunks)
            if citations is None:  # access revoked between retrieval and citation
                answer, citations, status = prompts.NO_INFO_MESSAGE, [], "no_evidence"

            return await self._finish(ctx, conv, text, answer, citations, bool(citations))
        finally:
            log.info(
                "employee_chat status=%s tenant=%s employee=%s reports=%s duration_ms=%d",
                status, ctx.tenant_id, ctx.employee_id, report_ids,
                int((time.monotonic() - started) * 1000),
            )

    async def get_conversation(self, ctx: EmployeeContext, conversation_id: str) -> dict[str, Any]:
        conv = await self.store.get(ctx, conversation_id)
        if conv is None:
            raise ConversationNotFound()
        access_cache: dict[str, bool] = {}
        out = []
        for m in conv.messages:
            if m.role == "assistant" and m.citations:
                cites = await self._revalidate(ctx, m.citations, access_cache)
                if cites is None:
                    out.append({"role": "assistant", "content": prompts.ACCESS_CHANGED_MESSAGE, "citations": []})
                    continue
                out.append({"role": "assistant", "content": m.content, "citations": [c.to_dict() for c in cites]})
            else:
                out.append({"role": m.role, "content": m.content, "citations": []})
        return {"success": True, "conversation_id": conv.id, "messages": out}

    # ---- internals -------------------------------------------------------- #

    async def _load_or_create(self, ctx, conversation_id: Optional[str]) -> Conversation:
        if conversation_id is None:
            return await self.store.create(ctx)
        conv = await self.store.get(ctx, conversation_id)
        if conv is None:  # unknown, expired, or someone else's: indistinguishable
            raise ConversationNotFound()
        return conv

    async def _retrieve(self, ctx: EmployeeContext, query: str) -> list[ReportChunk]:
        try:
            found = await asyncio.wait_for(
                self.source.search(ctx, query, limit=self.max_sources * 2),
                self.search_timeout_s,
            )
        except ChatError:
            raise
        except Exception as exc:  # adapter failure, timeout, etc.
            log.warning("report search failed: %s", type(exc).__name__)
            raise SourceUnavailable() from None

        # Defence in depth: re-check every report before anything reaches the model.
        access: dict[str, bool] = {}
        kept: list[ReportChunk] = []
        seen: set[tuple[str, Optional[str], str]] = set()
        budget = self.max_evidence_chars
        for c in found:
            if not (isinstance(c, ReportChunk) and c.report_id and c.title and c.content):
                continue
            key = (c.report_id, c.section, c.content[:80])
            if key in seen:
                continue
            if c.report_id not in access:
                access[c.report_id] = await self._safe_can_access(ctx, c.report_id)
            if not access[c.report_id]:
                continue
            seen.add(key)
            content = c.content[: self.max_chunk_chars]
            if budget - len(content) < 0 or len(kept) >= self.max_sources:
                break
            budget -= len(content)
            kept.append(ReportChunk(**{**c.__dict__, "content": content}))
        return kept

    async def _safe_can_access(self, ctx, report_id: str) -> bool:
        try:
            return bool(await asyncio.wait_for(self.source.can_access(ctx, report_id), self.search_timeout_s))
        except ChatError:
            raise
        except Exception:
            raise SourceUnavailable() from None

    async def _usable_history(self, ctx, conv: Conversation) -> list[dict[str, str]]:
        """Recent turns for interpreting follow-ups. Assistant turns whose
        source reports are no longer accessible are dropped."""
        cache: dict[str, bool] = {}
        out: list[dict[str, str]] = []
        for m in conv.messages[-self.history_turns * 2:]:
            if m.role == "assistant" and m.citations:
                if await self._revalidate(ctx, m.citations, cache) is None:
                    continue
            out.append({"role": m.role, "content": m.content[:800]})
        return out

    async def _call_model(self, question, labelled: dict[str, ReportChunk], history) -> str:
        sources = [{"label": lbl, **c.__dict__} for lbl, c in labelled.items()]
        try:
            return await asyncio.wait_for(
                self.model.generate(
                    system=prompts.get_system_prompt(),
                    messages=prompts.build_messages(question, sources, history),
                    timeout_s=self.model_timeout_s,
                ),
                self.model_timeout_s + 2,
            )
        except asyncio.TimeoutError:
            raise ModelTimeout() from None
        except ChatError:
            raise
        except Exception as exc:
            log.warning("model call failed: %s", type(exc).__name__)
            raise ModelUnavailable() from None

    async def _build_citations(self, ctx, chunks: Sequence[ReportChunk]) -> Optional[list[Citation]]:
        """Citations come only from actually-retrieved chunks, re-authorized now.
        Returns None if any used report is no longer accessible."""
        cites: dict[tuple[str, Optional[str]], Citation] = {}
        access: dict[str, bool] = {}
        for c in chunks:
            if c.report_id not in access:
                access[c.report_id] = await self._safe_can_access(ctx, c.report_id)
            if not access[c.report_id]:
                return None
            if (c.report_id, c.section) in cites:
                continue
            url = await self._safe_url(ctx, c.report_id)
            cites[(c.report_id, c.section)] = Citation(
                report_id=c.report_id, title=c.title, period=c.period,
                section=c.section, version=c.version, updated_at=c.updated_at, url=url,
            )
        return list(cites.values())

    async def _revalidate(self, ctx, cites: Sequence[Citation], cache: dict[str, bool]) -> Optional[list[Citation]]:
        out = []
        for c in cites:
            if c.report_id not in cache:
                cache[c.report_id] = await self._safe_can_access(ctx, c.report_id)
            if not cache[c.report_id]:
                return None
            out.append(Citation(**{**c.__dict__, "url": await self._safe_url(ctx, c.report_id)}))
        return out

    async def _safe_url(self, ctx, report_id: str) -> Optional[str]:
        try:
            url = await self.source.report_url(ctx, report_id)
        except Exception:
            return None
        # Only same-origin relative dashboard paths are ever returned.
        return url if isinstance(url, str) and url.startswith("/") and not url.startswith("//") else None

    async def _finish(self, ctx, conv, question, answer, citations, grounded) -> dict[str, Any]:
        await self.store.append(
            ctx, conv.id,
            [
                StoredMessage("user", question),
                StoredMessage("assistant", answer, list(citations), grounded),
            ],
        )
        return {
            "success": True,
            "message": answer,
            "grounded": grounded,
            "citations": [c.to_dict() for c in citations],
            "conversation_id": conv.id,
        }