
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol
import hashlib
import ipaddress
import os
from xml.parsers.expat import model
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import core
from core import (
    ChatError,
    ChatService,
    EmployeeContext,
    Forbidden,
    IntegrationNotConfigured,
    InvalidRequest,
    RateLimited,
    Unauthenticated,
)
from chatbot import prompts

log = logging.getLogger("employee_chat")

API_PREFIX = "/api/employee-chat"
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "employee-chat"
WIDGET_DIR = Path(__file__).resolve().parent / "widget"
WIDGET_FILES = {
    "widget.html": "text/html; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
}
_CONV_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

class AuthAdapter(Protocol):
    async def authenticate(self, request: Request) -> EmployeeContext:
        """Validate the dashboard session (cookie/header) and return the verified
        identity. Raise ``Unauthenticated`` for a missing/expired session."""
        ...

    async def feature_enabled(self, ctx: EmployeeContext) -> bool:
        """Existing feature flag / permission check for this employee + tenant."""
        ...


class UnconfiguredAuth:
    async def authenticate(self, request: Request) -> EmployeeContext:
        raise IntegrationNotConfigured()

    async def feature_enabled(self, ctx: EmployeeContext) -> bool:
        raise IntegrationNotConfigured()


class LocalDemoAuth:
    """
    LOCAL TESTING ONLY.
    Allows a simulated employee on the local machine.
    """

    async def authenticate(self, request: Request) -> EmployeeContext:

        client = request.client

        if client is None:
            raise Unauthenticated()

        try:
            address = ipaddress.ip_address(client.host)
        except ValueError:
            raise Unauthenticated() from None

        if not address.is_loopback:
            raise Unauthenticated()

        return EmployeeContext(
            employee_id="local-demo-employee",
            tenant_id="local-demo-tenant",
            roles=frozenset({"developer"}),
        )

    async def feature_enabled(self, ctx: EmployeeContext) -> bool:
        return True

@dataclass
class Settings:
    #: Extra origins allowed for cross-origin embedding (same-origin always works).
    allowed_origins: frozenset[str] = frozenset()
    max_body_bytes: int = 8 * 1024
    rate_limit_per_minute: int = 20
    serve_widget_assets: bool = True


class SlidingWindowLimiter:
    """Per-process limiter. Behind several workers, use the host's shared limiter."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check(self, key: tuple[str, str]) -> None:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            raise RateLimited()
        q.append(now)
        if len(self._hits) > 10000:  # bound memory
            for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                del self._hits[k]


# --------------------------------------------------------------------------- #
# Request model / helpers
# --------------------------------------------------------------------------- #


class MessageIn(BaseModel):
    conversation_id: Optional[str] = Field(default=None, max_length=64)
    message: str = Field(max_length=core.MAX_MESSAGE_CHARS * 4)  # core enforces the real limit


def _json(payload: dict[str, Any], status: int = 200, headers: Optional[dict[str, str]] = None) -> JSONResponse:
    h = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    h.update(headers or {})
    return JSONResponse(payload, status_code=status, headers=h)


def _error(exc: ChatError, headers: Optional[dict[str, str]] = None) -> JSONResponse:
    return _json({"success": False, "error": {"code": exc.code, "message": exc.user_message}}, exc.status, headers)


def _origin_ok(request: Request, settings: Settings) -> bool:
    origin = request.headers.get("origin")
    if origin is None:
        return True  # non-browser or same-origin GET; the custom header still applies
    if origin in settings.allowed_origins:
        return True
    host = request.headers.get("host", "")
    return re.sub(r"^https?://", "", origin) == host


def _check_csrf(request: Request, settings: Settings) -> None:
    if request.headers.get(CSRF_HEADER) != CSRF_VALUE or not _origin_ok(request, settings):
        raise Forbidden("Request rejected.")


async def _read_json_limited(request: Request, max_bytes: int) -> Any:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise InvalidRequest("Request is too large.")
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > max_bytes:
            raise InvalidRequest("Request is too large.")
    try:
        return json.loads(bytes(body) or b"null")
    except ValueError:
        raise InvalidRequest() from None




def create_app(
    *,
    auth: Optional[AuthAdapter] = None,
    report_source: Optional[core.ReportSource] = None,
    model: Optional[core.ModelGateway] = None,
    store: Optional[core.ConversationStore] = None,
    settings: Optional[Settings] = None,
    service: Optional[ChatService] = None,
) -> FastAPI:
    settings = settings or Settings()
    service = service or ChatService(
    model or core.OpenAICompatibleModelGateway(),
    store or core.InMemoryConversationStore(),
    auth = auth or UnconfiguredAuth(),
    report_source = report_source or core.UnconfiguredReportSource()
)
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    app = FastAPI(title="Employee Report Assistant", docs_url=None, redoc_url=None, openapi_url=None)
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=sorted(settings.allowed_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Requested-With"],
        )

    router = APIRouter(prefix=API_PREFIX)

    async def verified(request: Request, *, mutating: bool) -> EmployeeContext:
        if mutating:
            _check_csrf(request, settings)
        ctx = await auth.authenticate(request)
        if not await auth.feature_enabled(ctx):
            raise Forbidden()
        return ctx

    @router.get("/bootstrap")
    async def bootstrap(request: Request):
        ctx = await verified(request, mutating=False)
        limiter.check((ctx.tenant_id, ctx.employee_id))
        return _json({
            "success": True,
            "enabled": True,
            "max_message_length": core.MAX_MESSAGE_CHARS,
            "welcome": prompts.WELCOME_MESSAGE,
        })

    @router.post("/message")
    async def message(request: Request):
        ctx = await verified(request, mutating=True)
        limiter.check((ctx.tenant_id, ctx.employee_id))
        raw = await _read_json_limited(request, settings.max_body_bytes)
        try:
            body = MessageIn.model_validate(raw)
        except ValidationError:
            raise InvalidRequest() from None
        if body.conversation_id is not None and not _CONV_ID.match(body.conversation_id):
            raise core.ConversationNotFound()
        return _json(await service.handle_message(ctx, body.conversation_id, body.message))

    @router.get("/conversations/{conversation_id}")
    async def conversation(conversation_id: str, request: Request):
        ctx = await verified(request, mutating=False)
        limiter.check((ctx.tenant_id, ctx.employee_id))
        if not _CONV_ID.match(conversation_id):
            raise core.ConversationNotFound()
        return _json(await service.get_conversation(ctx, conversation_id))

    if settings.serve_widget_assets:
        # Static, secret-free assets. Hosts with their own asset pipeline can
        # set serve_widget_assets=False and ship the three files themselves.
        @router.get("/widget/{name}")
        async def widget_asset(name: str):
            media = WIDGET_FILES.get(name)
            path = WIDGET_DIR / name
            if media is None or not path.is_file():
                return Response(status_code=404)
            return Response(path.read_bytes(), media_type=media, headers={
                "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff",
            })

    app.include_router(router)

    @app.exception_handler(ChatError)
    async def on_chat_error(request: Request, exc: ChatError):
        extra = {"Retry-After": "30"} if isinstance(exc, RateLimited) else None
        return _error(exc, extra)

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, exc: Exception):
        log.exception("unhandled error in employee chat")  # details stay server-side
        return _error(ChatError())

    return app

def build_app() -> FastAPI:

    dev_mode = os.getenv(
        "EMPLOYEE_CHAT_DEV_MODE", "0"
    ).lower() in {"1", "true", "yes"}

    if dev_mode:

        environment = os.getenv(
            "APP_ENV", "development"
        ).lower()

        if environment == "production":
            raise RuntimeError(
                "Local demo mode cannot run in production."
            )

        test_reports = (
            Path(__file__).resolve().parent
            / "test_reports"
        )

        log.warning(
            "LOCAL DEMO MODE ENABLED. "
            "Do not use this configuration in production."
        )

        return create_app(
            auth=LocalDemoAuth(),
            report_source=core.FileReportSource(test_reports),
            model=core.OpenAICompatibleModelGateway(),
            store=core.InMemoryConversationStore(),
        )

    return create_app()


app = build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)