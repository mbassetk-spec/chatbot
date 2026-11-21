
(function () {
  'use strict';

  if (window.__employeeChatLoaded) return;
  window.__employeeChatLoaded = true;

  var script = document.currentScript;
  var apiBase = ((script && script.dataset.apiBase) || '/api/employee-chat').replace(/\/+$/, '');
  var assetsBase = (
    (script && script.dataset.assetsBase) ||
    (script && script.src ? script.src.replace(/\/[^/]*$/, '') : apiBase + '/widget')
  ).replace(/\/+$/, '');

  var REQUEST_TIMEOUT_MS = 45000;
  var STORAGE_KEY = 'employeeChat.conversationId';

  var state = {
    maxLength: 2000,
    conversationId: null,
    sending: false,
    sessionExpired: false,
    lastFailedText: null,
    open: false,
    minimized: false
  };
  var els = {};

  function apiFetch(path, options) {
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS);
    var init = {
      method: (options && options.method) || 'GET',
      credentials: 'same-origin',
      signal: controller.signal,
      headers: { 'Accept': 'application/json', 'X-Requested-With': 'employee-chat' }
    };
    if (options && options.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(options.body);
    }
    return fetch(apiBase + path, init).then(function (res) {
      return res.json().catch(function () { return null; }).then(function (data) {
        return { ok: res.ok, status: res.status, data: data };
      });
    }).finally(function () { clearTimeout(timer); });
  }

  function errorMessage(result) {
    var err = result && result.data && result.data.error;
    return (err && typeof err.message === 'string' && err.message) ||
      'Something went wrong. Please try again.';
  }


  function loadConversationId() {
    try { return window.sessionStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }
  function saveConversationId(id) {
    try {
      if (id) window.sessionStorage.setItem(STORAGE_KEY, id);
      else window.sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) { /* storage unavailable: conversation just won't survive reloads */ }
  }



  function mount() {
    return fetch(assetsBase + '/widget.html', { credentials: 'same-origin' })
      .then(function (res) {
        if (!res.ok) throw new Error('widget markup unavailable');
        return res.text();
      })
      .then(function (html) {
        var host = document.createElement('div');
        host.id = 'employee-chat-host';
        var shadow = host.attachShadow({ mode: 'open' });

        var css = document.createElement('link');
        css.rel = 'stylesheet';
        css.href = assetsBase + '/style.css';
        shadow.appendChild(css);

        // First-party static markup (never model output), parsed without scripts.
        var parsed = new DOMParser().parseFromString(html, 'text/html');
        var root = parsed.body.querySelector('.eac-root');
        if (!root) throw new Error('widget markup invalid');
        shadow.appendChild(document.importNode(root, true));
        document.body.appendChild(host);

        ['launcher', 'panel', 'status', 'new', 'minimize', 'close', 'body', 'welcome',
          'welcome-text', 'messages', 'loading', 'error', 'error-text', 'retry',
          'form', 'input', 'send'].forEach(function (id) {
          els[id] = shadow.getElementById('eac-' + id);
        });
        els.launcher.hidden = false;
      });
  }

  function bind() {
    els.launcher.addEventListener('click', openPanel);
    els.close.addEventListener('click', closePanel);
    els.minimize.addEventListener('click', toggleMinimize);
    els.new.addEventListener('click', newConversation);
    els.retry.addEventListener('click', function () {
      if (state.lastFailedText) send(state.lastFailedText, true);
    });
    els.form.addEventListener('submit', function (e) {
      e.preventDefault();
      send(els.input.value, false);
    });
    els.input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        els.form.requestSubmit();
      }
    });
    els.input.addEventListener('input', autosize);
    els.panel.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closePanel();
    });
  }


  function openPanel() {
    state.open = true;
    els.launcher.hidden = true;
    els.launcher.setAttribute('aria-expanded', 'true');
    els.panel.hidden = false;
    setMinimized(false);
    if (!state.sessionExpired) els.input.focus();
  }

  function closePanel() {
    state.open = false;
    els.panel.hidden = true;
    els.launcher.hidden = false;
    els.launcher.setAttribute('aria-expanded', 'false');
    els.launcher.focus();
  }

  function setMinimized(value) {
    state.minimized = value;
    els.panel.classList.toggle('eac-panel--minimized', value);
    els.minimize.setAttribute('aria-pressed', String(value));
    els.minimize.setAttribute('aria-label', value ? 'Restore' : 'Minimize');
    els.minimize.title = value ? 'Restore' : 'Minimize';
  }

  function toggleMinimize() {
    setMinimized(!state.minimized);
    if (!state.minimized && !state.sessionExpired) els.input.focus();
  }

  function newConversation() {
    if (state.sending) return;
    state.conversationId = null;
    saveConversationId(null);
    state.lastFailedText = null;
    els.messages.textContent = '';
    clearError();
    updateWelcome();
    if (!state.sessionExpired) els.input.focus();
  }


  function updateWelcome() {
    els.welcome.hidden = els.messages.childElementCount > 0;
  }

  function scrollToEnd() {
    els.body.scrollTop = els.body.scrollHeight;
  }

  function autosize() {
    els.input.style.height = 'auto';
    els.input.style.height = Math.min(els.input.scrollHeight, 120) + 'px';
  }

  function safeReportUrl(value) {
    if (typeof value !== 'string' || value.charAt(0) !== '/' || value.charAt(1) === '/') return null;
    try {
      var u = new URL(value, window.location.origin);
      if (u.origin !== window.location.origin) return null;
      return u.pathname + u.search + u.hash;
    } catch (e) { return null; }
  }

  function renderCitation(c) {
    var li = document.createElement('li');
    li.className = 'eac-citation';

    var title = document.createElement('span');
    title.className = 'eac-citation__title';
    var href = safeReportUrl(c.url);
    if (href) {
      var a = document.createElement('a');
      a.href = href;
      a.textContent = String(c.title || 'Report');
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      title.appendChild(a);
    } else {
      title.textContent = String(c.title || 'Report');
    }
    li.appendChild(title);

    var parts = [];
    if (c.period) parts.push('Period: ' + c.period);
    if (c.section) parts.push('Section: ' + c.section);
    if (c.version) parts.push('Version: ' + c.version);
    else if (c.updated_at) parts.push('Updated: ' + c.updated_at);
    if (parts.length) {
      var meta = document.createElement('span');
      meta.className = 'eac-citation__meta';
      meta.textContent = parts.join(' · ');
      li.appendChild(meta);
    }
    return li;
  }

  // All text (including model output) goes in via textContent, never innerHTML.
  function appendMessage(role, text, citations) {
    var li = document.createElement('li');
    li.className = 'eac-msg eac-msg--' + role;

    var bubble = document.createElement('p');
    bubble.className = 'eac-bubble';
    bubble.textContent = String(text);
    li.appendChild(bubble);

    if (role === 'assistant' && Array.isArray(citations) && citations.length) {
      var list = document.createElement('ul');
      list.className = 'eac-citations';
      list.setAttribute('aria-label', 'Sources');
      citations.forEach(function (c) {
        if (c && typeof c === 'object') list.appendChild(renderCitation(c));
      });
      li.appendChild(list);
    }
    els.messages.appendChild(li);
    updateWelcome();
    scrollToEnd();
  }

  function setBusy(busy) {
    els.loading.hidden = !busy;
    els.send.disabled = busy || state.sessionExpired;
    els.input.disabled = busy || state.sessionExpired;
    els.body.setAttribute('aria-busy', String(busy));
    if (busy) scrollToEnd();
  }

  function showError(message, canRetry) {
    els.error.hidden = false;
    els['error-text'].textContent = message;
    els.retry.hidden = !canRetry;
  }

  function clearError() {
    els.error.hidden = true;
    els['error-text'].textContent = '';
    els.retry.hidden = true;
  }

  function setOnline(online) {
    els.status.dataset.state = online ? 'online' : 'offline';
    els.status.setAttribute('aria-label', online ? 'Online' : 'Offline');
  }


  function send(rawText, isRetry) {
    if (state.sending || state.sessionExpired) return;
    var text = String(rawText || '').trim();
    if (!text) return;
    if (text.length > state.maxLength) {
      showError('Messages are limited to ' + state.maxLength + ' characters.', false);
      return;
    }

    state.sending = true;
    clearError();
    if (!isRetry) {
      appendMessage('user', text);
      els.input.value = '';
      autosize();
    }
    setBusy(true);

    apiFetch('/message', { method: 'POST', body: { conversation_id: state.conversationId, message: text } })
      .then(function (res) {
        if (res.ok && res.data && res.data.success) {
          state.lastFailedText = null;
          state.conversationId = res.data.conversation_id || state.conversationId;
          saveConversationId(state.conversationId);
          setOnline(true);
          appendMessage('assistant', res.data.message, res.data.citations);
          return;
        }
        state.lastFailedText = text;
        if (res.status === 401) {
          state.sessionExpired = true;
          showError('Your session has expired. Please refresh the page or sign in again.', false);
        } else if (res.status === 404) {
          // Conversation expired on the server; the retry starts a fresh one.
          state.conversationId = null;
          saveConversationId(null);
          showError('That conversation has expired. Retry to start a new one.', true);
        } else {
          showError(errorMessage(res), res.status >= 429 || res.status === 0);
        }
      })
      .catch(function () {
        state.lastFailedText = text;
        setOnline(false);
        showError('Could not reach the server. Check your connection and try again.', true);
      })
      .finally(function () {
        state.sending = false;
        setBusy(false);
        if (state.open && !state.minimized && !state.sessionExpired) els.input.focus();
      });
  }

  function restoreConversation() {
    var id = loadConversationId();
    if (!id) return Promise.resolve();
    return apiFetch('/conversations/' + encodeURIComponent(id)).then(function (res) {
      if (res.ok && res.data && Array.isArray(res.data.messages)) {
        state.conversationId = id;
        res.data.messages.forEach(function (m) {
          if (m && (m.role === 'user' || m.role === 'assistant')) {
            appendMessage(m.role, m.content, m.citations);
          }
        });
      } else {
        saveConversationId(null);
      }
    }).catch(function () { /* start fresh */ });
  }

  function init() {
    // Ask the backend first: unauthorized employees never get the widget at all.
    apiFetch('/bootstrap').then(function (res) {
      if (!(res.ok && res.data && res.data.enabled === true)) return null;
      if (typeof res.data.max_message_length === 'number') state.maxLength = res.data.max_message_length;
      return mount().then(function () {
        els.input.maxLength = state.maxLength;
        if (typeof res.data.welcome === 'string' && res.data.welcome) {
          els['welcome-text'].textContent = res.data.welcome;
        }
        bind();
        updateWelcome();
        return restoreConversation();
      });
    }).catch(function (err) {
      if (window.console && console.debug) console.debug('Employee chat unavailable:', err && err.message);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();