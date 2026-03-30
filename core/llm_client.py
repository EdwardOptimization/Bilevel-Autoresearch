"""Shared LLM client — multi-provider, with retry and think-tag stripping.

Supported providers (all OpenAI-compatible except Anthropic):
  deepseek  → api.deepseek.com          (env: DEEPSEEK_API_KEY)
  openai    → api.openai.com            (env: OPENAI_API_KEY)
  glm       → open.bigmodel.cn          (env: GLM_API_KEY)
  minimax   → api.minimaxi.chat         (env: MINIMAX_API_KEY)
  anthropic → api.anthropic.com         (env: ANTHROPIC_API_KEY)  [native SDK]
"""
import json
import logging
import os
import re
import time

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Module-level client caches ─────────────────────────────────────────────────
_client_cache: dict[tuple[str, str], OpenAI] = {}
_anthropic_client_cache: dict = {}


def _get_openai_client(api_key: str, base_url: str) -> OpenAI:
    """Get or create a cached OpenAI client for this (api_key, base_url) pair."""
    cache_key = (api_key, base_url)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(api_key=api_key, base_url=base_url)
    return _client_cache[cache_key]


def _get_anthropic_client(api_key: str):
    """Get or create a cached Anthropic client for this api_key."""
    if api_key not in _anthropic_client_cache:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install 'anthropic' package: pip install anthropic")
        _anthropic_client_cache[api_key] = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client_cache[api_key]


# ── Provider registry ─────────────────────────────────────────────────────────
PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "native_sdk": False,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "native_sdk": False,
    },
    "glm": {
        "base_url": os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "default_model": os.environ.get("GLM_MODEL", "glm-4"),
        "api_key_env": "GLM_API_KEY",
        "native_sdk": False,
    },
    "minimax": {
        "base_url": "https://api.minimaxi.chat/v1",
        "default_model": "MiniMax-M2.7-highspeed",
        "api_key_env": "MINIMAX_API_KEY",
        "native_sdk": False,
    },
    "anthropic": {
        "base_url": "",  # uses native SDK
        "default_model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "native_sdk": True,
    },
}

# Module-level config (set once via configure())
_provider_name: str = "deepseek"
_api_key: str = ""
_model_override: str = ""


def configure(provider: str, api_key: str, model: str = "") -> None:
    """Configure the global LLM client. Call once at startup."""
    global _provider_name, _api_key, _model_override
    if provider.lower() not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    _provider_name = provider.lower()
    _api_key = api_key
    _model_override = model


def get_provider_info() -> dict:
    """Return current provider info (for display)."""
    p = PROVIDERS[_provider_name]
    model = _model_override or p["default_model"]
    return {"provider": _provider_name, "model": model}


# ── Core call ─────────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    system: str = None,
    model: str = "",
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Call the configured LLM provider with automatic retry.

    Returns the response text with <think>...</think> tags stripped.
    """
    provider = PROVIDERS[_provider_name]
    use_model = model or _model_override or provider["default_model"]
    api_key = _api_key or os.environ.get(provider["api_key_env"], "")

    if not api_key:
        raise ValueError(
            f"No API key for provider '{_provider_name}'. "
            f"Set env var {provider['api_key_env']} or pass via configure()."
        )

    last_error = None
    for attempt in range(max_retries):
        try:
            if provider["native_sdk"]:
                text = _call_anthropic(prompt, system, use_model, max_tokens, api_key)
            else:
                text = _call_openai_compat(prompt, system, use_model, max_tokens, api_key, provider["base_url"])
            return _strip_thinking_tags(text)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}") from last_error


def _call_openai_compat(
    prompt: str, system: str, model: str, max_tokens: int, api_key: str, base_url: str
) -> str:
    client = _get_openai_client(api_key, base_url)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens
    )
    msg = response.choices[0].message
    content = msg.content or ""
    # Reasoning models (e.g. GLM-5.1, DeepSeek-R1) put output in reasoning_content
    if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
        content = msg.reasoning_content
    return content


def _call_anthropic(
    prompt: str, system: str, model: str, max_tokens: int, api_key: str
) -> str:
    client = _get_anthropic_client(api_key)
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks that some models include in output."""
    if not text:
        return text
    # Strip <think>...</think> blocks (MiniMax, DeepSeek R1, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip leftover opening tags if the model got cut off mid-think
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict | list:
    """
    Robustly parse JSON from an LLM response.

    Handles: plain JSON, markdown code fences, inline JSON.
    Uses non-greedy matching to avoid consuming multiple JSON objects.
    """
    if not text:
        return {"raw_content": ""}

    text = _strip_thinking_tags(text).strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fence (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find first balanced JSON object or array (non-greedy, balanced brace scan)
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # malformed — try next strategy

    logger.debug("parse_json_response: could not find valid JSON, returning raw_content")
    return {"raw_content": text}


# ── Instance-level client (no global state mutation) ──────────────────────────

class LLMClient:
    """
    Instance-level LLM client.

    Use this instead of configure() + call_llm() when multiple providers must
    coexist in the same process (e.g. inner loop uses MiniMax, outer loop uses
    DeepSeek).  Does not touch module-level globals.
    """

    def __init__(self, provider: str, api_key: str = "", model: str = ""):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
        pinfo = PROVIDERS[provider]
        self._api_key = api_key or os.environ.get(pinfo["api_key_env"], "")
        self._model = model or pinfo["default_model"]
        self._base_url = pinfo["base_url"]
        self._native_sdk = pinfo["native_sdk"]
        # Pre-create and cache the underlying client at construction time.
        if self._native_sdk:
            self._client = _get_anthropic_client(self._api_key)
        else:
            self._client = _get_openai_client(self._api_key, self._base_url)

    def call(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        last_error = None
        for attempt in range(max_retries):
            try:
                if self._native_sdk:
                    kwargs = {
                        "model": self._model,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if system:
                        kwargs["system"] = system
                    response = self._client.messages.create(**kwargs)
                    text = response.content[0].text
                else:
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    response = self._client.chat.completions.create(
                        model=self._model, messages=messages, max_tokens=max_tokens
                    )
                    text = response.choices[0].message.content or ""
                return _strip_thinking_tags(text)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts: {last_error}"
        ) from last_error
