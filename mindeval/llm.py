"""Model calls, through litellm: one request shape for every call, sync (the judge) or awaited (the interactions,
through `engine`).

Nothing provider-specific is decided here. Whatever a provider needs beyond the model, the messages
and the sampling settings (vLLM's `chat_template_kwargs`, Claude's `thinking`, OpenAI's
`reasoning_effort`) arrives in `params`, from config/patient.yaml for the patient and from
`--counselor_params` for the counselor.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_LIB = None


def _lib():
    """Import litellm on first use (it is slow to import) and configure it once.

    `drop_params` removes a parameter a provider does not accept instead of failing the call;
    `modify_params` lets litellm adapt a conversation to a provider's rules. Callers that fan out
    to threads call this first, so no worker waits on another's import.
    """
    global _LIB
    if _LIB is None:
        import litellm

        litellm.drop_params = True
        litellm.modify_params = True
        # Under `modify_params`, litellm also drops `thinking` from any Claude request whose last assistant
        # tool call carries no thinking block. Every patient turn replays its `commit_move` that way, so
        # every call after the first move would reason nothing, silently. Adaptive thinking accepts the
        # replay, so the guard is turned off; a litellm without the hook fails here rather than drop again.
        from litellm.llms.anthropic.chat import transformation as anthropic_chat
        assert hasattr(anthropic_chat, "last_assistant_with_tool_calls_has_no_thinking_blocks")
        anthropic_chat.last_assistant_with_tool_calls_has_no_thinking_blocks = lambda messages: False
        for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _LIB = litellm
    return _LIB


def _serialize_tool_calls(tool_calls) -> list[dict]:
    out = []
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None)
        out.append({"id": getattr(tc, "id", None),
                    "name": getattr(fn, "name", None) if fn else None,
                    "arguments": getattr(fn, "arguments", None) if fn else None})
    return out


def _plain(items) -> list[dict] | None:
    """A list of provider objects as plain dicts, so they can be written to a trace and sent back."""
    if not items:
        return None
    return [i.model_dump() if hasattr(i, "model_dump") else dict(i) for i in items]


def response_meta(response) -> dict:
    """What a completion carries besides its text: model, finish reason, the reasoning trace, tool
    calls and usage. Best-effort; empty fields are dropped."""
    try:
        choice = response.choices[0]
        msg = choice.message
    except Exception:  # noqa: BLE001 — metadata never breaks a call
        return {}
    meta: dict[str, Any] = {
        "model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        # litellm normalises a reasoning channel into `reasoning_content`; vLLM names it `reasoning`.
        "reasoning_content": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
        "thinking_blocks": getattr(msg, "thinking_blocks", None),
        # OpenAI models through litellm's Responses bridge (`azure/responses/...`, `openai/responses/...`) return
        # their reasoning as opaque items, encrypted when the call asks for `reasoning.encrypted_content`.
        "reasoning_items": _plain(getattr(msg, "reasoning_items", None)
                                  or (getattr(msg, "provider_specific_fields", None) or {}).get("reasoning_items")),
        "tool_calls": _serialize_tool_calls(getattr(msg, "tool_calls", None)),
    }
    usage = getattr(response, "usage", None)
    if usage is not None:
        try:
            meta["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        except Exception:  # noqa: BLE001
            meta["usage"] = str(usage)
    return {k: v for k, v in meta.items() if v not in (None, [], {})}


def _request(messages: list[dict], model: str, api_base: str | None, temperature: float | None,
             max_tokens: int | None, timeout: float, seed: int | None, tools: list[dict] | None,
             tool_choice: dict | str | None, params: dict | None, api_key: str | None) -> dict[str, Any]:
    """The litellm request, the same for the sync and the async call."""
    lib = _lib()
    kwargs: dict[str, Any] = {**(params or {}), "model": model, "messages": messages, "timeout": timeout}
    # A parameter the caller named explicitly is sent even where litellm's model table says the model
    # does not take it; `drop_params` would otherwise remove it without a word. Body extras pass as-is.
    # A key litellm already maps for this model (e.g. reasoning_effort -> Anthropic's `thinking`) is
    # left off the list: forcing it through allowed_openai_params too makes litellm send both the
    # translation and the raw field, and Anthropic rejects the raw one with a 400.
    supported = set(getattr(lib, "get_supported_openai_params", lambda **_: None)(model=model) or ())
    named = [k for k in (params or {}) if k not in ("extra_body", "extra_headers") and k not in supported]
    if named:
        kwargs["allowed_openai_params"] = named
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if seed is not None:
        kwargs["seed"] = seed
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return kwargs


def call_messages(messages: list[dict], model: str, api_base: str | None, temperature: float | None,
                  max_tokens: int | None = None, timeout: float = 600, *, seed: int | None = None,
                  tools: list[dict] | None = None, tool_choice: dict | str | None = None,
                  params: dict | None = None, api_key: str | None = None) -> tuple[str, dict]:
    """One chat completion. Returns (content, response_meta).

    No token budget is sent unless one is given: a reasoning model thinks for an unbounded number of
    tokens before it writes, and a ceiling that lands inside the thinking returns an empty body that
    reads as a refusal. `timeout` is the real bound. `seed` is best-effort (dropped where
    unsupported). A `tool_choice` naming a function makes the call mandatory.
    """
    response = _lib().completion(**_request(messages, model, api_base, temperature, max_tokens, timeout, seed,
                                            tools, tool_choice, params, api_key))
    return (response.choices[0].message.content or ""), response_meta(response)


async def acall_messages(messages: list[dict], model: str, api_base: str | None, temperature: float | None,
                         max_tokens: int | None = None, timeout: float = 600, *, seed: int | None = None,
                         tools: list[dict] | None = None, tool_choice: dict | str | None = None,
                         params: dict | None = None, api_key: str | None = None) -> tuple[str, dict]:
    """`call_messages`, awaited: the same request through litellm's `acompletion`."""
    response = await _lib().acompletion(**_request(messages, model, api_base, temperature, max_tokens, timeout,
                                                   seed, tools, tool_choice, params, api_key))
    return (response.choices[0].message.content or ""), response_meta(response)


def _extract_json(text: str):
    """The first complete JSON object or array in `text`, ignoring prose around it; None if none."""
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def parse_json(text: str) -> dict:
    """Parse a JSON answer, tolerating ```fences``` and prose around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj = _extract_json(text)
        if obj is not None:
            return obj
        raise


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_inline_trace(text: str, meta: dict) -> tuple[str, dict]:
    """The words after a reasoning trace left inline, with the trace moved to
    `meta["reasoning_content"]`.

    A chat template that ends the prompt with `<think>` starts the completion inside the trace, and
    a server without a reasoning parser returns `trace</think>reply` as the content. Split on the
    first `</think>`: the completion began inside the trace, so the first close ends it. A trace
    litellm already lifted out is not overwritten, and text with no close tag is returned as it is.
    """
    if THINK_CLOSE not in text:
        return text, meta
    head, _, tail = text.partition(THINK_CLOSE)
    head = head.strip()
    if head.startswith(THINK_OPEN):
        head = head[len(THINK_OPEN):].strip()
    if head and not meta.get("reasoning_content"):
        meta = {**meta, "reasoning_content": head}
    return tail, meta
