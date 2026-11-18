"""How the interaction runtime reaches a model: through NeMo UserSim's `acall_llm`, so the engine's call
accounting, its debug log and its hosted runtime see every call the patient and the counselor make.

`acall_llm` resolves a role (`user_model` is the patient, `assistant_model` the counselor) to a facade and hands it
the conversation. The facade here is `Facade`: one side's `CallSpec`, called through litellm by `llm.acall_messages`
exactly as the requests have always been built (provider params, `seed`, `tools`, a forced `tool_choice`).

On the way, `acall_llm` rewrites the conversation into Data Designer chat messages, and three things mindeval sends
do not survive that: an earlier turn's reasoning (deliberately cleared; the counselor's own earlier turns carry theirs
back, see `session.CARRIED`), a tool result's `name`, and the `None` content of a turn that is only a tool call
(Anthropic rejects the empty text block it becomes). So `chat` keeps the conversation as it built it beside the call,
and the facade sends that one when it is the same conversation (same length, same roles), which it always is unless
something other than `chat` called the facade.

Retries stay mindeval's: each call site has its own (a patient server under load is waited out, an empty or cut-off
reply is asked for again), so a failed request reaches `chat` as it failed, neither retried again inside `acall_llm`
nor counted there as a call: UserSim's accounting sees the calls that answered. On the models-file path one layer
sits below that: Data Designer's HTTP client retries a 502, 503, 504 or dropped connection (3 times, with backoff)
before the failure reaches mindeval; a 429 comes back at once, and Data Designer lowers how many requests it
sends that model at a time.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

# Not in UserSim's extension API (`usersim.extension`) as of 0.3.2: the aliases of the roles mindeval never calls.
from usersim.engine.config import MODEL_API_RESPONSE, MODEL_SUMMARY
from usersim.extension import MODEL_ASSISTANT, MODEL_JUDGE, MODEL_USER, acall_llm

from mindeval import llm
from mindeval.config import CallSpec

PATIENT, COUNSELOR = MODEL_USER, MODEL_ASSISTANT
log = logging.getLogger(__name__)

#: The call in flight in this task: the conversation as `chat` built it, and the facade's (text, meta) once it
#: answers. A dict, shared rather than copied, so the answer reaches `chat` whatever task the facade ran in.
_CALL: ContextVar[dict | None] = ContextVar("mindeval_call", default=None)


async def chat(models: dict[str, Any], role: str, messages: list[dict], **kwargs: Any) -> tuple[str, dict]:
    """One completion for `role` (PATIENT or COUNSELOR). Returns (content, response_meta), as `llm.call_messages`
    does. `kwargs` are the call's own settings (`seed`, `tools`, `tool_choice`, `timeout`); the side's model,
    endpoint, key, temperature and params come from its facade."""
    call: dict = {"messages": messages}
    token = _CALL.set(call)
    try:
        reply = await acall_llm(models, role, messages, **{k: v for k, v in kwargs.items() if v is not None})
    except Exception:
        if "error" in call:  # the facade's call failed: that failure, as it was, to the caller's retry loop
            raise call["error"] from None
        raise
    finally:
        _CALL.reset(token)
    if "meta" in call:
        return call["text"], call["meta"]
    meta = _meta_of(reply)  # answered by a facade that is not ours: a hosted run, or the models file's (HandsBack)
    for key in ("finish_reason", "usage"):
        if call.get(key):
            meta[key] = call[key]
    return reply.get("content") or "", meta


async def pause(seconds: float) -> None:
    """A wait between attempts; every other member keeps going meanwhile."""
    await asyncio.sleep(seconds)


async def retry_text(models: dict[str, Any], role: str, messages: list[dict], *, spec: CallSpec,
                     seed: int | None = None, what: str = "completion") -> tuple[str, dict]:
    """A text completion, retried with capped backoff, `spec.max_retries` attempts in all.

    An empty completion is a failure, not an answer: recording it would put a blank turn in the
    transcript. So is a `finish_reason=length` cut-off with no closed reasoning trace and no
    `reasoning_content` beside it: that is a trace the model never finished, not a reply.
    """
    last: Exception | None = None
    for attempt in range(1, spec.max_retries + 1):
        try:
            raw, meta = await chat(models, role, messages, seed=seed)
            raw, meta = llm.split_inline_trace(raw, meta)
            if not raw.strip():
                raise ValueError(f"empty model output (finish_reason={meta.get('finish_reason')})")
            if (meta.get("finish_reason") == "length" and llm.THINK_CLOSE not in raw
                    and not meta.get("reasoning_content")):
                raise ValueError("model output cut off (finish_reason=length) with no closed reasoning trace")
            return raw.strip(), meta
        except Exception as e:  # noqa: BLE001 — retry on anything, report on exhaustion
            last = e
            if attempt < spec.max_retries:
                await pause(min(2 ** attempt, 30))
    raise llm.TextCallError(f"{what} failed after {spec.max_retries} attempts: {type(last).__name__}: {last}") from last


def _meta_of(reply: dict) -> dict:
    """response_meta's fields, from `acall_llm`'s reply when no litellm response is at hand."""
    calls = [{"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
              "arguments": (c.get("function") or {}).get("arguments")} for c in reply.get("tool_calls") or []]
    meta = {"reasoning_content": reply.get("reasoning_content"), "tool_calls": calls}
    return {k: v for k, v in meta.items() if v}


class _HandedBack(TypeError):
    """A failed call on its way back to `chat`. A TypeError, because UserSim's `acall_llm` neither retries one
    (`_NEVER_RETRY`) nor records it as a call."""


@dataclass(frozen=True)
class Facade:
    """One side's model as UserSim's engine calls it: `acompletion`, answered through litellm."""

    spec: CallSpec

    @property
    def model_name(self) -> str:
        """The model behind the role, which UserSim folds into each trajectory's id."""
        return self.spec.model

    async def acompletion(self, messages: list, **kwargs: Any) -> SimpleNamespace:
        call = _CALL.get()
        sent = call["messages"] if call is not None and _same(call["messages"], messages) else _as_dicts(messages)
        if call is None or sent is not call["messages"]:
            log.warning("%s: a conversation that did not come through mindeval.engine.chat; sent as converted",
                        self.spec.model)
        s = self.spec
        timeout = kwargs.pop("timeout", None) or s.timeout
        try:
            text, meta = await llm.acall_messages(sent, s.model, s.api_base, s.temperature, s.max_tokens, timeout,
                                                  params=s.params, api_key=s.api_key, **kwargs)
        except Exception as e:
            if call is None:
                raise
            call["error"] = e  # chat re-raises it; the caller decides whether to retry
            raise _HandedBack(f"{s.model}: the call failed and goes back to mindeval") from None
        if call is not None:
            call.update(text=text, meta=meta)
        usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        message = SimpleNamespace(
            content=text, reasoning_content=meta.get("reasoning_content"),
            tool_calls=[{"id": c.get("id"), "type": "function",
                         "function": {"name": c.get("name"), "arguments": c.get("arguments")}}
                        for c in meta.get("tool_calls") or []])
        return SimpleNamespace(message=message, usage=SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens") or 0, completion_tokens=usage.get("completion_tokens") or 0))


def _same(built: list[dict], converted: list) -> bool:
    return len(built) == len(converted) and all(b.get("role") == getattr(c, "role", None)
                                                for b, c in zip(built, converted))


def _as_dicts(messages: list) -> list[dict]:
    out = []
    for m in messages:
        d = {"role": m.role, "content": m.content}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = m.tool_calls
        if getattr(m, "tool_call_id", None):
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


class Unused:
    """A role UserSim's engine requires a facade for and mindeval never calls (its judge, tool simulator and
    summarizer: a mindeval session has no in-loop judge and is never summarized)."""

    def __init__(self, role: str) -> None:
        self.model_name = f"unused/{role}"
        self._role = role

    async def acompletion(self, messages: list, **kwargs: Any):
        raise RuntimeError(f"mindeval does not call UserSim's {self._role}")


def facades(patient: CallSpec, counselor: CallSpec) -> dict[str, Any]:
    """The models UserSim's engine runs a mindeval session with, through litellm."""
    return session_models(Facade(patient), Facade(counselor))


class HandsBack:
    """A model UserSim built from a models file (`usersim.engine.external.model_facades`), wrapped so a failed call
    goes back to mindeval's own retry loop at once, as `Facade`'s do, instead of being retried inside `acall_llm`
    first. Data Designer's transport retries (module docstring) still come before it. It also sends the side's
    timeout when a call names none, and carries the reply's `finish_reason` and `usage`, which `acall_llm`'s reply
    leaves out, into the meta."""

    def __init__(self, model: Any, timeout: float | None = None) -> None:
        self.model, self.timeout = model, timeout

    @property
    def model_name(self) -> str:
        return self.model.model_name

    def __getattr__(self, name: str) -> Any:  # anything else UserSim reads from a model
        return getattr(self.model, name)

    async def acompletion(self, messages: list, **kwargs: Any) -> Any:
        call = _CALL.get()
        if kwargs.get("timeout") is None and self.timeout is not None:  # the side's timeout, as Facade sends it
            kwargs["timeout"] = self.timeout
        try:
            response = await self.model.acompletion(messages, **kwargs)
        except Exception as e:
            if call is None:
                raise
            call["error"] = e  # chat re-raises it; the caller decides whether to retry
            raise _HandedBack(f"{self.model_name}: the call failed and goes back to mindeval") from None
        if call is not None:  # acall_llm's reply leaves it out; mindeval's cut-off check reads it
            call["finish_reason"] = response.choices[0].finish_reason if response.choices else None
            u = response.usage  # also left out of the reply; litellm's names, as the traces record them
            if u is not None:
                call["usage"] = {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                                 "total_tokens": u.total_tokens, "reasoning_tokens": u.reasoning_tokens}
        return response


def adapter_of(models: dict[str, Any], role: str) -> str:
    """How `role`'s calls reach its model, as a session's setup records it: `usersim` for a model built from a
    models file (`HandsBack`), `litellm` for mindeval's own `Facade`."""
    return "usersim" if isinstance(models.get(role), HandsBack) else "litellm"


def session_models(patient: Any, counselor: Any) -> dict[str, Any]:
    """The models dict UserSim's engine needs for a mindeval session: the patient and the counselor, and the roles
    mindeval never calls. Either side can be a `Facade` or a model UserSim builds from a models file
    (`usersim.engine.external.model_facades`)."""
    return {PATIENT: patient, COUNSELOR: counselor,
            **{role: Unused(role) for role in (MODEL_JUDGE, MODEL_API_RESPONSE, MODEL_SUMMARY)}}
