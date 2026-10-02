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
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from usersim.engine.config import MODEL_API_RESPONSE, MODEL_ASSISTANT, MODEL_JUDGE, MODEL_SUMMARY, MODEL_USER
from usersim.engine.core.llm import acall_llm

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
        reply = await acall_llm(models, role, messages, **kwargs)
    finally:
        _CALL.reset(token)
    if "meta" in call:
        return call["text"], call["meta"]
    return reply.get("content") or "", _meta_of(reply)  # answered by a facade that is not ours: a hosted run


def _meta_of(reply: dict) -> dict:
    """response_meta's fields, from `acall_llm`'s reply when no litellm response is at hand."""
    calls = [{"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
              "arguments": (c.get("function") or {}).get("arguments")} for c in reply.get("tool_calls") or []]
    meta = {"reasoning_content": reply.get("reasoning_content"), "tool_calls": calls}
    return {k: v for k, v in meta.items() if v}


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
        text, meta = await llm.acall_messages(sent, s.model, s.api_base, s.temperature, s.max_tokens, timeout,
                                              params=s.params, api_key=s.api_key, **kwargs)
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
    """The models UserSim's engine runs a mindeval session with."""
    return {PATIENT: Facade(patient), COUNSELOR: Facade(counselor),
            **{role: Unused(role) for role in (MODEL_JUDGE, MODEL_API_RESPONSE, MODEL_SUMMARY)}}
