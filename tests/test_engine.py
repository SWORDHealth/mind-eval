"""engine.chat: every interaction call goes through UserSim's acall_llm, and arrives at litellm exactly as mindeval
built it — the counselor's carried reasoning, a tool result's name and a tool-call turn's None content included,
which acall_llm's own conversion would drop."""

import asyncio
from types import SimpleNamespace

import pytest
from usersim.engine.core import llm as usersim_llm

from mindeval import engine, llm
from mindeval.config import CallSpec

PATIENT = CallSpec(model="hosted_vllm/patient", api_base="http://patient.invalid/v1", temperature=1.0,
                   max_tokens=None, timeout=60, max_retries=1, api_key="patient-key",
                   params={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}})
COUNSELOR = CallSpec(model="openai/counselor", api_base=None, temperature=None, max_tokens=4096, timeout=30,
                     max_retries=1, params={"reasoning_effort": "high"})
ITEM = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}


def _litellm(monkeypatch, *responses):
    sent = []

    async def acompletion(**kw):
        sent.append(kw)
        return responses[len(sent) - 1]

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(acompletion=acompletion))
    return sent


def _response(content, calls=None, **extra):
    message = SimpleNamespace(content=content, tool_calls=calls, **extra)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="m", id="r",
                           usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 7, "completion_tokens": 3}))


def test_the_conversation_reaches_litellm_as_mindeval_built_it(monkeypatch):
    sent = _litellm(monkeypatch, _response("ok"))
    messages = [{"role": "system", "content": "be a counselor"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello", "reasoning_content": "greet them", "reasoning_items": [ITEM]},
                {"role": "user", "content": "and?"}]
    text, meta = asyncio.run(engine.chat(engine.facades(PATIENT, COUNSELOR), engine.COUNSELOR, messages, seed=5))
    (kw,) = sent
    assert kw["messages"] == messages  # the carried reasoning too, which acall_llm clears
    assert kw["model"] == "openai/counselor" and kw["max_tokens"] == 4096 and kw["timeout"] == 30
    assert kw["seed"] == 5 and kw["reasoning_effort"] == "high" and "temperature" not in kw
    assert text == "ok" and meta["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}


def test_a_patient_turn_keeps_its_tool_names_and_none_content(monkeypatch):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="commit_move", arguments='{"move": "x"}'))
    sent = _litellm(monkeypatch, _response(None, [call]))
    messages = [{"role": "system", "content": "be a member"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c0", "type": "function", "function": {"name": "commit_move", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c0", "name": "commit_move", "content": "accepted"},
                {"role": "assistant", "content": "fine"},
                {"role": "user", "content": "and?"}]
    tool = {"type": "function", "function": {"name": "commit_move", "parameters": {}}}
    choice = {"type": "function", "function": {"name": "commit_move"}}
    text, meta = asyncio.run(engine.chat(engine.facades(PATIENT, COUNSELOR), engine.PATIENT, messages, seed=1,
                                         tools=[tool], tool_choice=choice))
    (kw,) = sent
    assert kw["messages"] == messages
    assert kw["messages"][2]["content"] is None and kw["messages"][3]["name"] == "commit_move"
    assert kw["tools"] == [tool] and kw["tool_choice"] == choice and kw["api_key"] == "patient-key"
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert text == "" and meta["tool_calls"] == [{"id": "c1", "name": "commit_move", "arguments": '{"move": "x"}'}]


def test_meta_is_llm_response_meta_and_usersim_counts_the_call(monkeypatch):
    response = _response("<think>t</think>words", reasoning_content="r", reasoning_items=[ITEM])
    _litellm(monkeypatch, response)
    before = usersim_llm.get_call_stats().get(engine.PATIENT, {}).get("calls", 0)
    text, meta = asyncio.run(engine.chat(engine.facades(PATIENT, COUNSELOR), engine.PATIENT,
                                         [{"role": "user", "content": "hi"}]))
    assert (text, meta) == ("<think>t</think>words", llm.response_meta(response))
    assert usersim_llm.get_call_stats()[engine.PATIENT]["calls"] == before + 1


def test_a_reply_from_a_facade_that_is_not_ours_still_reads_as_meta():
    """A hosted run answers through its own facade: content, reasoning and tool calls come from acall_llm's reply."""
    class Host:
        model_name = "host/patient"

        async def acompletion(self, messages, **kw):
            calls = [{"id": "c9", "type": "function", "function": {"name": "commit_move", "arguments": "{}"}}]
            return SimpleNamespace(message=SimpleNamespace(content="", reasoning_content="why", tool_calls=calls))

    text, meta = asyncio.run(engine.chat({engine.PATIENT: Host()}, engine.PATIENT, [{"role": "user", "content": "x"}]))
    assert text == "" and meta == {"reasoning_content": "why",
                                   "tool_calls": [{"id": "c9", "name": "commit_move", "arguments": "{}"}]}


def test_the_roles_mindeval_never_calls_refuse(monkeypatch):
    monkeypatch.setattr(usersim_llm, "_MAX_RETRIES", 0)
    models = engine.facades(PATIENT, COUNSELOR)
    assert models[engine.PATIENT].model_name == "hosted_vllm/patient"
    with pytest.raises(RuntimeError, match="does not call"):
        asyncio.run(engine.chat(models, "judge_model", [{"role": "user", "content": "x"}]))
    assert {"user_model", "assistant_model", "judge_model", "api_response_model", "summary_model"} == set(models)
