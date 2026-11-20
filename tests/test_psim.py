"""The patient-sim protocol (mindeval/psim.py) against a scripted server. It needs the model's harness directory, which
is not part of this repository: MINDEVAL_PSIM_DIR, or the cluster's copy; skipped without it."""

import datetime as dt
import json
import os
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion

from mindeval import psim
from mindeval.config import CallSpec, load_config
from mindeval.models import MemberRow

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(os.environ.get("MINDEVAL_PSIM_DIR",
                              "/mnt/data/shared/models/PatientSim-Qwen3.8-27B-RL-2910-s300/harness"))
pytestmark = pytest.mark.skipif(not (HARNESS / "patient_sim.py").is_file(), reason="no patient-sim harness here")
ROW = MemberRow.model_validate_json((ROOT / "data/profiles.jsonl").read_text().splitlines()[0])
SPEC = CallSpec(model="hosted_vllm/patient-sim", api_base="http://fake/v1", temperature=1.0, max_tokens=8192,
                timeout=60, max_retries=3, params={"top_p": 1.0, "extra_body": {"top_k": -1}},
                protocol=psim.PROTOCOL, protocol_dir=str(HARNESS))


def _completion(content=None, calls=(), reasoning="thinking it over"):
    message = {"role": "assistant", "content": content, "reasoning": reasoning}
    if calls:
        message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}
                                 for i, (name, args) in enumerate(calls)]
    return ChatCompletion.model_validate({"id": "r", "object": "chat.completion", "created": 0, "model": "patient-sim",
                                          "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})


INTENT = {"moves": ["answer_question", "share"], "time_elapsed": "00:02:00", "sentiments": ["tired"]}


class Server:
    """Answers each request from a script of completions (or exceptions), and keeps the requests."""

    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        out = self.script.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _member(monkeypatch, script, **kw):
    server = Server(script)
    monkeypatch.setattr(psim, "_client", lambda *a: type("C", (), {"chat": type("X", (), {
        "completions": type("Y", (), {"create": staticmethod(server.create)})})})())
    monkeypatch.setattr(psim.time, "sleep", lambda s: None)
    start = dt.datetime(2026, 11, 5, 21, 0, tzinfo=dt.timezone.utc)
    member = psim.build_member(load_config(), ROW, ROW.situation, ROW.knobs, SPEC, start=start, **kw)
    return member, server


def test_prompt_carries_the_member_in_the_model_layout():
    cfg = load_config()
    ps = psim.load_protocol(str(HARNESS))
    prompt, parts = psim.render_system_prompt(cfg, ROW, ROW.situation, ROW.knobs, ps, memory_text="- Member said x.",
                                              earlier=[dt.datetime(2026, 11, 3, 20, tzinfo=dt.timezone.utc)])
    assert prompt.startswith(ps.render_role()) and ps.OPENING_PARAGRAPH in prompt
    sections = ["# Who you are", "# What is true about your life", "# What has happened, up to now", ps.PLAN_HEADING]
    assert [prompt.index(h) for h in sections] == sorted(prompt.index(h) for h in sections)
    assert ROW.formulation.strip().replace(" —", ",") in prompt and "- Member said x." in prompt
    assert "Tue 2026-11-03" in prompt and ROW.situation.event.replace(" —", ",") in prompt
    assert ROW.situation.goal.replace(" —", ",") in prompt and "`do_" not in prompt and "commit_move" not in prompt
    assert f"Name: {ROW.formulation.split(' is ')[0]}" in parts["profile"]


def test_a_session_opening_then_two_turns(monkeypatch):
    member, server = _member(monkeypatch, [
        _completion(calls=[("expose_intent", {**INTENT, "time_elapsed": "00:00:00"})]),   # the opening's intent
        _completion(calls=[("expose_intent", INTENT)]), _completion("it was a long week"),
        _completion(calls=[("expose_intent", INTENT), ("end_chat", {"reason": "need to sleep"})]),
        _completion("ok night"),
    ], memory_text=None, earlier=[])
    _, meta, flags = member.open(ROW.situation.opening_message, seed=1)
    assert meta["intent"]["time_elapsed"] == "00:00:00" and not flags
    assert member.messages[1]["content"].startswith("<developer>[Thu 2026-11-05 21:00 +00:00]")
    assert member.messages[-1] == {"role": "assistant", "content": ROW.situation.opening_message}
    words, intent, end, *_ = member.speak("How was the week?", seed=2)
    assert words == "it was a long week" and intent == INTENT and end is None
    assert member.messages[-4]["content"] == "[Thu 2026-11-05 21:01 +00:00] How was the week?"
    assert member.messages[-2] == {"role": "tool", "tool_call_id": "call_0", "content": "Recorded."}
    assert member.messages[-1]["reasoning"] == "thinking it over"  # reasoning stays in the history
    words, intent, end, _, arrived, sent, *_ = member.speak("Sounds tiring.", seed=3)
    assert words == "ok night" and end == {"reason": "need to sleep"}
    assert arrived == member.start + dt.timedelta(minutes=4) and sent == arrived + dt.timedelta(minutes=2)
    req = server.requests[-1]
    assert req["model"] == "patient-sim" and req["tool_choice"] == "auto" and req["top_p"] == 1.0
    assert req["extra_body"] == {"top_k": -1} and req["max_tokens"] == 8192 and "seed" in req


def test_dirty_words_are_retried_then_replaced(monkeypatch):
    dirty = _completion("**Honestly** it was fine")
    member, _ = _member(monkeypatch, [_completion(calls=[("expose_intent", INTENT)]), dirty] * 3,
                        memory_text=None, earlier=[])
    member.messages.append({"role": "assistant", "content": "hi"})
    words, *_, records, flags = member.speak("How are you?", seed=1)
    assert words == "i don't know" and flags == ["utterance_hygiene", "member_retry"] and len(records) == 3
    assert member.messages[-1]["content"] == "i don't know"
    assert records[-1]["hygiene_dropped"] == "**Honestly** it was fine"


def test_an_overflowing_history_drops_the_oldest_reasoning(monkeypatch):
    from openai import BadRequestError
    overflow = BadRequestError("This model's maximum context length is 65536 tokens.", response=type(
        "R", (), {"status_code": 400, "headers": {}, "request": None})(), body=None)
    member, _ = _member(monkeypatch, [overflow, _completion(calls=[("expose_intent", INTENT)]), _completion("fine")],
                        memory_text=None, earlier=[])
    member.messages += [{"role": "assistant", "content": "a", "reasoning": "old"},
                        {"role": "user", "content": "b"}, {"role": "assistant", "content": "c", "reasoning": "new"}]
    words, *_, flags = member.speak("And then?", seed=1)
    assert words == "fine" and flags == ["reasoning_trimmed", "member_retry"]
    assert "reasoning" not in member.messages[1] and member.messages[3]["reasoning"] == "new"
