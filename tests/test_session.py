"""The counselor's own earlier turns go back as Anthropic accepts them: a turn with one thinking block exactly as
returned, a turn with several (Claude interleaving thinking and text) without its thinking."""

import json
from types import SimpleNamespace

import pytest
from conftest import _clear_env, _response, interactions

from mindeval import llm, session
from mindeval.config import CallSpec, load_config, provider_seed, turn_seed
from mindeval.member import CONTROL_FIELD_PROBLEM, FALLBACK_UTTERANCE, MemberError
from mindeval.models import MemberRow


def _history_sent(monkeypatch, transcript):
    sent = []
    monkeypatch.setattr(llm, "retry_text", lambda messages, spec, what: (sent.append(messages) or ("ok", {})))
    counselor = session.Counselor(spec=SimpleNamespace(model="vertex_ai/claude-opus-5"), template=None)
    counselor.respond("system prompt", transcript)
    return sent[0]


def _block(signature):
    return {"type": "thinking", "thinking": "", "signature": signature}


def test_only_a_multi_block_turn_goes_back_without_its_thinking(monkeypatch):
    """Found on Claude Opus 5: litellm merges an interleaved turn's text and lists its thinking blocks, then
    rebuilds it thinking-first, which Anthropic rejects as a modified turn. Every later counselor call of the
    session then failed with a 400."""
    # The transcript is the member's: its "user" turns are the counselor's own.
    transcript = [{"role": "assistant", "content": "hi"},
                  {"role": "user", "content": "one block", "thinking_blocks": [_block("a")]},
                  {"role": "assistant", "content": "ok"},
                  {"role": "user", "content": "two blocks", "thinking_blocks": [_block("b"), _block("c")]},
                  {"role": "assistant", "content": "and?"}]
    sent = _history_sent(monkeypatch, transcript)
    counselor_turns = [m for m in sent if m["role"] == "assistant"]
    assert counselor_turns[0]["thinking_blocks"] == [_block("a")]
    assert "thinking_blocks" not in counselor_turns[1]
    assert counselor_turns[1]["content"] == "two blocks"


def test_a_single_block_turn_reaches_anthropic_in_its_original_order(monkeypatch):
    """A one-block turn is carried because litellm rebuilds it exactly as Claude returned it: thinking, then text.
    Checked against litellm's real Anthropic transform, not a stub."""
    anthropic_chat = pytest.importorskip("litellm.llms.anthropic.chat.transformation")
    transcript = [{"role": "assistant", "content": "hi"},
                  {"role": "user", "content": "one block", "thinking_blocks": [_block("a")]},
                  {"role": "assistant", "content": "and?"}]
    sent = _history_sent(monkeypatch, transcript)
    out = anthropic_chat.AnthropicConfig().transform_request(model="claude-opus-5", messages=sent,
                                                             optional_params={}, litellm_params={}, headers={})
    turn = [m for m in out["messages"] if m["role"] == "assistant"][0]
    assert [block["type"] for block in turn["content"]] == ["thinking", "text"]
    assert turn["content"][0]["signature"] == "a"


# These integration fixtures are invented locally; no real or reserved profile data is loaded.
def _control_hygiene_row():
    return MemberRow.model_validate({
        "member_id": "hygiene_fixture", "formulation": "Alex works at a bookshop.",
        "knobs": {"concealment_propensity": "low", "frustration_reactivity": "low",
                  "session_affect": "fear", "verbosity": "low", "writing_style": "plain"},
        "situation": {"situation_id": "hygiene_fixture_s0", "event": "A shift was quiet.",
                      "automatic_thoughts": ["The quiet felt strange."], "behaviors": ["Went home."],
                      "opening_message": "Work has been quiet this week."},
    })


def test_control_leak_exhaustion_writes_native_error_evidence_without_spoken_turn(tmp_path, monkeypatch):
    import mindeval.member as member_mod

    monkeypatch.setattr(member_mod, "_pause", lambda attempt: None)
    monkeypatch.setattr(llm, "retry_text", lambda *args, **kwargs: ("How was your shift?", {}))
    requests = []
    dirty = "I've been okay.\n\nend_conversation: False\n  "

    def completion(**kw):
        requests.append(kw)
        if kw.get("tool_choice") == member_mod.FORCE_MOVE:
            return _response(tool_call=json.dumps({"reasoning": "answer", "move": "do_answer_question"}))
        return _response(dirty)

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    row = _control_hygiene_row()
    patient = CallSpec(model="hosted_vllm/patient", api_base=None, temperature=1, max_tokens=None,
                       timeout=5, max_retries=2, force_move=True)
    counselor = session.Counselor(spec=patient, template="{{ memory }}")
    trace_path = tmp_path / "trace.jsonl"
    with pytest.raises(MemberError):
        session.run_session(load_config(), row, row.situation, row.knobs, counselor, patient=patient,
                            max_turns=1, seed=123, trace_path=trace_path, counselor_memory="No earlier sessions.")
    turns, footer = session.read_trace(trace_path)
    assert [turn.turn for turn in turns] == [0]
    assert footer.termination == "error" and footer.error.startswith("MemberError:")
    assert footer.member_messages == [{"role": "assistant", "content": row.situation.opening_message}]
    evidence, = [json.loads(line) for line in trace_path.with_name(session.ERROR_FILENAME).read_text().splitlines()]
    failed = evidence["record"]
    assert failed["turn"] == 1 and failed["utterance"] == ""
    failure = failed["calls"][-1]
    assert failure["result"] is None
    expected = [provider_seed(turn_seed(123, footer.session_id, 1, "speak") + attempt) for attempt in (1, 2)]
    assert [call["seed"] for call in requests[1:]] == expected
    assert [attempt["request_seed"] for attempt in failure["hygiene_attempts"]] == expected
    assert all(attempt["text"] == dirty for attempt in failure["hygiene_attempts"])
    assert all(attempt["problems"] == [CONTROL_FIELD_PROBLEM] for attempt in failure["hygiene_attempts"])
    assert FALLBACK_UTTERANCE not in [message["content"] for message in footer.member_messages]


def test_control_leak_failed_cli_session_preserves_partial_and_resumes_cleanly(tmp_path, monkeypatch):
    import mindeval.member as member_mod

    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    monkeypatch.setenv("MINDEVAL_PATIENT_MODEL", "hosted_vllm/patient")
    monkeypatch.setenv("MINDEVAL_COUNSELOR_MODEL", "hosted_vllm/counselor")
    monkeypatch.setattr(member_mod, "_pause", lambda attempt: None)
    row = _control_hygiene_row()
    profiles = tmp_path / "profiles.jsonl"
    profiles.write_text(row.model_dump_json() + "\n")
    template = tmp_path / "counselor.j2"
    template.write_text("{{ memory }}")
    state = {"dirty": True, "speech_calls": []}
    dirty = "I've been okay. Work is quiet.\n\nend_conversation: False"

    def completion(**kw):
        if kw["model"] == "hosted_vllm/counselor":
            return _response("How was your shift?")
        if kw.get("tool_choice") == member_mod.FORCE_MOVE:
            return _response(tool_call=json.dumps({"reasoning": "answer", "move": "do_answer_question"}))
        if "running log" in kw["messages"][0]["content"]:
            return _response("- The member talked about a quiet shift.")
        state["speech_calls"].append(kw)
        return _response(dirty if state["dirty"] else "I went home after the shift.")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    out = tmp_path / "run"
    args = ["--output_dir", str(out), "--profiles_path", str(profiles), "--members", "1", "--sessions", "1",
            "--max_turns", "1", "--max_workers", "1", "--counselor_system_prompt_path", str(template)]
    assert interactions(*args) == 1
    arc_dir = out / "members" / row.member_id / "arc"
    failed_turns, failed_footer = session.read_trace(arc_dir / "ep000" / "trace.jsonl")
    assert len(failed_turns) == 1 and failed_footer.termination == "error"
    assert not (arc_dir / "arc.json").exists()
    assert not (arc_dir / "ep000" / "log.md").exists()
    run_before = json.loads((out / "run.json").read_text())
    first_seeds = [call["seed"] for call in state["speech_calls"]]
    assert len(first_seeds) == load_config().patient.max_retries
    state["dirty"] = False
    assert interactions(*args) == 0
    partial, = (arc_dir / "partial").iterdir()
    saved_turns, saved_footer = session.read_trace(partial / "trace.jsonl")
    assert len(saved_turns) == 1 and saved_footer.termination == "error"
    saved_error = json.loads((partial / session.ERROR_FILENAME).read_text())
    assert saved_error["record"]["calls"][-1]["hygiene_attempts"][0]["text"] == dirty
    clean_turns, clean_footer = session.read_trace(arc_dir / "ep000" / "trace.jsonl")
    assert clean_footer.termination == "max_turns" and clean_footer.session_id == failed_footer.session_id
    assert clean_turns[1].utterance == "I went home after the shift."
    assert state["speech_calls"][-1]["seed"] == first_seeds[0]
    assert all(not member_mod._has_control_field(message["content"]) for message in clean_footer.member_messages)
    run_after = json.loads((out / "run.json").read_text())
    assert run_after["fingerprint"] == run_before["fingerprint"]
    assert run_after["seed_namespace"] == run_before["seed_namespace"]
    assert (arc_dir / "arc.json").is_file()
    calls_before = len(state["speech_calls"])
    assert interactions(*args) == 0
    assert len(state["speech_calls"]) == calls_before
