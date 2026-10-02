"""Member/build_member: unit-level regression tests for the member-side call machinery, driven
against a scripted litellm exactly like test_smoke.py's."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import _response

import mindeval.member as member_mod
from mindeval import llm
from mindeval.config import CallSpec, load_config
from mindeval.member import Member, build_member, move_tool
from mindeval.models import MemberRow

ROOT = Path(__file__).resolve().parents[1]


def _spec(**over):
    kw = {"model": "x", "api_base": None, "temperature": None, "max_tokens": None, "timeout": 5,
         "max_retries": 2, "force_move": False}
    kw.update(over)
    return CallSpec(**kw)


def _member(**over):
    return Member(system_prompt="be a member", spec=_spec(**over), tool=move_tool({"do_answer_question": "answer"}),
                 vocab=frozenset({"do_answer_question"}))


# -- speak(): attempt state must not leak across retries (member.py, MAX_STRAY_COMMITS) -----------

def test_speak_does_not_confuse_a_later_clean_attempt_with_an_earlier_stray_commit(monkeypatch):
    """A stray commit_move on attempt 1, once MAX_STRAY_COMMITS has already been reached this turn, is
    an ordinary retryable error (raise ValueError, not break); if attempt 2 then answers cleanly, the
    clean words must be kept — not mistaken for another stray commit because `stray` (or `hygiene`)
    leaked from attempt 1's dict-reset, which used to happen once per while-iteration, not per attempt."""
    monkeypatch.setattr(member_mod, "_pause", lambda attempt: None)
    calls = []

    def completion(**kw):
        n = len(calls) + 1
        calls.append(kw)
        if n <= 3:  # a stray commit_move: no words, an unwanted tool call
            return _response(tool_call=json.dumps({"reasoning": "again", "move": "do_answer_question"}),
                             call_id=f"call_{n}")
        return _response("I'm okay today.")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    m = _member(max_retries=2)
    # two while-iterations of a stray commit, each one attempt, bring strays to MAX_STRAY_COMMITS (2)
    said, records = m.speak(seed=0)
    assert said == "I'm okay today."
    assert len(calls) == 4  # not stuck retrying, or looping, past the clean answer
    assert not any(r["name"] == member_mod.STRAY_COMMIT for r in records[-1:])


# -- commit(): the content_json path must leave valid JSON in the convo history (member.py) --------

def test_commit_content_json_path_keeps_valid_json_in_the_convo(monkeypatch):
    """A server that ignores the forced function answers in prose; the prose becomes `content`, but
    the synthetic commit_move call appended to the convo must carry valid JSON regardless of whether
    the prose itself parses — Anthropic's history conversion parses a tool_use's `input` as JSON, and
    a later call in the session fails before any HTTP request is made if it cannot."""
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(
        completion=lambda **kw: _response("sure, let's go with do_answer_question I guess")))
    m = _member()
    move, call = m.commit(seed=0)
    assert move is None and call["via"] == "content_json"
    arguments = m.convo[-1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {}  # never the raw prose


def test_commit_content_json_path_with_an_embedded_move_reconstructs_clean_json(monkeypatch):
    prose = 'Sure -- {"reasoning": "answering it", "move": "do_answer_question"} is my move.'
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(prose)))
    m = _member()
    move, call = m.commit(seed=0)
    assert move is not None and move.move == "do_answer_question" and call["via"] == "content_json"
    arguments = m.convo[-1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"reasoning": "answering it", "move": "do_answer_question"}
    assert arguments != prose  # reconstructed JSON, never the surrounding prose


def test_content_json_arguments_no_longer_break_anthropics_history_conversion(monkeypatch):
    """The exact failure the review reproduced: litellm's real Anthropic transform raises
    'Failed to parse tool call arguments' when a tool call's arguments are prose. Checked against
    litellm's actual transform_request, not a stub, so this fails if the allow-list regresses."""
    anthropic_chat = pytest.importorskip("litellm.llms.anthropic.chat.transformation")
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(
        completion=lambda **kw: _response("sure, let's go with do_answer_question I guess")))
    m = _member()
    m.commit(seed=0)
    cfg = anthropic_chat.AnthropicConfig()
    out = cfg.transform_request(model="claude-sonnet-4-5", messages=m.convo, optional_params={},
                                litellm_params={}, headers={})
    assert out["messages"]  # transform_request must not raise AnthropicError on this history


# -- inline reasoning traces must be split off the patient's own calls, as the counselor path does -

def test_commit_and_speak_strip_an_inline_reasoning_trace(monkeypatch):
    """A chat template that ends the prompt with <think> starts the completion inside the trace; a
    server with no reasoning parser returns "trace</think>reply" as plain content. session.py's
    Counselor.respond splits this via llm.split_inline_trace; the patient's own calls (commit, speak)
    must do the same, not leave the trace as part of the move's content_json prose or the utterance."""
    def completion(**kw):
        if kw.get("tools") == [move_tool({"do_answer_question": "answer"})]:
            return _response("<think>picking a move</think>{\"reasoning\": \"ok\", "
                             "\"move\": \"do_answer_question\"}")
        return _response("<think>planning the reply</think>i'm doing fine")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    m = _member()
    move, call = m.commit(seed=0)
    assert move is not None and move.move == "do_answer_question"
    assert call["meta"].get("reasoning_content") == "picking a move"
    m.answer(m.accepted_text)
    said, records = m.speak(seed=1)
    assert said == "i'm doing fine"
    assert "<think>" not in said and "picking a move" not in said
    assert records[-1]["meta"].get("reasoning_content") == "planning the reply"


# -- build_member: an archetype's avoided moves must not still be named in the prose ----------------

def _moves(text):
    return set(re.findall(r"\bdo_[a-z_]+\b", text))


def _knob_texts(cfg):
    """Every knob line and the secondary note, as knobs.yaml and style2.yaml write them."""
    texts = {f"{knob}.{level}": text for knob in ("concealment_propensity", "frustration_reactivity", "session_affect")
             for level, text in getattr(cfg.knob_text, knob).items()}
    return {**texts, "secondary_note": cfg.style2.secondary_note}


def test_without_avoided_moves_takes_out_the_move_not_its_sentence():
    """The sentence a move sits in can also carry the session's affect or a move that is still
    allowed; only the avoided move goes, with the words that exist to introduce it."""
    cfg = load_config()
    drop = member_mod._without_avoided_moves
    # the affect's own definition stays; only the avoided move and its "when ..." go
    assert drop(cfg.knob_text.session_affect["sadness"], {"do_express_hopelessness"}) == (
        "When something does get to you this session, it lands as grief or heaviness — `do_express_sadness`. "
        "Most of what you say is still ordinary.")
    # one of two moves in a list: the other stays, and the verb agrees with it
    assert "`do_blame` is near to hand." in drop(cfg.knob_text.session_affect["anger"], {"do_express_frustration"})
    # a list loses its avoided items and keeps its own joint
    assert "You rarely reach for `do_express_frustration` or `do_question_process`, and when" in drop(
        cfg.knob_text.frustration_reactivity["low"], {"do_disagree", "do_resist"})
    # every move in a sentence avoided: the sentence closes up on what is left of it
    assert drop(cfg.knob_text.frustration_reactivity["medium"],
                {"do_express_frustration", "do_disagree", "do_resist"}).endswith(
        "before you come back. You rarely question the conversation itself.")
    assert "Reach readily for `do_question_process` when it is this conversation" in drop(
        cfg.knob_text.frustration_reactivity["high"], {"do_express_frustration", "do_disagree", "do_resist"})
    assert "simply stop: leaving mid-thread, with" in drop(cfg.knob_text.frustration_reactivity["high"],
                                                          {"do_disengage"})
    # parenthetical names go, and "So is" keeps the sentence it refers back to
    note = drop(cfg.style2.secondary_note, {"do_disengage", "do_deflect"})
    assert 'an "ok", an "idk" or letting the question go by are ordinary turns' in note
    assert "So is saying so when something the counsellor says does not fit you (do_disagree)" in note
    text = cfg.knob_text.session_affect["joy"]
    assert drop(text, set()) == text and drop(text, {"do_go_deeper"}) == text  # nothing to take out: untouched


def test_without_avoided_moves_on_every_archetype_and_knob_line():
    """Every archetype against every line it could be given, not just the 50 profiles' draws: no
    avoided move survives, every allowed move does, sentences naming no avoided move are untouched,
    and nothing is left dangling."""
    cfg = load_config()
    for name, archetype in cfg.archetypes.archetypes.items():
        avoided = set(archetype.avoided_moves)
        for key, text in _knob_texts(cfg).items():
            out = member_mod._without_avoided_moves(text, avoided)
            where = f"{name} x {key}: {out!r}"
            assert not _moves(out) & avoided, where
            assert _moves(text) - avoided <= _moves(out), where
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if not _moves(sentence) & avoided:
                    assert sentence in out, where
            assert not re.search(r"  |\s[,.;]|, ,|— [.,;]|^[^A-Z*`]|\s$", out), where


def test_build_member_never_names_an_avoided_move_in_the_prompt():
    """Every profile with an archetype: the tool enum already drops avoided moves (repertoire), and
    the rendered prompt must not still tell the member to reach for one by name."""
    cfg = load_config()
    rows = [MemberRow.model_validate_json(line)
           for line in (ROOT / "data/profiles.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    archetyped = [r for r in rows if r.archetype]
    assert archetyped  # the fixture data must actually exercise this
    for row in archetyped:
        member = build_member(cfg, row, row.situation, row.knobs, _spec())
        avoided = set(cfg.archetypes.archetypes[row.archetype].avoided_moves)
        for move in avoided:
            assert f"`{move}`" not in member.system_prompt and f"({move})" not in member.system_prompt, (
                f"{row.member_id} ({row.archetype}) still names avoided move {move!r} in its prompt")
