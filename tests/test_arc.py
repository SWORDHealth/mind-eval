"""arc.py: compact_log/evolve_situation retry backoff, and _trim_log's overshoot handling."""

import json
from pathlib import Path
from types import SimpleNamespace

from conftest import _response

import mindeval.arc as arc_mod
from mindeval import llm
from mindeval.config import CallSpec, load_config
from mindeval.models import MemberRow

ROOT = Path(__file__).resolve().parents[1]
PATIENT = CallSpec(model="patient", api_base=None, temperature=1.0, max_tokens=None, timeout=5, max_retries=1)


# -- _trim_log: keep the newest entries, never dropping an [open] one when it can be avoided --------

def test_trim_log_keeps_the_newest_entries():
    entries = [f"entry {i}" for i in range(5)]
    assert arc_mod._trim_log(entries, 3) == ["entry 2", "entry 3", "entry 4"]
    assert arc_mod._trim_log(entries, 10) == entries  # no overshoot: untouched


def test_trim_log_never_drops_an_open_entry_while_an_untagged_one_in_range_could_go_instead():
    entries = ["a", "[open] the one that matters", "b", "c", "d"]
    trimmed = arc_mod._trim_log(entries, 3)
    assert trimmed == ["[open] the one that matters", "c", "d"]  # oldest-first order kept


def test_trim_log_keeps_the_newest_opens_when_opens_alone_overshoot():
    entries = [f"[open] {i}" for i in range(5)]
    assert arc_mod._trim_log(entries, 3) == entries[-3:]


# -- compact_log / evolve_situation: retry with backoff, reusing member._pause's schedule -----------

def test_compact_log_retries_with_backoff(monkeypatch):
    pauses = []
    monkeypatch.setattr(arc_mod, "_pause", lambda attempt: pauses.append(attempt))
    calls = {"n": 0}

    def completion(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("compaction server down")
        return _response("- ok")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    session = SimpleNamespace(member_messages=[{"role": "assistant", "content": "hi"}])
    entries = arc_mod.compact_log([], session, PATIENT, max_entries=5, arc_seed=0, arc_id="a", episode_id=0)
    assert entries == ["ok"]
    assert pauses == [1, 2]  # backoff before the 2nd and 3rd attempts, none after the last one


def test_compact_log_keeps_the_newest_entries_of_an_overshooting_log(monkeypatch):
    """compact_log itself, not just _trim_log: a model that returns more entries than max_entries must
    not lose the session that just ended (the newest entries)."""
    log = "\n".join(f"- entry {i}" for i in range(5))
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(log)))
    session = SimpleNamespace(member_messages=[{"role": "assistant", "content": "hi"}])
    entries = arc_mod.compact_log([], session, PATIENT, max_entries=3, arc_seed=0, arc_id="a", episode_id=0)
    assert entries == ["entry 2", "entry 3", "entry 4"]


def test_patient_calls_between_sessions_strip_an_inline_reasoning_trace(monkeypatch):
    """A patient served without a reasoning parser returns "trace</think>answer" as its content, as
    the member calls already handle: compaction must not read the trace as a log line, and evolution
    must not take a JSON object drafted inside the trace for the answer."""
    monkeypatch.setattr(arc_mod, "_pause", lambda attempt: None)
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(
        "Okay, the member said hi.\nI should log that.\n</think>\n- Member said hi.")))
    session = SimpleNamespace(member_messages=[{"role": "assistant", "content": "hi"}])
    assert arc_mod.compact_log([], session, PATIENT, max_entries=5, arc_seed=0, arc_id="a",
                               episode_id=0) == ["Member said hi."]

    fields = {"automatic_thoughts": ["t"], "behaviors": ["b"], "physical": [], "goal": None, "checkpoints": [],
              "opening_message": "hi again"}
    draft, final = {**fields, "event": "draft"}, {**fields, "event": "The week went on."}
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(
        f"Draft: {json.dumps(draft)}\n</think>\n{json.dumps(final)}")))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    situation = arc_mod.evolve_situation(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                                         2.0, PATIENT, arc_seed=0, arc_id="a", episode_id=1, now_line="now",
                                         beat=beat, log=[], knobs=row.knobs)
    assert situation.event == "The week went on."


def test_evolve_situation_retries_with_backoff(monkeypatch):
    pauses = []
    monkeypatch.setattr(arc_mod, "_pause", lambda attempt: pauses.append(attempt))
    calls = {"n": 0}
    next_situation = {"event": "e", "automatic_thoughts": ["t"], "behaviors": ["b"], "physical": [],
                      "goal": None, "checkpoints": [], "opening_message": "hi again"}

    def completion(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("patient server down")
        return _response(json.dumps(next_situation))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    situation = arc_mod.evolve_situation(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                                         2.0, PATIENT, arc_seed=0, arc_id="a", episode_id=1, now_line="now",
                                         beat=beat, log=[], knobs=row.knobs)
    assert situation.opening_message == "hi again"
    assert pauses == [1, 2]


def test_evolve_situation_sees_the_log_and_the_next_sessions_conduct(monkeypatch):
    """The situation writer gets the whole running log, so a fact from session 1 that session 2 never came
    back to still shapes session 3; and the next session's archetype, concealment and affect, so its
    opening, spoken verbatim, can't give away more than the rest of that session is allowed to."""
    prompts = []
    next_situation = {"event": "e", "automatic_thoughts": ["t"], "behaviors": ["b"], "physical": [],
                      "goal": None, "checkpoints": [], "opening_message": "hi again"}

    def completion(**kw):
        prompts.append(kw["messages"][0]["content"])
        return _response(json.dumps(next_situation))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    log = ["Asked in session 1 to be called Sam from now on.", "[open] the letter to the landlord"]
    guarded = row.knobs.model_copy(update={"concealment_propensity": "high", "session_affect": "fear"})
    open_ = row.knobs.model_copy(update={"concealment_propensity": "low", "session_affect": "joy"})
    for knobs in (guarded, open_):
        arc_mod.evolve_situation(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                                 2.0, PATIENT, arc_seed=0, arc_id="a", episode_id=2, now_line="now", beat=beat,
                                 log=log, knobs=knobs)
    assert all(entry in p for p in prompts for entry in log)
    for prompt, knobs in zip(prompts, (guarded, open_)):
        archetype_text, avoids, conduct = arc_mod.session_conduct(cfg, row, knobs)
        assert all(line in prompt for line in conduct + avoids)
        assert archetype_text is None or archetype_text in prompt
    assert prompts[0] != prompts[1]

    arc_mod.evolve_situation(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                             2.0, PATIENT, arc_seed=0, arc_id="a", episode_id=1, now_line="now", beat=beat,
                             log=[], knobs=guarded)
    assert "Their log so far" not in prompts[-1]  # no log yet: no empty section
