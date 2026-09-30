"""arc.py: compact_log/evolve_situation retry backoff, and _trim_log's overshoot handling."""

import json
from pathlib import Path
from types import SimpleNamespace

import mindeval.arc as arc_mod
from mindeval import llm
from mindeval.config import CallSpec, load_config
from mindeval.models import MemberRow

ROOT = Path(__file__).resolve().parents[1]
PATIENT = CallSpec(model="patient", api_base=None, temperature=1.0, max_tokens=None, timeout=5, max_retries=1)


def _text_response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="x", id="r")


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
        return _text_response("- ok")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    session = SimpleNamespace(member_messages=[{"role": "assistant", "content": "hi"}])
    entries = arc_mod.compact_log([], session, PATIENT, max_entries=5, arc_seed=0, arc_id="a", episode_id=0)
    assert entries == ["ok"]
    assert pauses == [1, 2]  # backoff before the 2nd and 3rd attempts, none after the last one


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
        return _text_response(json.dumps(next_situation))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    situation = arc_mod.evolve_situation(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                                         2.0, PATIENT, arc_seed=0, arc_id="a", episode_id=1, now_line="now",
                                         beat=beat)
    assert situation.opening_message == "hi again"
    assert pauses == [1, 2]
