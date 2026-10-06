"""arc.py: compact_log/evolve_situation retry backoff, and _trim_log's overshoot handling."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from conftest import _response, no_pause, scripted

import mindeval.arc as arc_mod
from mindeval import engine, llm
from mindeval.config import CallSpec, load_config
from mindeval.models import MemberRow

ROOT = Path(__file__).resolve().parents[1]
PATIENT = CallSpec(model="patient", api_base=None, temperature=1.0, max_tokens=None, timeout=5, max_retries=1)
MODELS = engine.facades(PATIENT, PATIENT)


def _compact(*args, **kw):
    return asyncio.run(arc_mod.compact_log(*args, **kw))


def _evolve(*args, **kw):
    return asyncio.run(arc_mod.evolve_situation(*args, **kw))


def _recording(pauses):
    async def pause(attempt):
        pauses.append(attempt)
    return pause


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
    monkeypatch.setattr(arc_mod, "_pause", _recording(pauses))
    calls = {"n": 0}

    def completion(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("compaction server down")
        return _response("- ok\nSummary: They said hi.")

    monkeypatch.setattr(llm, "_LIB", scripted(completion))
    session = SimpleNamespace(member_messages=[{"role": "assistant", "content": "hi"}])
    entries, summary = _compact([], session, MODELS, max_entries=5, arc_seed=0, arc_id="a", episode_id=0)
    assert entries == ["ok"] and summary == "They said hi."
    assert pauses == [1, 2]  # backoff before the 2nd and 3rd attempts, none after the last one


def test_compact_log_keeps_the_newest_entries_of_an_overshooting_log(monkeypatch):
    """compact_log itself, not just _trim_log: a model that returns more entries than max_entries must
    not lose the session that just ended (the newest entries)."""
    log = "\n".join(f"- entry {i}" for i in range(5)) + "\n\nSummary: Five things happened."
    monkeypatch.setattr(llm, "_LIB", scripted(lambda **kw: _response(log)))
    session = SimpleNamespace(member_messages=[{"role": "assistant", "content": "hi"}])
    entries, summary = _compact([], session, MODELS, max_entries=3, arc_seed=0, arc_id="a", episode_id=0)
    assert entries == ["entry 2", "entry 3", "entry 4"] and summary == "Five things happened."


def test_patient_calls_between_sessions_strip_an_inline_reasoning_trace(monkeypatch):
    """A patient served without a reasoning parser returns "trace</think>answer" as its content, as
    the member calls already handle: compaction must not read the trace as a log line, and evolution
    must not take a JSON object drafted inside the trace for the answer."""
    monkeypatch.setattr(arc_mod, "_pause", no_pause)
    monkeypatch.setattr(llm, "_LIB", scripted(lambda **kw: _response(
        "Okay, the member said hi.\nSummary: a draft one.\n</think>\n- Member said hi.\nSummary: Member said hi.")))
    session = SimpleNamespace(member_messages=[{"role": "assistant", "content": "hi"}])
    assert _compact([], session, MODELS, max_entries=5, arc_seed=0, arc_id="a",
                    episode_id=0) == (["Member said hi."], "Member said hi.")

    fields = {"automatic_thoughts": ["t"], "behaviors": ["b"], "physical": [], "goal": None, "checkpoints": [],
              "opening_message": "hi again"}
    draft, final = {**fields, "event": "draft"}, {**fields, "event": "The week went on."}
    monkeypatch.setattr(llm, "_LIB", scripted(lambda **kw: _response(
        f"Draft: {json.dumps(draft)}\n</think>\n{json.dumps(final)}")))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    situation = _evolve(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                                         2.0, MODELS, arc_seed=0, arc_id="a", episode_id=1, now_line="now",
                                         beat=beat, log=[], knobs=row.knobs)
    assert situation.event == "The week went on."


def test_evolve_situation_retries_with_backoff(monkeypatch):
    pauses = []
    monkeypatch.setattr(arc_mod, "_pause", _recording(pauses))
    calls = {"n": 0}
    next_situation = {"event": "e", "automatic_thoughts": ["t"], "behaviors": ["b"], "physical": [],
                      "goal": None, "checkpoints": [], "opening_message": "hi again"}

    def completion(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("patient server down")
        return _response(json.dumps(next_situation))

    monkeypatch.setattr(llm, "_LIB", scripted(completion))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    situation = _evolve(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                                         2.0, MODELS, arc_seed=0, arc_id="a", episode_id=1, now_line="now",
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

    monkeypatch.setattr(llm, "_LIB", scripted(completion))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    log = ["Asked in session 1 to be called Sam from now on.", "[open] the letter to the landlord"]
    guarded = row.knobs.model_copy(update={"concealment_propensity": "high", "session_affect": "fear"})
    open_ = row.knobs.model_copy(update={"concealment_propensity": "low", "session_affect": "joy"})
    for knobs in (guarded, open_):
        _evolve(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                                 2.0, MODELS, arc_seed=0, arc_id="a", episode_id=2, now_line="now", beat=beat,
                                 log=log, knobs=knobs)
    assert all(entry in p for p in prompts for entry in log)
    for prompt, knobs in zip(prompts, (guarded, open_)):
        archetype_text, avoids, conduct = arc_mod.session_conduct(cfg, row, knobs)
        assert all(line in prompt for line in conduct + avoids)
        assert archetype_text is None or archetype_text in prompt
    assert prompts[0] != prompts[1]

    _evolve(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]),
                             2.0, MODELS, arc_seed=0, arc_id="a", episode_id=1, now_line="now", beat=beat,
                             log=[], knobs=guarded)
    assert "Their log so far" not in prompts[-1]  # no log yet: no empty section


# -- session summaries: written with the log, and every earlier one feeds the next situation ----------------------

def test_the_summary_is_what_follows_the_last_summary_line():
    log, summary = arc_mod.split_summary("- a thing\n- another\n\n## Summary:\nThey came in.\nThey left calmer.\n")
    assert arc_mod.parse_log(log) == ["a thing", "another"] and summary == "They came in. They left calmer."
    for bad in ("- a thing\n", "- a thing\nSummary:   \n"):
        try:
            arc_mod.split_summary(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass


def test_evolve_situation_sees_every_earlier_session_summary(monkeypatch):
    """Session k's situation is written from every earlier session's summary, the ones long gone from the log
    (j < k - 2) included, each with how long ago it was; with more than the previous one, it is asked to keep a
    thread from an earlier one alive."""
    prompts = []
    next_situation = {"event": "e", "automatic_thoughts": ["t"], "behaviors": ["b"], "physical": [],
                      "goal": None, "checkpoints": [], "opening_message": "hi again"}

    def completion(**kw):
        prompts.append(kw["messages"][0]["content"])
        return _response(json.dumps(next_situation))

    monkeypatch.setattr(llm, "_LIB", scripted(completion))
    cfg = load_config()
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    beat = next(iter(cfg.episode.episode_beats.vocabulary))
    summaries = [("about 3 weeks", "Member decided to call the landlord about the leak."),
                 ("about 2 weeks", "Member talked about their sister's visit."),
                 ("about 4 days", "Member was tired and kept it short.")]
    for given in (summaries, summaries[-1:], []):
        _evolve(cfg, row, row.situation, SimpleNamespace(agenda=None, member_messages=[]), 2.0, MODELS,
                arc_seed=0, arc_id="a", episode_id=len(given), now_line="now", beat=beat, log=[], knobs=row.knobs,
                summaries=given)
    every, only_previous, none = prompts
    assert "- Conversation 1, about 3 weeks ago: Member decided to call the landlord about the leak." in every
    assert all(text in every for _, text in summaries) and "an earlier conversation than the" in every
    assert "Conversation 1, about 4 days ago" in only_previous and "an earlier conversation than the" not in only_previous
    assert "Every conversation they have had" not in none


def test_an_arc_carries_each_sessions_summary_into_every_later_situation(tmp_path, monkeypatch):
    """Three sessions: the third session's situation is written from the first session's summary as well as the
    second's; every summary is kept beside its session, on arc.json, and through a resume."""
    from mindeval.probe import Runner
    from mindeval.session import Counselor

    evolve_prompts = []

    def completion(**kw):
        first = kw["messages"][0]["content"]
        if any(t["function"]["name"] == "commit_move" for t in kw.get("tools") or []):
            return _response(tool_call=json.dumps({"reasoning": "ok", "move": "do_answer_question"}))
        if kw["model"] == "openai/counselor":
            return _response("How has it been?")
        if "running log" in first:
            n = first.count("- Member talked")  # the prior log grows by one entry a session: each summary its own
            log = "\n".join(f"- Member talked ({i})." for i in range(n + 1))
            return _response(f"{log}\nSummary: Session {n + 1}, about the landlord.")
        if "Answer with JSON only" in first:
            evolve_prompts.append(first)
            return _response(json.dumps({"event": "e", "automatic_thoughts": ["t"], "behaviors": ["b"],
                                         "physical": [], "goal": "g", "checkpoints": ["c"], "opening_message": "hi"}))
        return _response("fine")

    monkeypatch.setattr(llm, "_LIB", scripted(completion))
    row = next(MemberRow.model_validate_json(line) for line in
              (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)
    spec = CallSpec(model="openai/counselor", api_base=None, temperature=1.0, max_tokens=None, timeout=5, max_retries=1)
    counselor = Counselor.from_text(spec, (ROOT / "examples/counselor_system.j2").read_text())
    arc = asyncio.run(arc_mod.run_arc(load_config(), row, counselor, runner=Runner(PATIENT, spec, max_turns=2),
                                      patient=PATIENT, sessions=3, max_turns=2, arc_seed=1, out=tmp_path))
    summaries = [e.summary for e in arc.episodes]
    assert all(summaries) and len(set(summaries)) == 3
    assert [(tmp_path / f"ep{i:03d}" / "summary.md").read_text().strip() for i in range(3)] == summaries
    assert summaries[0] in evolve_prompts[1] and summaries[1] in evolve_prompts[1]  # session 3 sees session 1's
    assert summaries[0] in evolve_prompts[0] and summaries[1] not in evolve_prompts[0]
    assert json.loads((tmp_path / "arc.json").read_text())["episodes"][0]["summary"] == summaries[0]
