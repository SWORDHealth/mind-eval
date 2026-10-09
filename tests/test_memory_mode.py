"""`--memory_mode`: what each side carries between sessions.

One member through three sessions against a scripted model whose every answer is a hash of what it was sent, so a
change to any input anywhere shows in the files. The clock is a counter, so the files are deterministic.

  - compacted, the default, reproduces the files the harness wrote before the flag existed, byte for byte
    (`fixtures/memory_golden/`, written by `regen_memory_golden.py` at c85e278);
  - none gives the counselor its session-1 text in every session and leaves the member exactly as in compacted;
  - full gives both sides the earlier sessions verbatim instead of the log;
  - session 1 is identical everywhere, and so is the situation each member walks into session 2 with.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from jsonargparse import CLI

from mindeval import arc as arc_module
from mindeval import llm
from mindeval.scripts import generate_interactions

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "fixtures" / "memory_golden"
COUNSELOR = "openai/counselor"
#: The files compared byte for byte (run.json carries wall-clock times and is left out).
FILES = ("arc.json", "ep000/trace.jsonl", "ep000/log.md", "ep000/situation.yaml", "ep001/trace.jsonl",
         "ep001/log.md", "ep001/situation.yaml", "ep002/trace.jsonl", "ep002/log.md", "ep002/situation.yaml")


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def _response(content=None, tool_call=None):
    calls = None
    if tool_call is not None:
        calls = [SimpleNamespace(id="call_1", function=SimpleNamespace(name="commit_move", arguments=tool_call))]
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="fake", id="r")


def _fake(calls: list):
    """Every answer is derived from the request, so equal files mean equal inputs."""
    def completion(**kw):
        msgs = kw["messages"]
        first = msgs[0]["content"]
        calls.append({"model": kw["model"], "messages": msgs})
        if isinstance(kw.get("tool_choice"), dict):
            return _response(tool_call=json.dumps({"reasoning": f"r{_h(msgs)}", "move": "do_answer_question"}))
        if kw["model"] == COUNSELOR:
            return _response(f"Counselor {_h(first)} {_h(msgs[1:])}: how was it?")
        if "running log" in first:
            return _response(f"- Member said {_h(first)}.\n- [open] Member will try {_h(first)[:4]}.")
        if "case notes" in first:
            return _response(f"- Member says {_h(first)}.")
        if "what a person remembers" in first:
            return _response(f"- I said {_h(first)}.")
        if "Answer with JSON only" in first:
            return _response(json.dumps({"event": f"event {_h(first)}", "automatic_thoughts": ["same"],
                                         "behaviors": ["stayed in"], "physical": [], "goal": "talk",
                                         "checkpoints": ["the money"], "opening_message": f"hi again {_h(first)}"}))
        return _response(f"member {_h(msgs)}")
    return completion


def _env(tmp: Path, monkeypatch) -> list:
    tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp)
    for name in [k for k in os.environ if k.startswith("MINDEVAL_")]:
        monkeypatch.delenv(name)
    for name, value in {"PATIENT_MODEL": "hosted_vllm/patient", "PATIENT_API_BASE": "http://patient.invalid/v1",
                        "PATIENT_API_KEY": "patient-key", "COUNSELOR_MODEL": COUNSELOR,
                        "COUNSELOR_API_KEY": "counselor-key"}.items():
        monkeypatch.setenv(f"MINDEVAL_{name}", value)
    calls: list = []
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=_fake(calls)))
    clock = iter(range(10 ** 6))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock) / 1000)
    return calls


def produce(tmp: Path, monkeypatch, mode: str | None = None) -> tuple[Path, list]:
    """Run the CLI for one member x three sessions; returns the arc directory and every model call."""
    calls = _env(tmp, monkeypatch)
    out = tmp / "run"
    argv = ["--output_dir", str(out), "--members", "1", "--sessions", "3", "--max_turns", "3", "--max_workers", "1"]
    if mode is not None:
        argv += ["--memory_mode", mode]
    assert CLI([generate_interactions.main], as_positional=False, args=argv) == 0
    member = json.loads((out / "run.json").read_text())["member_ids"][0]
    return out / "members" / member / "arc", calls


def _session(arc: Path, ep: int) -> dict:
    return json.loads((arc / f"ep{ep:03d}" / "trace.jsonl").read_text().splitlines()[-1])


def test_compacted_is_the_harness_as_it_was(tmp_path, monkeypatch):
    for mode in (None, "compacted"):
        arc, _ = produce(tmp_path / str(mode), monkeypatch, mode)
        for name in FILES:
            assert (arc / name).read_bytes() == (GOLDEN / name).read_bytes(), f"{mode}: {name} differs"


def test_modes_change_only_the_counselor_memory(tmp_path, monkeypatch):
    arcs = {m: produce(tmp_path / m, monkeypatch, m)[0] for m in ("compacted", "none", "full")}
    first = {m: (a / "ep000" / "trace.jsonl").read_bytes() for m, a in arcs.items()}
    assert first["none"] == first["compacted"] == first["full"]  # session 1 is the same everywhere
    for m, a in arcs.items():  # the member walks into session 2 with the same situation in every mode
        assert (a / "ep001" / "situation.yaml").read_bytes() == (arcs["compacted"] / "ep001" / "situation.yaml").read_bytes()
    # none changes nothing on the member's side: its session-2 prompt is compacted's
    assert _session(arcs["none"], 1)["member_system_prompt"] == _session(arcs["compacted"], 1)["member_system_prompt"]
    s1_prompt = _session(arcs["compacted"], 0)["counselor_system_prompt"]
    for ep in (1, 2):
        assert _session(arcs["none"], ep)["counselor_system_prompt"] == s1_prompt
        log = (arcs["compacted"] / f"ep{ep - 1:03d}" / "log.md").read_text().strip()
        assert log in _session(arcs["compacted"], ep)["counselor_system_prompt"]
    for side, field in (("counsellor", "counselor_system_prompt"), ("member", "member_system_prompt")):
        full = _session(arcs["full"], 2)[field]
        assert "Session 1" in full and "Session 2" in full, side
        for ep in (0, 1):  # every earlier session, verbatim, oldest first
            for msg in _session(arcs["full"], ep)["member_messages"]:
                assert msg["content"] in full, side
        assert full.index("Session 1") < full.index("Session 2")
        assert (arcs["full"] / "ep001" / "log.md").read_text().strip() not in full  # the transcripts replace the log
        assert (arcs["full"] / "ep001" / "log.md").is_file()  # compaction still runs
    assert "# Memory" in _session(arcs["full"], 2)["counselor_system_prompt"]
    assert "# What you remember" in _session(arcs["full"], 2)["member_system_prompt"]


def test_full_memory_keeps_the_latest_sessions_under_the_cap():
    sessions = [SimpleNamespace(member_messages=[{"role": "assistant", "content": f"s{i} " + "x" * 50},
                                                 {"role": "user", "content": f"c{i}"}]) for i in range(5)]
    text = arc_module.full_memory(sessions, "counsellor", max_chars=300)  # the intro and two sessions
    assert "s4 " in text and "s3 " in text and "s0 " not in text
    assert "3 earlier sessions are not included" in text
    assert arc_module.full_memory(sessions[:1], "member", max_chars=10 ** 6).count("Session 1") == 1


def test_split_gives_each_side_its_own_log(tmp_path, monkeypatch):
    arc, _ = produce(tmp_path / "split", monkeypatch, "split")
    for ep in (1, 2):
        notes = (arc / f"ep{ep - 1:03d}" / "log_counselor.md").read_text().strip()
        mine = (arc / f"ep{ep - 1:03d}" / "log_member.md").read_text().strip()
        assert notes.startswith("- Member says") and mine.startswith("- I said")
        session = _session(arc, ep)
        assert notes in session["counselor_system_prompt"] and mine not in session["counselor_system_prompt"]
        assert mine in session["member_system_prompt"] and notes not in session["member_system_prompt"]
        assert (arc / f"ep{ep - 1:03d}" / "log.md").is_file()  # the shared log is still kept


def test_fork_resumes_after_session_one(tmp_path, monkeypatch):
    """analysis/fork_run.py: session 1 copied, session 2's situation the source's, then any mode."""
    import sys
    sys.path.insert(0, str(ROOT.parent / "analysis"))
    import fork_run

    source, _ = produce(tmp_path / "source", monkeypatch, None)
    src_run = source.parents[2]
    member = source.parent.name
    (tmp_path / "ids.txt").write_text(member + "\n")
    for mode in ("none", "split"):
        _env(tmp_path / f"fork_{mode}", monkeypatch)
        out = tmp_path / f"fork_{mode}" / "run"
        argv = ["fork_run.py", "--src", str(src_run), "--members_file", str(tmp_path / "ids.txt"), "--",
                "--output_dir", str(out), "--sessions", "3", "--max_turns", "3", "--max_workers", "1",
                "--memory_mode", mode]
        monkeypatch.setattr(sys, "argv", argv)
        assert fork_run.main() == 0
        arc = out / "members" / member / "arc"
        for name in ("ep000/trace.jsonl", "ep000/log.md", "ep000/situation.yaml", "ep001/situation.yaml"):
            assert (arc / name).read_bytes() == (source / name).read_bytes(), f"{mode}: {name}"
        assert json.loads((arc / "arc.json").read_text())["arc_id"] == json.loads((source / "arc.json").read_text())["arc_id"]
        if mode == "none":  # the member walks into session 2 exactly as in the source run
            assert _session(arc, 1)["member_system_prompt"] == _session(source, 1)["member_system_prompt"]
            assert _session(arc, 1)["counselor_system_prompt"] == _session(source, 0)["counselor_system_prompt"]
        else:  # split catches each side's log up on the forked session 1
            assert (arc / "ep000" / "log_counselor.md").read_text().strip() in _session(arc, 1)["counselor_system_prompt"]
            assert (arc / "ep000" / "log_member.md").read_text().strip() in _session(arc, 1)["member_system_prompt"]
        assert json.loads((out / "fork.json").read_text())["members"] == [member]
