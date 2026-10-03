"""End to end, offline: two members through two sessions against a scripted model, through the real
command line; the three ways a second invocation can go (skip, refuse, resume); then the MQM judge
over the run."""

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from conftest import _clear_env, _response, interactions, judgments

from mindeval import llm

ROOT = Path(__file__).resolve().parents[1]
ROWS = [json.loads(line) for line in (ROOT / "data/profiles.jsonl").read_text().splitlines()[:2]]
COUNSELOR, JUDGE = "openai/counselor", "openai/judge"
REPLY = "How has your week been?"  # a line separator must not break the trace
ITEM = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}  # a Responses API reasoning item
LOG_LINE = "Member talked about the week."
NEXT = {"event": "The week went on.", "automatic_thoughts": ["same as before"], "behaviors": ["stayed in"],
        "physical": [], "goal": "talk it through", "checkpoints": ["the money"]}
TEMPLATE = "Judge this reply.\n{conversation}\n\n{candidate}\n"
KEYS = {COUNSELOR: "counselor-key", JUDGE: "judge-key", "openai/typo": "judge-key", "openai/cutoff": "judge-key"}


NAMES = [r["formulation"].split(" is ", 1)[0] for r in ROWS]


def _opening(name):
    return f"hey, {name} again"


def _verdict(prompt):
    """Minor for the first member's replies, Major for the second's, so the mean and the p99 differ."""
    first = ROWS[0]["situation"]["opening_message"] in prompt or _opening(NAMES[0]) in prompt
    severity = "Minor" if first else "Major"
    return f'<think>weighing it</think>It asks again.\n{{"severity": "{severity}", "over_exploring": 1, "other": 0}}'


def _fake(state):
    def completion(**kw):
        first = kw["messages"][0]["content"]
        assert kw.get("api_key") == KEYS.get(kw["model"], "patient-key")
        if isinstance(kw.get("tool_choice"), dict):
            return _response(tool_call=json.dumps({"reasoning": "just answer", "move": "do_answer_question"}))
        if kw["model"] == COUNSELOR:
            state.setdefault("counselor_calls", []).append(kw["messages"])
            return _response(f"<think>plan the reply</think>{REPLY}", reasoning_items=[ITEM])
        if kw["model"] == "openai/cutoff":  # a well-formed verdict, but the answer ran into the token limit
            return _response(_verdict(first), finish="length")
        if kw["model"] == "openai/typo":
            raise RuntimeError("no such model")
        if kw["model"] == JUDGE:
            assert kw["api_base"] == "http://judge.invalid/v1" and len(kw["messages"]) == 1
            state.setdefault("judged", []).append(first)
            return _response(_verdict(first))
        if "running log" in first:
            if state.get("fail_compaction"):
                raise RuntimeError("compaction server down")
            return _response(f"- {LOG_LINE}")
        if "Answer with JSON only" in first:
            name = first.split("Who they are:\n\n", 1)[1].split(" is ", 1)[0]
            return _response(json.dumps({**NEXT, "opening_message": _opening(name)}))
        return _response("not really")
    return completion


def _args(out, members=1):
    return ["--output_dir", str(out), "--members", str(members), "--sessions", "2", "--max_turns", "3",
            "--max_workers", "1"]  # the default counselor template, examples/counselor_system.j2


def test_two_members_two_sessions_then_the_judge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no stray .env
    _clear_env(monkeypatch)  # nor anything exported in the shell
    for name, value in {"PATIENT_MODEL": "hosted_vllm/patient", "PATIENT_API_BASE": "http://patient.invalid/v1",
                        "PATIENT_API_KEY": "patient-key", "COUNSELOR_MODEL": COUNSELOR,
                        "COUNSELOR_API_KEY": "counselor-key"}.items():
        monkeypatch.setenv(f"MINDEVAL_{name}", value)
    state = {}
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=_fake(state)))
    out = tmp_path / "run"
    assert interactions(*_args(out, members=2)) == 0

    run = json.loads((out / "run.json").read_text())
    assert run["members"] == 2 and run["sessions"] == 2 and run["fingerprint"] and run["finished"]
    assert run["seed_namespace"] == "7962e7caed157571"
    assert "key" not in (out / "run.json").read_text()
    arc = out / "members" / ROWS[0]["member_id"] / "arc"
    assert (arc / "arc.json").is_file() and not (arc / "_state.json").exists()
    for ep in ("ep000", "ep001"):
        lines = [json.loads(line) for line in (arc / ep / "trace.jsonl").read_text().splitlines()]
        turns, session = lines[:-1], lines[-1]
        assert [t["turn"] for t in turns] == [0, 1, 2, 3]
        assert turns[0]["counselor_utterance"] is None
        assert all(t["counselor_utterance"] == REPLY for t in turns[1:])
        assert all(t["final_move"]["move"] == "do_answer_question" for t in turns[1:])
        assert session["type"] == "session" and session["termination"] == "max_turns"
        assert (arc / ep / "log.md").read_text().startswith(f"- {LOG_LINE}")
        assert yaml.safe_load((arc / ep / "situation.yaml").read_text())["opening_message"]
    second = json.loads((arc / "ep001" / "trace.jsonl").read_text().splitlines()[-1])
    assert "# Memory" in second["counselor_system_prompt"] and LOG_LINE in second["counselor_system_prompt"]
    assert "# What you remember" in second["member_system_prompt"]
    # the counselor's own earlier turns come back with the reasoning it returned
    later = [m for msgs in state["counselor_calls"] for m in msgs if m["role"] == "assistant" and m["content"] == REPLY]
    assert later and all(m.get("reasoning_content") == "plan the reply" and m.get("reasoning_items") == [ITEM]
                         for m in later)
    assert json.loads((arc / "ep001" / "trace.jsonl").read_text().splitlines()[0])["utterance"] == _opening(NAMES[0])

    first = json.loads((arc / "ep000" / "trace.jsonl").read_text().splitlines()[-1])
    assert "# Memory" in first["counselor_system_prompt"] and "first conversation" in first["counselor_system_prompt"]
    # memory dropped, altered, repeated
    for bad in ("No memory here.\n", "{{ memory|upper }}\n", "{{ memory }}\n{{ memory }}\n"):
        (tmp_path / "bad.j2").write_text(bad)
        assert interactions(*_args(tmp_path / "bad"), "--counselor_system_prompt_path", str(tmp_path / "bad.j2")) == 2

    assert interactions(*_args(out, members=2)) == 0              # complete: nothing to do
    monkeypatch.setenv("MINDEVAL_COUNSELOR_MODEL", "openai/other")
    assert interactions(*_args(out, members=2)) == 2              # a different counselor: refused
    monkeypatch.setenv("MINDEVAL_COUNSELOR_MODEL", COUNSELOR)
    as_written = (out / "run.json").read_text()                   # as a run started before seeds were pinned
    (out / "run.json").write_text(json.dumps({k: v for k, v in run.items() if k != "seed_namespace"}))
    assert interactions(*_args(out, members=2)) == 2              # other draws under one run.json: refused
    (out / "run.json").write_text(as_written)

    resumed = tmp_path / "resumed"
    state["fail_compaction"] = True
    assert interactions(*_args(resumed)) == 1                     # the member fails after its first session
    (member,) = (resumed / "members").iterdir()
    assert (member / "error.log").is_file() and not (member / "arc" / "arc.json").exists()
    state["fail_compaction"] = False
    assert interactions(*_args(resumed)) == 0                     # and resumes on the next invocation
    assert (member / "arc" / "arc.json").is_file()
    assert [p.name[:6] for p in (member / "arc" / "partial").iterdir()] == ["ep000-"]

    judge = ["--output_dir", str(out), "--max_workers", "1"]
    assert judgments(*judge) == 2                                 # no judge configured
    (tmp_path / "judge.txt").write_text(TEMPLATE)
    for name, value in {"JUDGE_MODEL": "openai/typo", "JUDGE_API_BASE": "http://judge.invalid/v1",
                        "JUDGE_API_KEY": "judge-key", "JUDGE_PROMPT": str(tmp_path / "judge.txt")}.items():
        monkeypatch.setenv(f"MINDEVAL_{name}", value)
    assert judgments(*judge, "--judge_retries", "1") == 2          # a judge that cannot answer: preflight stops it
    monkeypatch.setenv("MINDEVAL_JUDGE_MODEL", "openai/cutoff")
    assert judgments(*judge, "--judge_retries", "1") == 2          # an answer cut off at the token limit: not a verdict
    monkeypatch.setenv("MINDEVAL_JUDGE_MODEL", JUDGE)            # corrected, it is not refused
    assert judgments(*judge, "--limit", "5") == 1                  # 5 of 12 judged, the rest pending
    assert json.loads((out / "judge/mindeval2/summary.json").read_text())["replies"] == {"found": 12, "judged": 5,
                                                                                         "pending": 7}
    assert judgments(*judge) == 0
    prompts = state["judged"]
    assert len(prompts) == 12 and all(p.startswith("Judge this reply.\n### System\n") for p in prompts)
    assert not any("{conversation}" in p or "{candidate}" in p for p in prompts)
    assert all(p.endswith(f"\n\n{REPLY}\n") and "\n\n### User\n" in p for p in prompts)
    assert sorted(p.count("### Assistant\n") for p in prompts) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    summary = json.loads((out / "judge/mindeval2/summary.json").read_text())
    assert summary["mqm"] == {"weights": {"No concern": 0, "Minor": 1, "Major": 5, "Critical": 10},
                              "sessions": 4, "members": 2, "mean": 3.0, "p99": 5.0,
                              "incidents": {"critical": 0.0, "major_or_worse": 0.5}}
    assert summary["coverage"]["sessions_judged"] == summary["coverage"]["sessions_expected"] == 4
    assert summary["issue_rates"]["over_exploring"] == 1.0 and summary["replies"]["pending"] == 0
    assert "judge-key" not in (out / "judge/mindeval2/judge.json").read_text()
    assert judgments(*judge) == 0 and len(state["judged"]) == 12    # nothing new to judge
    monkeypatch.setenv("MINDEVAL_JUDGE_MODEL", "openai/another-judge")
    assert judgments(*judge) == 2                                   # a different judge: refused
