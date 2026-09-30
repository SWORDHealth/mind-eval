"""The judge abstraction: `judge/mindeval1/` (v1's rubric) and `judge/mindeval2/` (MindJudge-27B's
MQM) built and run through the real CLI, against a scripted litellm exactly like test_smoke.py's."""

import functools
import hashlib
import json
import threading
from types import SimpleNamespace

from conftest import _clear_env, judgments
from jsonargparse import CLI

from mindeval import judge_prompts, llm
from mindeval.arc import counselor_memory
from mindeval.config import load_config
from mindeval.judge import MindEval1Judge, read_session
from mindeval.scripts import generate_interactions, generate_judgments
from mindeval.utils import parse_judge_scores

JUDGE, JUDGE_KEY = "openai/judge", "judge-key"
#: Two members' fixed scores, chosen so every criterion mean differs (a column swap would be caught).
ALPHA = {"Clinical Accuracy & Competence": 6, "Ethical & Professional Conduct": 5, "Assessment & Response": 4,
        "Therapeutic Relationship & Alliance": 3, "AI-Specific Communication Quality": 2}
BETA = dict.fromkeys(ALPHA, 1)


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="fake", id="r")


def _v1_answer(scores):
    lines = "\n".join(f"{k}: {v}" for k, v in scores.items())
    return f"<think>weighing it</think>\n{lines}\n"


def _knobs():
    return {"concealment_propensity": "low", "frustration_reactivity": "low", "session_affect": "sadness",
            "verbosity": "low", "writing_style": "plain"}


def _turn(turn, utterance, counselor_utterance=None):
    return {"type": "turn", "session_id": "s", "run_id": "r", "turn": turn, "seed": 0, "turn_seed": 0,
            "fingerprint": "fp", "knobs": _knobs(), "counselor_utterance": counselor_utterance,
            "utterance": utterance}


def _session_line(member_id, episode, *, formulation, counselor_system_prompt, termination):
    return {"type": "session", "session_id": f"{member_id}-{episode}", "run_id": "r", "seed": 0, "fingerprint": "fp",
            "profile_id": member_id, "situation_id": "sit", "knobs": _knobs(), "termination": termination,
            "counselor_system_prompt": counselor_system_prompt, "formulation": formulation}


def _write_session(out, member_id, episode, *, opening, replies, formulation, counselor_system_prompt=None,
                   termination="max_turns", log_entries=None):
    """A trace.jsonl matching models.py's TurnRecord/SessionRecord shape: turn 0 is the member's
    opening, turn k>=1 is `replies[k-1]` (the counselor) and a filler member reply. `log_entries`, if
    given, writes this episode's log.md too (the memory the *next* episode's counselor receives),
    matching what `arc.compact_log` leaves behind."""
    lines = [_turn(0, opening)]
    for i, reply in enumerate(replies, start=1):
        lines.append(_turn(i, f"member reply {i}", counselor_utterance=reply))
    lines.append(_session_line(member_id, episode, formulation=formulation,
                               counselor_system_prompt=counselor_system_prompt, termination=termination))
    path = out / "members" / member_id / "arc" / episode / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    if log_entries is not None:
        (path.parent / "log.md").write_text("\n".join(f"- {e}" for e in log_entries) + "\n")
    return path


def _write_run(out, members, sessions):
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"members": members, "sessions": sessions}))


def _set_judge_env(monkeypatch, *, prompt_path=None):
    for name, value in {"JUDGE_MODEL": JUDGE, "JUDGE_API_BASE": "http://judge.invalid/v1",
                        "JUDGE_API_KEY": JUDGE_KEY}.items():
        monkeypatch.setenv(f"MINDEVAL_{name}", value)
    if prompt_path is not None:
        monkeypatch.setenv("MINDEVAL_JUDGE_PROMPT", str(prompt_path))


def test_mindeval1_happy_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)

    out = tmp_path / "run"
    _write_run(out, members=2, sessions=2)
    openings = {"m_alpha": "Alpha here, feeling low.", "m_beta": "Beta checking in."}
    # session 1 has no memory yet; session 2's memory is session 1's compacted log (its log.md)
    logs = {"m_alpha": ["Alpha talked about feeling low."], "m_beta": ["Beta checked in about work."]}
    for member in ("m_alpha", "m_beta"):
        _write_session(out, member, "ep000", opening=openings[member], replies=["How are you?", "Tell me more."],
                      formulation=f"{member} is 30, works retail, alone.", log_entries=logs[member])
        _write_session(out, member, "ep001", opening=openings[member], replies=["How are you?", "Tell me more."],
                      formulation=f"{member} is 30, works retail, alone.")

    prompts = []

    def completion(**kw):
        prompt = kw["messages"][0]["content"]
        prompts.append(prompt)
        assert kw.get("api_key") == JUDGE_KEY
        return _response(_v1_answer(ALPHA if openings["m_alpha"] in prompt else BETA))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1") == 0

    jdir = out / "judge" / "mindeval1"
    verdicts = [json.loads(line) for line in (jdir / "verdicts.jsonl").read_text().splitlines()]
    assert len(verdicts) == 4 and len(prompts) == 4          # 2 members x 2 sessions, not x 2 replies
    assert all(v["key"].count("/") == 1 for v in verdicts)    # "member/episode", no per-reply "/tN"
    assert all(v["answer"].strip().endswith(f"{ALPHA['AI-Specific Communication Quality']}")
              or v["answer"].strip().endswith(f"{BETA['AI-Specific Communication Quality']}") for v in verdicts)

    summary = json.loads((jdir / "summary.json").read_text())
    assert summary["means"] == {
        "Clinical Accuracy & Competence": 3.5, "Ethical & Professional Conduct": 3.0,
        "Assessment & Response": 2.5, "Therapeutic Relationship & Alliance": 2.0,
        "AI-Specific Communication Quality": 1.5, "Overall score": 2.5, "Average score": 2.5,
    }
    identity = json.loads((jdir / "judge.json").read_text())
    assert identity["version"] == "mindeval1" and identity["prompt"] == "built-in"

    # the counselor's memory for each session, computed the same way the judge should: nothing for
    # session 1, session 1's compacted log for session 2
    no_memory_yet = counselor_memory([])
    with_memory = {member: counselor_memory(entries) for member, entries in logs.items()}
    assert len({no_memory_yet, *with_memory.values()}) == 3  # all distinct, so the checks below are meaningful

    for p in prompts:
        assert "### System" not in p
        # the few-shots earlier in the template have their own <member_details>/<conversation>; only
        # the new instance, after this marker, is this session's
        new_instance = p.split("Now, consider a new instance:", 1)[1]
        assert "<member_details>" in new_instance
        details = new_instance.split("<member_details>", 1)[1].split("</member_details>", 1)[0].strip()
        # never the member's private formulation; always what the counselor had for that session
        assert details in (no_memory_yet, *with_memory.values())
        assert "works retail" not in details
        convo = new_instance.split("<conversation>", 1)[1].split("</conversation>", 1)[0].strip()
        assert convo.startswith("<member>\n")                 # opens with the member's turn
        assert convo.index("<member>") < convo.index("<therapist>")
        opening = convo.split("<member>\n", 1)[1].split("\n</member>", 1)[0]
        assert opening in openings.values()                    # ... starting with the opening message


def test_missing_criterion_is_never_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Gamma is 50.")

    calls = []

    def completion(**kw):
        calls.append(1)
        scores = {k: 4 for k in ALPHA if k != "Assessment & Response"}  # one criterion missing
        return _response(_v1_answer(scores))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    # judge_retries=1: no retry, so the call count below pins the attempt count directly
    rv = judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1",
                   "--judge_retries", "1")
    assert rv == 2  # the run's only unit is the preflight one; a judge that never answers stops the pass (exit 2)
    assert len(calls) == 1

    verdicts = [json.loads(line) for line in (out / "judge/mindeval1/verdicts.jsonl").read_text().splitlines()]
    assert len(verdicts) == 1 and "_error" in verdicts[0]
    assert not (set(MindEval1Judge.KEYS) & set(verdicts[0]))  # never a silently-defaulted score (incl. no "3")
    # the end of the unparseable answer travels with the error, so it can be audited without a re-judge:
    # the ratings that were given, not the reasoning before them
    assert "Assessment & Response" in verdicts[0]["_error"] and "Clinical Accuracy & Competence: 4" in verdicts[0][
        "_error"]
    assert "weighing it" not in verdicts[0]["_error"]


def test_mindeval1_keeps_the_verdict_after_a_long_inline_reasoning_trace(tmp_path, monkeypatch):
    """A judge served without a reasoning parser puts its whole trace before `</think>` in the
    content. The row must keep the ratings (re-parseable without a new judgement) and the trace as
    reasoning_content, not the first few thousand characters of the trace."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Kappa is 44.")
    ratings = "\n".join(f"{k}: {v}" for k, v in ALPHA.items())
    answer = "<think>" + "reasoning " * 600 + "</think>\n" + ratings
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(answer)))
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1") == 0

    row = json.loads((out / "judge/mindeval1/verdicts.jsonl").read_text())
    assert parse_judge_scores(row["answer"])[0]["Clinical Accuracy & Competence"] == 6
    assert "reasoning" not in row["answer"]
    assert row["reasoning_content"].startswith("reasoning reasoning")
    assert len(row["reasoning_content"]) <= MindEval1Judge.ANSWER_MAX_CHARS


def test_mindeval2_refuses_missing_counselor_system_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text("Judge this reply.\n{conversation}\n\n{candidate}\n")
    _set_judge_env(monkeypatch, prompt_path=prompt_path)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    trace = _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"],
                           formulation="Delta is 22.", counselor_system_prompt=None)

    def refuses_any_call(**kw):
        raise AssertionError("mindeval2 must refuse before any call is made")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=refuses_any_call))
    rv = judgments("--output_dir", str(out), "--judge_version", "mindeval2", "--max_workers", "1")
    assert rv == 2
    assert str(trace) in capsys.readouterr().err

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(_v1_answer(ALPHA))))
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1") == 0
    assert (out / "judge/mindeval1/summary.json").is_file()


def test_both_versions_run_concurrently_on_one_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text("Judge this reply.\n{conversation}\n\n{candidate}\n")
    _set_judge_env(monkeypatch, prompt_path=prompt_path)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Epsilon is 33.",
                  counselor_system_prompt="Be kind.")

    def completion(**kw):
        prompt = kw["messages"][0]["content"]
        if "<member_details>" in prompt:  # mindeval1's format
            return _response(_v1_answer(ALPHA))
        return _response('<think>ok</think>{"severity": "Minor", "vague": 0, "other": 1}')  # mindeval2's format

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))

    results = {}

    def run(version):
        results[version] = judgments("--output_dir", str(out), "--judge_version", version, "--max_workers", "1")

    threads = [threading.Thread(target=run, args=(v,)) for v in ("mindeval1", "mindeval2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"mindeval1": 0, "mindeval2": 0}
    for v in ("mindeval1", "mindeval2"):
        jdir = out / "judge" / v
        assert json.loads((jdir / "judge.json").read_text())["version"] == v
        assert (jdir / "verdicts.jsonl").is_file() and (jdir / "summary.json").is_file()

    mindeval2_dir = out / "judge" / "mindeval2"
    before = {p.name: p.read_bytes() for p in mindeval2_dir.iterdir()}
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1") == 0  # no-op
    after = {p.name: p.read_bytes() for p in mindeval2_dir.iterdir()}
    assert before == after


def test_mindeval1_is_not_blocked_by_mindeval2s_lock(tmp_path, monkeypatch):
    """Each judge version locks its own directory (`judge/<version>/.lock`), not one shared
    `judge/.lock`. A barrier holds a real mindeval2 pass inside its own `with hold_lock(...)` — through
    the actual generate_judgments.py code path, not a path this test guesses at — while mindeval1 runs
    to completion. `hold_lock`'s `LOCK_NB` never blocks, so if the two versions ever shared one lock
    file this fails deterministically (mindeval1 sees exit code 2, refused) rather than hanging.
    Checked by hand: regressing generate_judgments.py to `hold_lock(output_dir / "judge")` for every
    version makes this test fail, and restoring `.../ "judge" / judge_version` makes it pass again."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text("Judge this reply.\n{conversation}\n\n{candidate}\n")
    _set_judge_env(monkeypatch, prompt_path=prompt_path)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Eta is 41.",
                  counselor_system_prompt="Be kind.")

    mindeval2_holding = threading.Event()
    release_mindeval2 = threading.Event()

    def completion(**kw):
        prompt = kw["messages"][0]["content"]
        if "<member_details>" in prompt:  # mindeval1's format: must not be blocked by mindeval2's lock
            return _response(_v1_answer(ALPHA))
        mindeval2_holding.set()  # mindeval2's format: its own hold_lock() is held for this whole call
        assert release_mindeval2.wait(timeout=5), "mindeval1 never got to run while mindeval2 held its lock"
        return _response('<think>ok</think>{"severity": "Minor", "vague": 0, "other": 1}')

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))

    result = {}

    def run_mindeval2():
        result["mindeval2"] = judgments("--output_dir", str(out), "--judge_version", "mindeval2",
                                        "--max_workers", "1")

    t = threading.Thread(target=run_mindeval2)
    t.start()
    assert mindeval2_holding.wait(timeout=5)  # mindeval2's own lock is held now

    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1") == 0

    release_mindeval2.set()
    t.join(timeout=5)
    assert result["mindeval2"] == 0


def test_judge_prompt_env_required_only_for_mindeval2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)  # no prompt_path: MINDEVAL_JUDGE_PROMPT stays unset

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Zeta is 27.")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(_v1_answer(ALPHA))))
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1") == 0

    def refuses_any_call(**kw):
        raise AssertionError("mindeval2 must refuse before any call is made")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=refuses_any_call))
    rv = judgments("--output_dir", str(out), "--judge_version", "mindeval2", "--max_workers", "1")
    assert rv == 2
    assert "MINDEVAL_JUDGE_PROMPT" in capsys.readouterr().err


def test_fingerprint_is_pinned():
    # The config of MindEval2 a18de8be (7962e7caed157571) with the YAML comments removed: the same data,
    # other bytes. A change here means config/*.yaml or prompts/*.j2 changed.
    assert load_config().fingerprint == "37868bb5e406de0d"


def _wrap(fn):
    """`fn` wrapped so jsonargparse still sees its real signature (`inspect.signature` follows
    `__wrapped__`) but the call is captured instead of run, so a renamed flag fails to parse."""
    captured = {}

    @functools.wraps(fn)
    def stub(**kwargs):
        captured.update(kwargs)
        return 0

    return stub, captured


def test_generate_interactions_flag_names(tmp_path):
    stub, captured = _wrap(generate_interactions.main)
    args = [
        "--output_dir", str(tmp_path / "run"), "--counselor_system_prompt_path", str(tmp_path / "counselor.j2"),
        "--counselor_temperature", "0.5", "--counselor_max_tokens", "111", "--counselor_params", '{"a": 1}',
        "--counselor_timeout", "12", "--counselor_retries", "2", "--patient_timeout", "13",
        "--patient_params", '{"b": 2}', "--profiles_path", str(tmp_path / "profiles.jsonl"), "--members", "7",
        "--sessions", "4", "--max_turns", "9", "--seed", "3", "--max_workers", "5", "--dry_run", "true",
    ]
    assert CLI([stub], as_positional=False, args=args) == 0
    assert captured == {
        "output_dir": tmp_path / "run", "counselor_system_prompt_path": tmp_path / "counselor.j2",
        "counselor_temperature": 0.5, "counselor_max_tokens": 111, "counselor_params": {"a": 1},
        "counselor_timeout": 12.0, "counselor_retries": 2, "patient_timeout": 13.0, "patient_params": {"b": 2},
        "profiles_path": tmp_path / "profiles.jsonl", "members": 7, "sessions": 4, "max_turns": 9, "seed": 3,
        "max_workers": 5, "dry_run": True,
    }


def test_generate_judgments_flag_names(tmp_path):
    stub, captured = _wrap(generate_judgments.main)
    args = [
        "--output_dir", str(tmp_path / "run"), "--judge_version", "mindeval1", "--judge_temperature", "0.7",
        "--judge_max_tokens", "222", "--judge_params", '{"c": 3}', "--judge_timeout", "77", "--judge_retries", "4",
        "--max_workers", "6", "--limit", "8",
    ]
    assert CLI([stub], as_positional=False, args=args) == 0
    assert captured == {
        "output_dir": tmp_path / "run", "judge_version": "mindeval1", "judge_temperature": 0.7,
        "judge_max_tokens": 222, "judge_params": {"c": 3}, "judge_timeout": 77.0, "judge_retries": 4,
        "max_workers": 6, "limit": 8,
    }


#: The v1 rubric's prefix (through "Now, consider a new instance:"), sha256'd rather than compared
#: against `legacy:mindeval/judge_prompts.py` directly (that ref is not fetched in CI). Checked locally
#: while writing this test: byte-identical to legacy's JUDGE_PROMPT_TEMPLATE over the same span.
V1_RUBRIC_PREFIX_SHA256 = "f4e24ab453b5ea65bba6a81726ef8020a059a1bc88628d2ad8bddf0205f09d87"


def test_mindeval1_prompt_keeps_v1_rubric():
    template = judge_prompts.JUDGE_PROMPT_TEMPLATE.template
    marker = "Now, consider a new instance:"
    prefix = template[:template.index(marker) + len(marker)]
    assert hashlib.sha256(prefix.encode()).hexdigest() == V1_RUBRIC_PREFIX_SHA256


def test_judge_params_with_api_key_is_refused_and_never_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text("Judge this reply.\n{conversation}\n\n{candidate}\n")
    _set_judge_env(monkeypatch, prompt_path=prompt_path)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Theta is 29.",
                  counselor_system_prompt="Be kind.")

    def refuses_any_call(**kw):
        raise AssertionError("a screened --judge_params must be refused before any call is made")

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=refuses_any_call))
    rv = judgments("--output_dir", str(out), "--judge_version", "mindeval2", "--max_workers", "1",
                   "--judge_params", '{"api_key": "sk-leak"}')
    assert rv == 2
    assert not (out / "judge").exists()  # refused before the lock (and so the directory) is even created


def test_judge_params_are_part_of_the_judge_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text("Judge this reply.\n{conversation}\n\n{candidate}\n")
    _set_judge_env(monkeypatch, prompt_path=prompt_path)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Iota is 36.",
                  counselor_system_prompt="Be kind.")

    seen_params = []

    def completion(**kw):
        seen_params.append(kw.get("reasoning_effort"))
        return _response('<think>ok</think>{"severity": "Minor", "vague": 0, "other": 1}')

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    args = ["--output_dir", str(out), "--judge_version", "mindeval2", "--max_workers", "1"]
    assert judgments(*args, "--judge_params", '{"reasoning_effort": "low"}') == 0
    identity = json.loads((out / "judge/mindeval2/judge.json").read_text())
    assert identity["params"] == {"reasoning_effort": "low"} and seen_params == ["low"]

    # a later pass with different params, once verdicts exist, is refused like a different model
    rv = judgments(*args, "--judge_params", '{"reasoning_effort": "high"}')
    assert rv == 2 and seen_params == ["low"]  # refused before any call
    assert json.loads((out / "judge/mindeval2/judge.json").read_text())["params"] == {"reasoning_effort": "low"}


def test_preflight_does_not_stall_forever_on_a_unit_that_always_fails(tmp_path, monkeypatch):
    """The preflight is a check on the judge, not on one unlucky unit: it must only gate a pass while
    the directory holds no successful verdict yet. Once one exists, the first pending unit in sorted
    order (after pass 1, exactly the ones that errored) is an ordinary error, not a pass-stopping
    preflight — otherwise one unit that fails every attempt (a long session, a content filter, a
    reasoning trace over max_tokens) stops every later pass, and nothing sorted after it is ever judged."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)

    out = tmp_path / "run"
    _write_run(out, members=3, sessions=1)
    openings = {"m000": "m000 opens.", "m001": "m001 opens, and always fails.", "m002": "m002 opens."}
    for member, opening in openings.items():
        _write_session(out, member, "ep000", opening=opening, replies=["how are you?"],
                       formulation=f"{member} is 40.")

    def completion(**kw):
        if openings["m001"] in kw["messages"][0]["content"]:
            raise RuntimeError("m001 always fails")
        return _response(_v1_answer(ALPHA))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    args = ["--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1", "--judge_retries", "1"]

    # pass 1: --limit 2 judges m000 (the preflight unit, which succeeds) then m001 (an ordinary error,
    # since it is not batch[0]); m002 is not yet attempted.
    assert judgments(*args, "--limit", "2") == 1
    verdicts = [json.loads(line) for line in (out / "judge/mindeval1/verdicts.jsonl").read_text().splitlines()]
    assert {v["member"] for v in verdicts if "_error" not in v} == {"m000"}
    assert {v["member"] for v in verdicts if "_error" in v} == {"m001"}

    # pass 2: batch[0] in sorted order is now m001, the always-failing unit. With the old preflight
    # this would raise and m002 would never be judged; done is non-empty (m000), so it must not.
    assert judgments(*args) == 1  # m001 is still pending, but that must not block m002
    verdicts = [json.loads(line) for line in (out / "judge/mindeval1/verdicts.jsonl").read_text().splitlines()]
    assert {v["member"] for v in verdicts if "_error" not in v} == {"m000", "m002"}


def test_read_session_tolerates_a_torn_multibyte_trailing_write(tmp_path):
    """A trace read mid-append (the README says the judge may run alongside generate_interactions.py)
    can end inside a multi-byte character — an em dash or a curly quote are common with
    ensure_ascii=False. That must mean "not finished", never a UnicodeDecodeError."""
    complete = json.dumps({"type": "turn", "turn": 0, "counselor_utterance": None, "utterance": "hi"})
    torn = "café is where we".encode()[:-1]  # a dangling lead byte of a 2-byte UTF-8 character
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes((complete + "\n").encode() + b'{"type": "turn", "turn": 1, "utterance": "' + torn)
    session = read_session(trace)
    assert session["finished"] is False


def test_a_verdict_quoting_u2028_is_read_back_whole_and_not_re_judged(tmp_path, monkeypatch):
    """verdicts.jsonl is written with ensure_ascii=False, which leaves U+2028/U+2029/U+0085 raw, and
    was read back with splitlines(), which treats them as line breaks. A verdict whose justification
    quotes text containing one must not be torn into two undecodable halves and silently re-judged."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text("Judge this reply.\n{conversation}\n\n{candidate}\n")
    _set_judge_env(monkeypatch, prompt_path=prompt_path)

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Lambda is 50.",
                  counselor_system_prompt="Be kind.")

    calls = []

    def completion(**kw):
        calls.append(1)
        return _response('<think>ok</think>{"severity": "Minor", "vague": 0, '
                         '"justification": "It pauses.\\u2028Then continues."}')

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    args = ["--output_dir", str(out), "--judge_version", "mindeval2", "--max_workers", "1"]
    assert judgments(*args) == 0
    assert len(calls) == 1
    assert " " in (out / "judge/mindeval2/verdicts.jsonl").read_text()

    assert judgments(*args) == 0  # a no-op rerun: the verdict must be read back whole, not re-judged
    assert len(calls) == 1


def test_parse_judge_scores_takes_the_last_match_per_criterion():
    # Commentary before the ratings (e.g. restating a criterion while discussing it) must not be
    # mistaken for the score itself; only the rating that follows, in the fixed output format, counts.
    text = ("Clinical Accuracy & Competence: 2 issues stand out here, but overall it's solid.\n"
            "Clinical Accuracy & Competence: 5\n"
            "Ethical & Professional Conduct: 4\n"
            "Assessment & Response: 3\n"
            "Therapeutic Relationship & Alliance: 3\n"
            "AI-Specific Communication Quality: 3\n")
    scores = parse_judge_scores(text)[0]
    assert scores["Clinical Accuracy & Competence"] == 5
