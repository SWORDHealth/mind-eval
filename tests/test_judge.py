"""The judge abstraction: `judge/mindeval1/` (v1's rubric) and `judge/mindeval2/` (MQM, with the prompt
in judge_prompts.py) built and run through the real CLI, against a scripted litellm exactly like
test_smoke.py's."""

import functools
import hashlib
import json
import threading
from types import SimpleNamespace

from conftest import _clear_env, _response, judgments
from jsonargparse import CLI

from mindeval import judge_prompts, llm
from mindeval.arc import counselor_memory
from mindeval.config import load_config
from mindeval.judge import MindEval1Judge, MindEval2Judge, read_session
from mindeval.scripts import generate_interactions, generate_judgments
from mindeval.utils import parse_judge_scores

JUDGE, JUDGE_KEY = "openai/judge", "judge-key"
#: Two members' fixed scores, chosen so every criterion mean differs (a column swap would be caught).
ALPHA = {"Clinical Accuracy & Competence": 6, "Ethical & Professional Conduct": 5, "Assessment & Response": 4,
        "Therapeutic Relationship & Alliance": 3, "AI-Specific Communication Quality": 2}
BETA = dict.fromkeys(ALPHA, 1)


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
    # every session opens differently, so each prompt below can be traced back to its member and session
    openings = {("m_alpha", "ep000"): "Alpha here, feeling low.", ("m_alpha", "ep001"): "Alpha again, a bit better.",
                ("m_beta", "ep000"): "Beta checking in.", ("m_beta", "ep001"): "Beta back after a week."}
    # session 1 has no memory yet; session 2's memory is session 1's compacted log (its log.md)
    logs = {"m_alpha": ["Alpha talked about feeling low."], "m_beta": ["Beta checked in about work."]}
    for (member, episode), opening in openings.items():
        _write_session(out, member, episode, opening=opening, replies=["How are you?", "Tell me more."],
                      formulation=f"{member} is 30, works retail, alone.",
                      log_entries=logs[member] if episode == "ep000" else None)

    prompts = []

    def completion(**kw):
        prompt = kw["messages"][0]["content"]
        prompts.append(prompt)
        assert kw.get("api_key") == JUDGE_KEY
        return _response(_v1_answer(ALPHA if "Alpha" in prompt.split("Now, consider a new instance:", 1)[1] else BETA))

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

    # the counselor's memory for each session, exactly what it received in {{ memory }}: the "no earlier
    # sessions" text in session 1, that member's own session-1 log in session 2
    expected = {(member, episode): counselor_memory(logs[member] if episode == "ep001" else [])
                for member, episode in openings}
    assert len(set(expected.values())) == 3  # all distinct, so the checks below are meaningful

    seen = set()
    for p in prompts:
        assert "### System" not in p
        # the few-shots earlier in the template have their own <member_details>/<conversation>; only
        # the new instance, after this marker, is this session's
        new_instance = p.split("Now, consider a new instance:", 1)[1]
        convo = new_instance.split("<conversation>", 1)[1].split("</conversation>", 1)[0].strip()
        assert convo.startswith("<member>\n")                 # opens with the member's turn
        assert convo.index("<member>") < convo.index("<therapist>")
        opening = convo.split("<member>\n", 1)[1].split("\n</member>", 1)[0]
        (session,) = [k for k, v in openings.items() if v == opening]  # ... starting with that session's opening
        details = new_instance.split("<member_details>", 1)[1].split("</member_details>", 1)[0].strip()
        # never the member's private formulation; exactly what the counselor had for this session
        assert details == expected[session], session
        seen.add(session)
    assert seen == set(openings)


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
    assert not (out / "judge/mindeval1/judge.json").exists()  # no verdict, so no judge pinned

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
    assert parse_judge_scores(row["answer"])["Clinical Accuracy & Competence"] == 6
    assert "reasoning" not in row["answer"]
    assert row["reasoning_content"].startswith("reasoning reasoning")
    assert len(row["reasoning_content"]) <= MindEval1Judge.ANSWER_MAX_CHARS

    # an answer that is itself long (commentary before the ratings, no trace) keeps its end, where they are
    row = MindEval1Judge().parse("commentary " * 600 + ratings, {})
    assert len(row["answer"]) == MindEval1Judge.ANSWER_MAX_CHARS
    assert parse_judge_scores(row["answer"]) == parse_judge_scores(ratings)


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


#: sha256 of the judge prompt as the MindEval 2 judges were validated with it: one user message, the example between
#: the task and the rubric. The two messages below are that text split in two, with "Below is part of a
#: conversation" reworded for a system message.
VALIDATED_SINGLE_PROMPT_SHA256 = "f7c660a806c46dacd79826c23ff981991dc2c4b4f5d03d160d3446da6289c444"
#: The judge's identity for the built-in prompt (system and user template); every mindeval2 verdict depends on it.
MINDEVAL2_JUDGE_PROMPT_SHA256 = "22fec230b106a39de498e010add505af0d3256fcad20adf23138489b368b27fd"


def test_mindeval2_judge_prompt_is_the_validated_one_split_in_two():
    system, user = judge_prompts.MINDEVAL2_JUDGE_SYSTEM_PROMPT, judge_prompts.MINDEVAL2_JUDGE_USER_PROMPT
    assert "{conversation}" not in system and "{candidate}" not in system
    assert user.count("{conversation}") == 1 and user.count("{candidate}") == 1
    task, rubric = system.split("## Known issues to watch for")
    single = (task.replace("The next message holds part of a conversation", "Below is part of a conversation")
              + user + "\n\n## Known issues to watch for" + rubric)
    assert hashlib.sha256(single.encode()).hexdigest() == VALIDATED_SINGLE_PROMPT_SHA256
    assert MindEval2Judge().prompt_sha256 == MINDEVAL2_JUDGE_PROMPT_SHA256


def test_mindeval2_uses_the_built_in_prompt_unless_given_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)  # no prompt_path: MINDEVAL_JUDGE_PROMPT stays unset

    out = tmp_path / "run"
    _write_run(out, members=1, sessions=1)
    _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Zeta is 27.",
                   counselor_system_prompt="You are a counselor.")
    calls = []

    def completion(**kw):
        calls.append(kw["messages"])
        return _response('It asks a good question.\n{"severity": "No concern", "vague": 0, "other": 0}')

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval2", "--max_workers", "1") == 0
    # Every call: the same system message (so a provider can cache it), then the example alone.
    assert calls and all([m["role"] for m in c] == ["system", "user"] for c in calls)
    assert all(c[0]["content"] == judge_prompts.MINDEVAL2_JUDGE_SYSTEM_PROMPT for c in calls)
    assert all(c[1]["content"].startswith("## Conversation so far\n\n### System\nYou are a counselor.")
               and c[1]["content"].endswith("### Candidate Response\nhow are you?") for c in calls)
    identity = json.loads((out / "judge" / "mindeval2" / "judge.json").read_text())
    assert identity["prompt"] == "built-in" and identity["prompt_sha256"] == MINDEVAL2_JUDGE_PROMPT_SHA256

    # mindeval1 takes no prompt file, so a MINDEVAL_JUDGE_PROMPT meant for mindeval2 doesn't stop it.
    monkeypatch.setenv("MINDEVAL_JUDGE_PROMPT", str(tmp_path / "absent.txt"))
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=lambda **kw: _response(_v1_answer(ALPHA))))
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1") == 0


def test_a_severe_reply_diluted_by_a_long_session_still_shows_as_an_incident():
    """The review's case: one Critical among 30 replies averages to 0.33, under a one-reply session with a
    Minor (1.0). The mean stays the headline, by design; `incidents` is where the Critical shows."""
    from mindeval.judge import conversation_severity

    long_session = [{"member": "m0", "episode": "ep000", "severity": "Critical"}] + [
        {"member": "m0", "episode": "ep000", "severity": "No concern"}] * 29
    short_session = [{"member": "m1", "episode": "ep000", "severity": "Minor"}]
    mqm = conversation_severity(long_session + short_session)
    assert mqm["mean"] == round((10 / 30 + 1) / 2, 3) and mqm["p99"] == 1.0
    assert mqm["incidents"] == {"critical": 0.5, "major_or_worse": 0.5}
    assert conversation_severity([])["incidents"] == {"critical": None, "major_or_worse": None}


def test_fingerprint_is_pinned():
    # The config of MindEval2 a18de8be (7962e7caed157571) with the YAML comments removed (37868bb5e406de0d),
    # then evolve_situation.j2 given the running log and the next session's conduct (5987fd25e55199ab),
    # then style2.yaml's three move descriptions folded into moves.yaml and member_system.j2's dead
    # `agenda_plain` branch dropped, both leaving every rendered prompt as it was (b2d4dfd7b708a9ad),
    # then the Guard's guard.yaml (off) and moves.yaml's negative_expressive (fa531e499d8e19bd).
    # A change here means config/*.yaml or prompts/*.j2 changed.
    assert load_config().fingerprint == "fa531e499d8e19bd"


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


def test_judge_defaults_are_the_benchmark_judges_settings(tmp_path, monkeypatch):
    """With no judge flags, a pass runs with GPT-6.1 Sol's settings: high reasoning effort, no temperature,
    32768 max tokens; '{}' sends no extra params."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text("Judge this reply.\n{conversation}\n\n{candidate}\n")
    _set_judge_env(monkeypatch, prompt_path=prompt_path)
    calls = []

    def completion(**kw):
        calls.append(kw)
        return _response('<think>ok</think>{"severity": "Minor", "vague": 0, "other": 1}')

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    for name, extra in (("default", []), ("none", ["--judge_params", "{}"])):
        out = tmp_path / name
        _write_run(out, members=1, sessions=1)
        _write_session(out, "m000", "ep000", opening="hi", replies=["how are you?"], formulation="Iota is 36.",
                      counselor_system_prompt="Be kind.")
        assert judgments("--output_dir", str(out), "--max_workers", "1", *extra) == 0
    identity = json.loads((tmp_path / "default/judge/mindeval2/judge.json").read_text())
    assert identity["params"] == {"reasoning_effort": "high"}
    assert identity["temperature"] is None and identity["max_tokens"] == 32768
    assert calls[0]["reasoning_effort"] == "high" and "temperature" not in calls[0]
    assert json.loads((tmp_path / "none/judge/mindeval2/judge.json").read_text())["params"] == {}
    assert "reasoning_effort" not in calls[-1]


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


def test_preflight_of_a_fresh_directory_tries_past_a_first_unit_that_always_fails(tmp_path, monkeypatch):
    """The same stall, one step earlier: the always-failing unit sorts first in a directory with no
    verdict yet. The preflight tries the next units rather than stopping every pass on that one; only
    a judge that fails all of the first PREFLIGHT_UNITS stops the pass."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)
    args = ["--judge_version", "mindeval1", "--max_workers", "1", "--judge_retries", "1"]

    def run_dir(name, failing, members=3):
        out = tmp_path / name
        _write_run(out, members=members, sessions=1)
        for i in range(members):
            _write_session(out, f"m00{i}", "ep000", opening=f"m00{i} opens.", replies=["how are you?"],
                           formulation="Mu is 40.")
        calls = []

        def completion(**kw):
            member = kw["messages"][0]["content"].split(" opens.", 1)[0][-4:]
            calls.append(member)
            if member in failing:
                raise RuntimeError("content filter")
            return _response(_v1_answer(ALPHA))

        monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
        return out, calls

    out, calls = run_dir("one_bad", failing={"m000"})
    for _ in range(2):  # every pass judges what it can; m000 stays pending, and never stops the pass
        assert judgments("--output_dir", str(out), *args) == 1
    verdicts = [json.loads(line) for line in (out / "judge/mindeval1/verdicts.jsonl").read_text().splitlines()]
    assert {v["member"] for v in verdicts if "_error" not in v} == {"m001", "m002"}
    assert (out / "judge/mindeval1/judge.json").is_file()

    out, calls = run_dir("bad_judge", failing={"m000", "m001", "m002"}, members=4)
    assert judgments("--output_dir", str(out), *args) == 2
    assert calls == ["m000", "m001", "m002"]  # PREFLIGHT_UNITS of them, then the pass stops; m003 is never tried
    assert not (out / "judge/mindeval1/judge.json").exists()


def test_verdicts_without_their_judge_json_are_refused(tmp_path, monkeypatch):
    """Verdicts with no judge.json beside them can't be tied to a judge (the rows don't record one), so a
    pass refuses them, under any judge, rather than pin itself and mix its verdicts with someone else's."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)

    out = tmp_path / "run"
    _write_run(out, members=3, sessions=1)
    for member in ("m000", "m001", "m002"):
        _write_session(out, member, "ep000", opening=f"{member} hi", replies=["How are you?"], formulation="Nu.")
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        return _response(_v1_answer(ALPHA))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    args = ["--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1", "--limit", "1"]
    meta = out / "judge/mindeval1/judge.json"
    verdicts = out / "judge/mindeval1/verdicts.jsonl"

    assert judgments(*args) == 1  # model A judges one session
    before = verdicts.read_text()
    meta.unlink()
    monkeypatch.setenv("MINDEVAL_JUDGE_MODEL", "openai/a-different-judge")
    assert judgments(*args) == 2  # model B is refused before any call
    monkeypatch.setenv("MINDEVAL_JUDGE_MODEL", JUDGE)
    assert judgments(*args) == 2  # so is model A: nothing on disk says it wrote them
    assert calls == [JUDGE] and verdicts.read_text() == before and not meta.exists()


def test_a_torn_last_verdict_never_swallows_the_next_one(tmp_path, monkeypatch):
    """A pass killed mid-write leaves a last line with no line break. The next pass must start its verdicts on
    a line of their own: a torn fragment is cut off (its unit is judged again), a whole record is kept."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _set_judge_env(monkeypatch)

    out = tmp_path / "run"
    _write_run(out, members=3, sessions=1)
    for member in ("m000", "m001", "m002"):
        _write_session(out, member, "ep000", opening=f"{member} hi", replies=["How are you?"], formulation="Nu.")
    calls = []

    def completion(**kw):
        calls.append(next(m for m in ("m000", "m001", "m002") if f"{m} hi" in kw["messages"][0]["content"]))
        return _response(_v1_answer(ALPHA))

    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=completion))
    args = ["--output_dir", str(out), "--judge_version", "mindeval1", "--max_workers", "1"]
    verdicts = out / "judge/mindeval1/verdicts.jsonl"

    assert judgments(*args, "--limit", "1") == 1
    with verdicts.open("ab") as f:  # killed while writing m001's verdict
        f.write(b'{"key": "m001/ep000", "input_sha": "')
    assert judgments(*args, "--limit", "1") == 1  # m001 is judged again and lands on its own line
    verdicts.write_bytes(verdicts.read_bytes().rstrip(b"\n"))  # a whole last record that lost its line break
    assert judgments(*args) == 0  # only m002 is left; nothing is judged twice
    assert calls == ["m000", "m001", "m002"]
    rows = [json.loads(line) for line in verdicts.read_text().splitlines()]
    assert [r["member"] for r in rows] == ["m000", "m001", "m002"]
    assert judgments(*args) == 0 and calls == ["m000", "m001", "m002"]  # and a later pass finds them all


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
    scores = parse_judge_scores(text)
    assert scores["Clinical Accuracy & Competence"] == 5
