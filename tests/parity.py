"""Behavioural goldens for `generate_interactions.py`: scripted scenarios run through the real command line, and
everything they leave behind recorded — every file under the output directory and every model request, in order.

`tests/test_parity.py` replays each scenario and compares it with `tests/goldens/<scenario>.json`; a change to the
interaction runtime that is meant to change behaviour regenerates them, and the diff is the review:

    python tests/parity.py            # rewrite tests/goldens/*.json
    python tests/parity.py --check    # compare only

The model is scripted per counselor reply (`Script`): each scenario names what the patient does on its commit and
speak calls at each turn, so the paths that matter run end to end — a forced and an auto commit, a commit answered in
prose, an invalid move and its retry, a stray commit while speaking, agenda calls, a dirty utterance replaced by the
fallback, the member leaving, a session that dies and the run that resumes it, and arcs of three sessions.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
GOLDENS = TESTS / "goldens"
sys.path[:0] = [str(ROOT), str(TESTS)]

from conftest import interactions  # noqa: E402

import mindeval.arc as arc_mod  # noqa: E402
import mindeval.member as member_mod  # noqa: E402
from mindeval import llm  # noqa: E402

COUNSELOR = "openai/counselor"
MOVE = "do_answer_question"
ITEM = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}
NEXT = {"event": "The week went on.", "automatic_thoughts": ["same as before"], "behaviors": ["stayed in"],
        "physical": [], "goal": "talk it through", "checkpoints": ["the money", "sleep"]}
#: Keys dropped from every recorded file: wall-clock measurements, which no two runs share.
TIMING_KEYS = {"timings", "ms", "evolve_ms", "compact_ms"}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def _call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _response(content=None, calls=None, finish="stop", **extra):
    message = SimpleNamespace(content=content, tool_calls=calls, **extra)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], model="fake", id="r")


class Script:
    """A scripted litellm. It answers from the request alone (and how often the same request was seen, which is how a
    retry of an unchanged request is told apart), and records every request. `plan` maps a counselor reply number to
    what the patient does on that turn's commit and speak calls; turns it does not name answer plainly."""

    def __init__(self, plan: dict[int, dict] | None = None) -> None:
        self.plan = plan or {}
        self.requests: list[dict] = []
        self.seen: Counter = Counter()
        self.down: set[str] = set()  # call kinds that currently fail every attempt

    # litellm's two entry points: the sync one today, the async one once the runtime is async.
    def completion(self, **kw):
        return self._answer(kw)

    async def acompletion(self, **kw):
        return self._answer(kw)

    def _answer(self, kw: dict):
        self.requests.append(_request(kw))
        messages = kw["messages"]
        first = messages[0]["content"]
        attempt = self.seen[_sha([messages, kw.get("tools"), kw.get("tool_choice")])]
        self.seen[_sha([messages, kw.get("tools"), kw.get("tool_choice")])] += 1
        if kw["model"] == COUNSELOR:
            k = sum(m["role"] == "assistant" for m in messages) + 1
            return _response(f"<think>plan reply {k}</think>Reply {k}: how has it been?", reasoning_items=[ITEM])
        if "running log" in first:
            if "compact" in self.down:
                raise RuntimeError("compaction server down")
            return _response(f"- Entry {_sha(first)[:6]}: they talked it through.")
        if "Answer with JSON only" in first:
            name = first.split("Who they are:\n\n", 1)[1].split(" is ", 1)[0]
            return _response(json.dumps({**NEXT, "opening_message": f"hey, {name} again"}))
        turn = sum(m["role"] == "user" for m in messages)
        step = self.plan.get(turn, {})
        tools = [t["function"]["name"] for t in kw.get("tools") or []]
        if "commit_move" in tools:
            return self._commit(step.get("commit", "tool"), messages, turn)
        return self._speak(step.get("speak", "text"), messages, turn, attempt)

    def _commit(self, how: str, messages: list[dict], turn: int):
        move = {"reasoning": f"turn {turn}", "move": MOVE}
        retried = messages[-1]["role"] == "tool" and "did not come through" in messages[-1]["content"]
        if how == "content_json":
            return _response(f"<think>which move</think>Going with {json.dumps(move)}.")
        if how == "invalid" and not retried:
            return _response(calls=[_call("commit_move", {**move, "move": "do_fly"})])
        if how == "leave":
            move["end_conversation"] = True
        return _response(calls=[_call("commit_move", move, f"call_c{turn}")])

    def _speak(self, how: str, messages: list[dict], turn: int, attempt: int):
        last = messages[-1]
        # Down only once the member remembers an earlier session, i.e. from the second session on.
        if how == "fail" and "speak" in self.down and "# What you remember" in messages[0]["content"]:
            raise RuntimeError("patient server down")
        if how == "stray" and "already committed" not in (last.get("content") or ""):
            return _response(calls=[_call("commit_move", {"reasoning": "again", "move": MOVE}, f"call_s{turn}")])
        if how == "todos":
            names = [m.get("name") for m in messages[-4:] if m["role"] == "tool"]
            if "complete_todo" not in names:
                return _response(calls=[_call("complete_todo", {"todo": "the money"}, f"call_t{turn}")])
            if "add_todo" not in names:
                return _response(calls=[_call("add_todo", {"todo": "ask about the landlord"}, f"call_a{turn}")])
        if how == "dirty":
            return _response(f"**Header** turn {turn}, attempt {attempt}")
        return _response(f"<think>what to say</think>i guess turn {turn} was ok")


def _request(kw: dict) -> dict:
    """A request as recorded: the messages by count, hash and last two (so a diff points at the call and shows its
    end), everything else as sent, the key never."""
    out = {k: v for k, v in kw.items() if k not in ("messages", "api_key", "tools")}
    out["messages"] = {"n": len(kw["messages"]), "sha": _sha(kw["messages"]), "tail": kw["messages"][-2:]}
    if kw.get("tools"):
        out["tools"] = {"names": [t["function"]["name"] for t in kw["tools"]], "sha": _sha(kw["tools"])}
    return json.loads(json.dumps(out, default=str))


def _drop_timings(obj):
    if isinstance(obj, dict):
        return {k: _drop_timings(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [_drop_timings(v) for v in obj]
    return obj


def _files(out: Path) -> dict:
    """Every file the run left, by path relative to the output directory, as it should compare."""
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        if path.name == ".lock":
            continue
        if "/partial/" in rel:  # partial/ep001-<timestamp>/...
            head, tail = rel.split("/partial/", 1)
            name, _, rest = tail.partition("/")
            rel = f"{head}/partial/{name.split('-')[0]}-<when>/{rest}"
        text = path.read_text(encoding="utf-8")
        if path.name == "error.log":
            files[rel] = "<traceback>"
        elif path.suffix == ".jsonl":
            files[rel] = [_drop_timings(json.loads(line)) for line in text.splitlines() if line.strip()]
        elif path.name == "run.json":
            run = json.loads(text)
            for k in ("git", "started", "resumed"):
                run.pop(k)
            run["finished"] = bool(run["finished"])
            run["counselor"]["system_prompt"] = Path(run["counselor"]["system_prompt"]).name
            run["profiles"]["path"] = Path(run["profiles"]["path"]).name
            files[rel] = run
        elif path.suffix == ".json":
            files[rel] = _drop_timings(json.loads(text))
        else:
            files[rel] = text
    return files


@contextlib.contextmanager
def _world(tmp: Path, script: Script, patient_model: str):
    """The scenario's environment: a scripted litellm, no backoff waits, the endpoints in MINDEVAL_*, and a working
    directory with no .env in it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MINDEVAL_")}
    env.update({"MINDEVAL_PATIENT_MODEL": patient_model, "MINDEVAL_PATIENT_API_BASE": "http://patient.invalid/v1",
                "MINDEVAL_COUNSELOR_MODEL": COUNSELOR})
    cwd = os.getcwd()
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        stack.enter_context(mock.patch.object(llm, "_LIB", script))
        stack.enter_context(mock.patch.object(member_mod, "_pause", lambda attempt: None))
        stack.enter_context(mock.patch.object(arc_mod, "_pause", lambda attempt: None))
        stack.enter_context(mock.patch.object(llm.time, "sleep", lambda s: None))
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(cwd)


def _args(out: Path, members: int, sessions: int, max_turns: int) -> list[str]:
    return ["--output_dir", str(out), "--members", str(members), "--sessions", str(sessions),
            "--max_turns", str(max_turns), "--max_workers", "1"]


def _two_members(tmp: Path) -> dict:
    """Two members through three sessions each: the plain path, compaction and evolution twice per arc."""
    script, out = Script(), tmp / "run"
    with _world(tmp, script, "hosted_vllm/patient"):
        codes = [interactions(*_args(out, 2, 3, 3))]
    return {"codes": codes, "requests": script.requests, "files": _files(out)}


def _every_path(tmp: Path) -> dict:
    """One session in which each turn takes a different path, ending with the member leaving."""
    plan = {2: {"commit": "content_json"}, 3: {"commit": "invalid", "speak": "stray"}, 4: {"speak": "todos"},
            5: {"speak": "dirty"}, 6: {"commit": "leave"}}
    script, out = Script(plan), tmp / "run"
    with _world(tmp, script, "hosted_vllm/patient"):
        codes = [interactions(*_args(out, 1, 1, 8))]
    return {"codes": codes, "requests": script.requests, "files": _files(out)}


def _auto_commit(tmp: Path) -> dict:
    """A patient config/patient.yaml offers the move to as `auto`, with its provider params."""
    script, out = Script(), tmp / "run"
    with _world(tmp, script, "anthropic/claude-patient"):
        codes = [interactions(*_args(out, 1, 1, 2))]
    return {"codes": codes, "requests": script.requests, "files": _files(out)}


def _dies_then_resumes(tmp: Path) -> dict:
    """The second session dies mid-turn (the patient stops answering), and the next invocation resumes it."""
    script, out = Script({2: {"speak": "fail"}}), tmp / "run"
    with _world(tmp, script, "hosted_vllm/patient"):
        script.down = {"speak"}
        codes = [interactions(*_args(out, 1, 2, 3))]
        script.down = set()
        codes.append(interactions(*_args(out, 1, 2, 3)))
    return {"codes": codes, "requests": script.requests, "files": _files(out)}


SCENARIOS = {"two_members": _two_members, "every_path": _every_path, "auto_commit": _auto_commit,
             "dies_then_resumes": _dies_then_resumes}


def run(name: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(json.dumps(SCENARIOS[name](Path(tmp)), ensure_ascii=False, default=str))


def golden(name: str) -> dict:
    return json.loads((GOLDENS / f"{name}.json").read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    check = "--check" in argv
    GOLDENS.mkdir(exist_ok=True)
    status = 0
    for name in SCENARIOS:
        got = run(name)
        path = GOLDENS / f"{name}.json"
        if check:
            same = path.is_file() and golden(name) == got
            status |= not same
            print(f"{name}: {'same' if same else 'DIFFERENT'}")
        else:
            path.write_text(json.dumps(got, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            print(f"{name}: {len(got['requests'])} requests, {len(got['files'])} files -> {path.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
