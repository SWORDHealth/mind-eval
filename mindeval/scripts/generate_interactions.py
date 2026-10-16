"""`python mindeval/scripts/generate_interactions.py`: takes every member of the profile file through
its sessions against one counselor and writes the transcripts. Every session runs on NeMo UserSim's
engine (`mindeval/probe.py`), which also writes it as a UserSim row under `<output_dir>/usersim/`.
Members run concurrently; each is independent, and a member that fails is recorded and left for the
next invocation to resume.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from jsonargparse import CLI
from pydantic import ValidationError

from mindeval import __version__, engine, env, llm
from mindeval.arc import counselor_memory, member_seed, run_arc, set_aside_partial
from mindeval.config import PACKAGE_DIR, SEED_FINGERPRINT, CallSpec, Config, load_config
from mindeval.member import build_member
from mindeval.models import MemberRow
from mindeval.probe import Runner
from mindeval.session import Counselor
from mindeval.utils import InputError, check_params, deep_merge, hold_lock

DEFAULT_PROFILES = PACKAGE_DIR.parent / "data" / "profiles.jsonl"
DEFAULT_COUNSELOR = PACKAGE_DIR.parent / "examples" / "counselor_system.j2"
#: The fields of run.json that make two invocations the same benchmark run. Endpoints, timeouts,
#: retries and the worker count may change between invocations; these may not.
IDENTITY = ("mindeval_version", "fingerprint", "seed_namespace", "patient.model", "counselor.model",
            "counselor.system_prompt_sha256", "counselor.temperature", "counselor.max_tokens",
            "counselor.params", "patient.params", "patient.max_tokens", "patient.force_move",
            "profiles.sha256", "sessions", "max_turns", "seed", "usersim_models.sha256")


def _missing(names: list[str]) -> InputError:
    return InputError("set " + ", ".join(names) + " in the environment or in a .env file (see .env.example)")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_members(path: Path, n: int, cfg: Config) -> list[MemberRow]:
    if not path.is_file():
        raise InputError(f"profiles file not found: {path}")
    rows: list[MemberRow] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = MemberRow.model_validate_json(line)
        except ValidationError as e:
            raise InputError(f"{path}:{i}: not a valid member row:\n{e}") from None
        if row.archetype is not None and row.archetype not in cfg.archetypes.archetypes:
            raise InputError(f"{path}:{i}: unknown archetype {row.archetype!r}")
        rows.append(row)
    ids = [r.member_id for r in rows]
    if len(set(ids)) != len(ids):
        raise InputError(f"{path}: member ids are not unique")
    if n < 1 or n > len(rows):
        raise InputError(f"--members {n}, but {path} has {len(rows)} members")
    return rows[:n]


def _git_describe() -> str | None:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=PACKAGE_DIR.parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — informational only
        return None


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def run_manifest(counselor_system_prompt_path: Path, profiles_path: Path, sessions: int, max_turns: int, seed: int,
                 max_workers: int, cfg: Config, patient: CallSpec, counselor: CallSpec,
                 rows: list[MemberRow]) -> dict:
    return {
        "mindeval_version": __version__, "git": _git_describe(), "fingerprint": cfg.fingerprint,
        # What every arc is drawn from (config.SEED_FINGERPRINT). An identity field of its own: a run
        # started before it was pinned has the same version and fingerprint but other draws, and none.
        "seed_namespace": SEED_FINGERPRINT,
        "patient": {"model": patient.model, "api_base": patient.api_base, "temperature": patient.temperature,
                    "max_retries": patient.max_retries, "timeout": patient.timeout, "params": patient.params,
                    "max_tokens": patient.max_tokens, "force_move": patient.force_move},
        "counselor": {"model": counselor.model, "api_base": counselor.api_base,
                      "system_prompt": str(counselor_system_prompt_path),
                      "system_prompt_sha256": _sha(counselor_system_prompt_path.read_bytes()),
                      "temperature": counselor.temperature, "max_tokens": counselor.max_tokens,
                      "params": counselor.params, "timeout": counselor.timeout, "max_retries": counselor.max_retries},
        "profiles": {"path": str(profiles_path), "sha256": _sha(profiles_path.read_bytes()),
                     "rows": sum(1 for line in profiles_path.read_text().splitlines() if line.strip())},
        "members": len(rows), "member_ids": [m.member_id for m in rows], "sessions": sessions,
        "max_turns": max_turns, "seed": seed, "workers": max_workers,
    }


def _as_sent(model: Any) -> dict:
    """What every request to a models-file alias carries besides the call's own messages, tools and seed, as a
    CallSpec's fields: on this path the run records these, not mindeval's defaults."""
    sent = model.consolidate_kwargs()
    sent.pop("extra_headers", None)  # may carry a credential
    sent.pop("timeout", None)  # the side's timeout is mindeval's --*_timeout on this path too (HandsBack sends it)
    return {"model": model.model_name, "api_base": model.model_provider.endpoint,
            "temperature": sent.pop("temperature", None), "max_tokens": sent.pop("max_tokens", None), "params": sent}


def _usersim_models(path: Path, cfg: Config, *, patient_timeout: float, counselor_timeout: float,
                    counselor_retries: int, max_workers: int, flags: dict) -> tuple[CallSpec, CallSpec, dict, bool]:
    """The patient and counselor from a UserSim models file: their CallSpecs (mindeval's own policy and the run's
    record; what is sent comes from the file), the models every call goes through, and whether the counselor's
    reasoning is replayed. `flags` maps each flag the file replaces to (value, default): any value other than its
    default is refused, since the file, not the flag, decides what is sent."""
    from usersim.cli._errors import ConfigError
    from usersim.engine.external import model_facades, replays_assistant_reasoning

    if not path.is_file():
        raise InputError(f"--usersim_models: {path} not found")
    given = [flag for flag, (value, default) in flags.items() if value != default]
    if given:
        raise InputError(f"{', '.join(given)} do not apply with --usersim_models: set them on the alias in {path}")
    try:
        facades = model_facades(path, aliases=("user_model", "assistant_model"))
        replay = replays_assistant_reasoning(path)
    except ConfigError as e:
        raise InputError(f"--usersim_models {path}: {e}") from None
    patient_model, counselor_model = facades["user_model"], facades["assistant_model"]
    for alias, model in facades.items():
        if model.max_parallel_requests < max_workers:
            print(f"mindeval: {alias} allows {model.max_parallel_requests} requests at a time "
                  f"(max_parallel_requests in {path}), fewer than --max_workers {max_workers}: members will "
                  "wait for each other's calls", file=sys.stderr)
    patient = CallSpec(**_as_sent(patient_model), timeout=patient_timeout, max_retries=cfg.patient.max_retries,
                       force_move=cfg.patient.provider(patient_model.model_name).force_move)
    cspec = CallSpec(**_as_sent(counselor_model), timeout=counselor_timeout, max_retries=counselor_retries)
    models = engine.session_models(engine.HandsBack(patient_model, patient.timeout),
                                   engine.HandsBack(counselor_model, cspec.timeout))
    return patient, cspec, models, replay


def check_identity(out: Path, manifest: dict) -> dict | None:
    """The existing run.json, if this invocation may resume it; raises if it may not."""
    path = out / "run.json"
    if not path.is_file():
        if (out / "members").exists():
            raise InputError(f"{out} holds member output but no run.json; use a fresh --output_dir")
        return None
    old = json.loads(path.read_text())
    diffs = [f"  {k}: {_get(old, k)!r} -> {_get(manifest, k)!r}" for k in IDENTITY if _get(old, k) != _get(manifest, k)]
    if diffs:
        raise InputError(f"{out} is a different run; resuming it would mix two benchmarks:\n" + "\n".join(diffs)
                         + "\nUse a fresh --output_dir.")
    if manifest["members"] < old.get("members", 0):
        raise InputError(f"{out} is a run of {old['members']} members; pass --members {old['members']} (or more) "
                         f"to resume it")
    return old


async def preflight(cfg: Config, row: MemberRow, counselor: Counselor, patient: CallSpec, models: dict) -> None:
    """Two real calls before any member starts: a counselor reply, and one `commit_move`, which must come
    back as a tool call. Some servers reject a forced function or answer it in prose."""
    opening = row.situation.opening_message
    try:
        text, _ = await counselor.respond(models, counselor.prompt_with(counselor_memory([])),
                                          [{"role": "assistant", "content": opening}])
    except Exception as e:  # noqa: BLE001
        raise InputError(f"preflight: the counselor did not answer: {e}") from None
    member = build_member(cfg, row, row.situation, row.knobs, patient, models=models)
    member.said(opening)
    member.hears(text)
    move, call = await member.commit(seed=0)
    if move is None or call["via"] != "tool_call":
        choice = "a named tool_choice" if patient.force_move else "tool_choice auto"
        raise InputError(f"preflight: the patient did not return a commit_move tool call (via={call['via']}, "
                         f"error={call['error']}). The patient server must support tool calling with {choice}.")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(output_dir: Path, counselor_system_prompt_path: Path = DEFAULT_COUNSELOR,
         counselor_temperature: float | None = 1.0, counselor_max_tokens: int | None = None,
         counselor_params: dict | None = None, counselor_timeout: float = 600, counselor_retries: int = 5,
         patient_timeout: float = 600, patient_params: dict | None = None, profiles_path: Path = DEFAULT_PROFILES,
         members: int = 50, sessions: int = 3, max_turns: int = 30, seed: int = 0, max_workers: int = 16,
         usersim_models: Path | None = None, dry_run: bool = False) -> int:
    """Runs every member of the profile file through its sessions against one counselor.

    The transcripts are written under output_dir. Rerun with the same output_dir to resume: finished
    members are kept and unfinished sessions restart.

    Models, endpoints and keys come from the environment (or a .env file in the working directory):
    MINDEVAL_PATIENT_MODEL and MINDEVAL_COUNSELOR_MODEL, each with optional _API_BASE and _API_KEY. See
    .env.example. With --usersim_models they come from a UserSim models file instead, and every call
    goes through UserSim's own model layer (see the README).

    Args:
        output_dir: output directory; rerun with the same one to resume
        counselor_system_prompt_path: Jinja template of its system prompt: any text around one
            {{ memory }}, kept unchanged, with no other variables, filters or tags
        counselor_temperature: null to leave it unsent
        counselor_max_tokens: not sent when omitted
        counselor_params: JSON merged into every request, e.g. '{"reasoning_effort": "high"}'
        counselor_timeout: seconds per request
        counselor_retries: attempts per reply
        patient_timeout: seconds per request (compaction has its own 1800)
        patient_params: JSON merged over the settings config/patient.yaml gives this kind of model,
            e.g. '{"reasoning_effort": "high"}' for a model it does not list
        profiles_path: JSONL of members
        members: run the first N members of the file
        sessions: sessions per member
        max_turns: counselor replies per session, at most
        seed: base seed; each member's is derived from it
        max_workers: members run concurrently
        usersim_models: a UserSim models file naming the patient (user_model) and the counselor
            (assistant_model); their settings, providers and keys come from it, and replay_reasoning on
            assistant_model gives the counselor its earlier reasoning back
        dry_run: validate and render prompts; no model calls, no writes

    Returns:
        0 if every member is complete, 1 if some member is not yet finished.
    """
    env.load_env_file(Path(".env"))
    try:
        counselor_params = counselor_params or {}
        patient_params = patient_params or {}
        for flag, params in (("--counselor_params", counselor_params), ("--patient_params", patient_params)):
            check_params(flag, params)

        cfg = load_config()
        if not counselor_system_prompt_path.is_file():
            raise InputError(f"counselor system prompt not found: {counselor_system_prompt_path}")
        for flag, value in (("--sessions", sessions), ("--max_turns", max_turns), ("--max_workers", max_workers),
                            ("--counselor_retries", counselor_retries)):
            if value < 1:
                raise InputError(f"{flag} must be at least 1")
        for flag, value in (("--patient_timeout", patient_timeout), ("--counselor_timeout", counselor_timeout)):
            if value <= 0:
                raise InputError(f"{flag} must be positive")
        models, replay = None, False
        if usersim_models is not None:
            patient, cspec, models, replay = _usersim_models(usersim_models, cfg, patient_timeout=patient_timeout,
                                                             counselor_timeout=counselor_timeout,
                                                             counselor_retries=counselor_retries,
                                                             max_workers=max_workers,
                                                             flags={"--counselor_params": (counselor_params, {}),
                                                                    "--patient_params": (patient_params, {}),
                                                                    "--counselor_max_tokens":
                                                                        (counselor_max_tokens, None),
                                                                    "--counselor_temperature":
                                                                        (counselor_temperature, 1.0)})
            rows = load_members(profiles_path, members, cfg)
        else:
            missing: list[str] = []
            pe, ce = env.endpoint("PATIENT", missing), env.endpoint("COUNSELOR", missing)
            if missing:
                raise _missing(missing)
            rows = load_members(profiles_path, members, cfg)
            provider = cfg.patient.provider(pe.model)
            patient = CallSpec(model=pe.model, api_base=pe.api_base, api_key=pe.api_key,
                               temperature=cfg.patient.temperature, max_tokens=provider.max_tokens,
                               timeout=patient_timeout, max_retries=cfg.patient.max_retries,
                               # deep, not shallow: a --patient_params extra_body must not silently drop
                               # provider.params' own extra_body keys (e.g. chat_template_kwargs for vLLM)
                               params=deep_merge(provider.params, patient_params), force_move=provider.force_move)
            cspec = CallSpec(model=ce.model, api_base=ce.api_base, api_key=ce.api_key,
                             temperature=counselor_temperature, max_tokens=counselor_max_tokens,
                             timeout=counselor_timeout, max_retries=counselor_retries, params=counselor_params)
        text = counselor_system_prompt_path.read_text(encoding="utf-8")
        try:
            counselor = Counselor.from_text(cspec, text)
        except ValueError as e:
            raise InputError(f"{counselor_system_prompt_path}: {e}") from None
        manifest = run_manifest(counselor_system_prompt_path, profiles_path, sessions, max_turns, seed, max_workers,
                                cfg, patient, cspec, rows)
        if usersim_models is not None:  # absent otherwise, so a litellm run's run.json is as it was
            manifest["usersim_models"] = {"path": str(usersim_models), "sha256": _sha(usersim_models.read_bytes()),
                                          "replay_reasoning": replay}
        old = check_identity(output_dir, manifest)

        arc_dirs = {m.member_id: output_dir / "members" / m.member_id / "arc" for m in rows}
        todo = [m for m in rows if not (arc_dirs[m.member_id] / "arc.json").is_file()]
        print(f"mindeval {__version__} | fingerprint {cfg.fingerprint} | {len(rows)} members x {sessions} "
              f"sessions x <= {max_turns} counselor replies | {len(rows) - len(todo)} already complete", flush=True)

        if dry_run:
            for m in rows:
                build_member(cfg, m, m.situation, m.knobs, patient)
            print(f"dry run: {len(rows)} profiles valid, session-1 prompts render; counselor prompt "
                  f"{manifest['counselor']['system_prompt_sha256'][:12]}; would write to {output_dir}"
                  + ("" if old is None else " (resuming)"))
            return 0
        if not todo:
            print("nothing to do: every member is complete")
            return 0

        with hold_lock(output_dir):
            old = check_identity(output_dir, manifest)  # again, under the lock
            llm._lib()  # import litellm once, before anything awaits it

            manifest["started"] = old["started"] if old else _now()
            manifest["resumed"] = (old.get("resumed", []) + [_now()]) if old else []
            manifest["members"] = max(manifest["members"], old.get("members", 0)) if old else manifest["members"]
            manifest["finished"] = None
            # The run's rows, as UserSim stores a run: its id is when the run first started.
            runner = Runner(patient, cspec, max_turns=max_turns, rows_dir=output_dir / "usersim",
                            run_id=str(int(datetime.datetime.fromisoformat(manifest["started"]).timestamp())),
                            provenance={"git": manifest["git"], "bank_version": {
                                "seed_namespace": SEED_FINGERPRINT, "profiles_sha256": manifest["profiles"]["sha256"],
                                "counselor_system_prompt_sha256": manifest["counselor"]["system_prompt_sha256"]}},
                            models=models, replay_reasoning=replay)
            counts = {"done": 0, "failed": 0}

            async def one(row: MemberRow, slots: asyncio.Semaphore) -> None:
                async with slots:
                    arc_dir = arc_dirs[row.member_id]
                    moved = None
                    t0 = time.monotonic()
                    try:
                        moved = set_aside_partial(arc_dir)
                        arc = await run_arc(cfg, row, counselor, runner=runner, patient=patient, sessions=sessions,
                                            max_turns=max_turns, arc_seed=member_seed(row.member_id, seed),
                                            out=arc_dir)
                        outcome = "done"
                        detail = " ".join(e.termination.replace("member_terminated", "left")
                                          .replace("max_turns", "cap") for e in arc.episodes)
                    except Exception as e:  # noqa: BLE001 — recorded; the member resumes on the next invocation
                        outcome, detail = "failed", f"{type(e).__name__}: {str(e)[:200]}"
                        arc_dir.parent.mkdir(parents=True, exist_ok=True)
                        with (arc_dir.parent / "error.log").open("a", encoding="utf-8") as f:
                            f.write(f"--- {_now()}\n{traceback.format_exc()}\n")
                    counts[outcome] += 1
                    n = counts["done"] + counts["failed"]
                    note = f" (resumed; partial session moved to {moved})" if moved else ""
                    print(f"[{n}/{len(todo)}] {row.member_id} {outcome} in {time.monotonic() - t0:.0f}s: "
                          f"{detail}{note}", flush=True)

            async def everyone() -> None:
                await preflight(cfg, todo[0], counselor, patient, runner.models)
                _write_json(output_dir / "run.json", manifest)
                slots = asyncio.Semaphore(max_workers)  # members start in file order as slots free up
                await asyncio.gather(*(one(row, slots) for row in todo))

            try:
                asyncio.run(everyone())
            except KeyboardInterrupt:
                print(f"\ninterrupted. Rerun the same command to resume: finished members are kept and unfinished "
                      f"sessions restart. Output: {output_dir}", flush=True)
                os._exit(130)

            complete = sum(1 for m in rows if (arc_dirs[m.member_id] / "arc.json").is_file())
            manifest["finished"] = _now() if complete == len(rows) else None
            _write_json(output_dir / "run.json", manifest)
            print(f"{complete}/{len(rows)} members complete; {counts['failed']} failed this run"
                  + ("" if complete == len(rows)
                     else " (see members/<id>/error.log; rerun the same command to resume)"))
            return 0 if complete == len(rows) else 1
    except InputError as e:
        print(f"mindeval: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(CLI([main], as_positional=False))
