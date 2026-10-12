"""Shared by the analysis scripts: the run registry, verdicts, session scores, the member bootstrap and manifests.

Everything here reads run outputs and nothing writes into a run directory. Statistics are plain Python so the
analysis needs nothing beyond the harness's own environment.

Units. A session is one member x episode. Its score is MindEval2's MQM conversation severity: the mean severity
weight of the session's judged counselor replies (No concern 0, Minor 1, Major 5, Critical 10). A counselor's score
in a session index is the mean over members. Intervals come from resampling members with replacement, the same
draw for every counselor, so a counselor comparison stays paired within member.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SEVERITIES = ("No concern", "Minor", "Major", "Critical")
WEIGHT = {"No concern": 0, "Minor": 1, "Major": 5, "Critical": 10}
NOT_ISSUES = {"key", "input_sha", "member", "episode", "turn", "severity", "justification", "usage", "judge", "_error"}
ISSUES = ("under_guiding", "over_exploring", "treating_perspective_as_fact", "out_of_scope_responding",
          "gives_bad_advice", "excessive_verbosity", "formulaic_responding", "overstatement", "uncritical_endorsement",
          "anthropomorphizing", "false_fact", "false_memory", "missed_shared_context", "other")
EPISODES = ("ep000", "ep001", "ep002")
BOOT = 2000


def runs_root(arg: str | None = None) -> Path:
    root = arg or os.environ.get("MINDEVAL_RUNS_ROOT")
    if not root:
        sys.exit("set --runs_root or MINDEVAL_RUNS_ROOT to the directory runs.yaml's paths are relative to")
    return Path(root)


#: The memory ablation forks runs of the mind-eval port (this fingerprint): every such run in runs.yaml, and GLM-5.3's
#: port run, which runs.yaml does not list (the paper's GLM-5.3 row is the earlier harness's run).
ABLATION_FINGERPRINT = "7962e7caed157571"
#: glm53_nvfp4 forks the same run against the NVFP4 weights (the FP8 server went away mid-ablation); its conditions are
#: compared with its own compacted rerun only.
ABLATION_EXTRA = {"glm53": {"label": "GLM-5.3", "path": "mindeval2_preserve/glm53"},
                  "glm53_nvfp4": {"label": "GLM-5.3 NVFP4", "path": "mindeval2_preserve/glm53"},
                  # API counselors: their paper runs are at the earlier harness, so each has a 3-session source run at
                  # this fingerprint made for the ablation (mindeval_qwen/go_api.sh).
                  "claude_sonnet5": {"label": "Claude Sonnet 5", "path": "mindeval_qwen/claude_sonnet5"},
                  "gpt56terra": {"label": "GPT-5.6 Terra", "path": "mindeval_qwen/gpt56terra"}}


def ablation_counselors(path: Path = HERE / "runs.yaml") -> dict[str, dict]:
    """key -> {label, path}: the counselors the memory ablation can fork, with each source run's path under the runs
    root."""
    out = {r["key"]: {"label": r["label"], "path": r["path"]} for r in yaml.safe_load(path.read_text())["runs"]
           if r["fingerprint"] == ABLATION_FINGERPRINT}
    return {**out, **ABLATION_EXTRA}


def registry(root: Path, path: Path = HERE / "runs.yaml") -> tuple[list[dict], dict]:
    """The runs (each with its run.json under `run`) and the judges. A run whose fingerprint differs from the
    registry's is refused."""
    reg = yaml.safe_load(path.read_text())
    runs = []
    for r in reg["runs"]:
        run = json.loads((root / r["path"] / "run.json").read_text())
        if run["fingerprint"] != r["fingerprint"]:
            raise SystemExit(f"{r['key']}: run.json fingerprint {run['fingerprint']} != registry {r['fingerprint']}")
        runs.append({**r, "dir": root / r["path"], "run": run})
    return runs, reg["judges"]


def judge_dir(root: Path, run: dict, judge: str, judges: dict) -> Path:
    if judge == "own":  # MindEval2's own runs keep verdicts in judge/, the mind-eval port's in judge/mindeval2/
        d = run["dir"] / "judge"
        return d if (d / "verdicts.jsonl").is_file() else d / "mindeval2"
    return root / judges[judge]["dir"] / run["key"] / "judge" / "mindeval2"


def load_verdicts(path: Path) -> dict[str, dict]:
    """Key -> verdict. The last successful line for a key wins; error lines are skipped."""
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(v, dict) and "_error" not in v and v.get("severity") in WEIGHT:
                out[v["key"]] = v
    return out


def trace_replies(run_dir: Path) -> dict[str, int]:
    """Key -> 1 for every counselor reply in a finished session, as the judge enumerates them."""
    keys = {}
    for trace in sorted(run_dir.glob("members/*/arc/ep*/trace.jsonl")):
        member, ep = trace.parents[2].name, trace.parent.name
        finished, turns = False, []
        for line in trace.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "session":
                finished = rec.get("termination") != "error"
            elif rec.get("counselor_utterance"):
                turns.append(rec["turn"])
        if finished:
            for t in turns:
                keys[f"{member}/{ep}/t{t}"] = 1
    return keys


def sessions(verdicts: dict[str, dict]) -> dict[tuple[str, str], dict]:
    """(member, episode) -> {"score", "n", "flags": Counter, "severity": Counter}."""
    acc: dict[tuple[str, str], dict] = {}
    for v in verdicts.values():
        s = acc.setdefault((v["member"], v["episode"]), {"w": [], "flags": Counter(), "severity": Counter()})
        s["w"].append(WEIGHT[v["severity"]])
        s["severity"][v["severity"]] += 1
        s["flags"].update(k for k in ISSUES if v.get(k) == 1)
    return {k: {"score": sum(s["w"]) / len(s["w"]), "n": len(s["w"]), "flags": s["flags"], "severity": s["severity"]}
            for k, s in acc.items()}


def load_judge(root: Path, runs: list[dict], judge: str, judges: dict, check_coverage: bool = True) -> dict:
    """run key -> sessions(...) for one judge, with a coverage check against the traces."""
    out, problems = {}, []
    for r in runs:
        path = judge_dir(root, r, judge, judges) / "verdicts.jsonl"
        if not path.is_file():
            problems.append(f"{r['key']}: no verdicts at {path}")
            continue
        v = load_verdicts(path)
        if check_coverage:
            expected = trace_replies(r["dir"])
            v = {k: x for k, x in v.items() if k in expected}
            missing = len(expected) - len(v)
            if missing:
                problems.append(f"{r['key']}: {missing} of {len(expected)} replies unjudged")
        out[r["key"]] = sessions(v)
    return {"sessions": out, "problems": problems}


# -- statistics ---------------------------------------------------------------------------------------------------

def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def quantile(sorted_xs: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    if not sorted_xs:
        return float("nan")
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def ci(draws: list[float]) -> tuple[float, float]:
    xs = sorted(x for x in draws if not math.isnan(x))
    return quantile(xs, 0.025), quantile(xs, 0.975)


def member_draws(members: list[str], n: int = BOOT, seed: int = 0) -> list[list[str]]:
    """The bootstrap resamples of members, fixed by `seed` so every script draws the same ones."""
    rng = random.Random(seed)
    return [[members[rng.randrange(len(members))] for _ in members] for _ in range(n)]


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Kendall's tau-b; ties in either ranking count as neither concordant nor discordant."""
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else float("nan")


def ranks(scores: dict[str, float]) -> dict[str, int]:
    """1 = lowest MQM (best). Ties share the better rank."""
    order = sorted(scores.values())
    return {k: order.index(v) + 1 for k, v in scores.items()}


# -- output -------------------------------------------------------------------------------------------------------

def git_describe() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=HERE,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 — informational
        return "unknown"


def judge_identity(root: Path, runs: list[dict], judge: str, judges: dict) -> dict:
    ids = {}
    for r in runs:
        p = judge_dir(root, r, judge, judges) / "judge.json"
        if p.is_file():
            j = json.loads(p.read_text())
            ids[r["key"]] = {k: j.get(k) for k in ("model", "prompt_sha256", "temperature", "max_tokens", "version")}
    return ids


def write_manifest(out: Path, script: str, runs: list[dict], judges_used: dict, extra: dict | None = None) -> None:
    """manifest.json beside the outputs: the analysis code's git SHA, and for every run its run.json identity
    (code, fingerprint, models, seed) and each judge's identity."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": script, "analysis_git": git_describe(),
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "runs": [{"key": r["key"], "path": r["path"], "git": r["run"].get("git"),
                  "fingerprint": r["run"]["fingerprint"], "patient_model": r["run"]["patient"]["model"],
                  "counselor_model": r["run"]["counselor"]["model"],
                  "counselor_prompt_sha256": r["run"]["counselor"].get("system_prompt_sha256"),
                  "profiles_sha256": r["run"]["profiles"]["sha256"], "seed": r["run"]["seed"],
                  "members": r["run"]["members"], "sessions": r["run"]["sessions"],
                  "max_turns": r["run"]["max_turns"]} for r in runs],
        "judges": judges_used, **(extra or {}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def sha_short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
