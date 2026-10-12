"""Session by session: a trend over the sessions, or a step that switches on with memory?

    python analysis/trajectory.py --runs_root <dir> --ablation <memory_ablation dir> --labels <ablation_labels dir>
        --out <dir> [--exclude <leaks.jsonl>]

Ablation runs (sessions 1-6, session 1 shared by every condition): per judge view and memory condition, the mean over
counselors at each session of MQM and of its memory and exploration parts (drivers.py's split), and the labelers'
member repeats and use of notes; then per condition the step (session 2 minus session 1) and the trend (each member's
least-squares slope over sessions 2-6, per session). Session 1 is scored from the compacted run in every condition (the
same session). Paper runs (sessions 1-3, every judge): the same per session, with the step from 1 to 2 and the change
from 2 to 3. Intervals: 95%, members resampled within each counselor (pool from ablation.py).
Per counselor (by_counselor.csv), with and without memory: session MQM at each session, the step and the trend, each
with a member-bootstrap interval.
Writes trajectory.csv, steps.csv, by_counselor.csv, report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import pool  # noqa: E402
from common import EXCLUDE, judge_dir, load_verdicts, mean, registry, runs_root, set_exclusions  # noqa: E402
from drivers import parts  # noqa: E402

FIELDS = {"MQM": "_mqm", "memory part": "group:memory", "exploration part": "group:exploration"}
MODES = ("compacted", "none", "full", "split")
#: Conditions drawn per counselor (by_counselor.csv): with and without the counselor's memory; full and split are
#: deferred.
BY_COUNSELOR_MODES = ("compacted", "none")
PAPER = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview"}


def load_labels(d: Path) -> dict:
    keys = {k["id"]: k for k in map(json.loads, (d / "keys.jsonl").read_text().splitlines())} \
        if (d / "keys.jsonl").is_file() else {}
    out = {}
    if (d / "labels.jsonl").is_file():
        for line in (d / "labels.jsonl").read_text().splitlines():
            x = json.loads(line)
            if "parsed" in x and x["id"] in keys:
                k = keys[x["id"]]
                out[k["member"], k["episode"]] = x["parsed"]
    return out


def slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 3:
        return None
    mx, my = mean(x for x, _ in points), mean(y for _, y in points)
    den = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / den if den else None


def pooled_row(per_c: dict[str, dict[str, float]]) -> dict | None:
    per_c = {c: v for c, v in per_c.items() if v}
    return pool(per_c) if len(per_c) >= 2 else None


def series(scores: dict[str, dict[tuple[str, str], float]], eps: list[str]) -> tuple[list, dict, dict]:
    """scores: counselor -> (member, episode) -> value. Returns per-episode pooled means, the step and the slope."""
    per_ep = []
    for e in eps:
        r = pooled_row({c: {m: v for (m, ee), v in s.items() if ee == e} for c, s in scores.items()})
        per_ep.append((e, r))
    step = pooled_row({c: {m: s[m, eps[1]] - s[m, eps[0]] for (m, ee) in s if ee == eps[0] and (m, eps[1]) in s}
                       for c, s in scores.items()})
    trend = {}
    for c, s in scores.items():
        trend[c] = {}
        for m in {m for (m, _) in s}:
            b = slope([(int(e[2:]) + 1, s[m, e]) for e in eps[1:] if (m, e) in s])
            if b is not None:
                trend[c][m] = b
    return per_ep, step, pooled_row(trend)


def fmt(r: dict | None, nd: int = 2) -> str:
    return "-" if not r or r.get("diff") is None else f"{r['diff']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"


def lvl(r: dict | None, nd: int = 2) -> str:
    return "-" if not r or r.get("diff") is None else f"{r['diff']:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path, help="later_sessions/ablation_labels")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--exclude_counselors", default="glm53")
    args = ap.parse_args()
    set_exclusions(args.exclude)
    args.out.mkdir(parents=True, exist_ok=True)
    traj, steps, by_c = [], [], []
    lines = ["# Session by session: trend or step?", "",
             "Step: session 2 minus session 1. Trend: each member's slope over sessions 2-6 (per session). Means over "
             "counselors; 95% intervals resample members within each counselor.", ""]
    eps6 = [f"ep{i:03d}" for i in range(6)]
    cdirs = [c for c in sorted((args.ablation / "main6").iterdir())
             if c.is_dir() and c.name not in args.exclude_counselors.split(",")]
    # Judges.
    for judge, jl in (("deepseek_v4", "DeepSeek-V4.1"), ("mindjudge", "MindJudge-27B")):
        for view in ("reference", "as_sent"):
            lines += [f"## {jl}, {view.replace('_', ' ')} (ablation, sessions 1-6)", "",
                      "| condition | measure | " + " | ".join(f"s{i + 1}" for i in range(6)) + " | step s1→s2 | trend s2-6 |",
                      "|---|---|" + "---|" * 8]
            for mode in MODES:
                vals = {f: {} for f in FIELDS}
                for cdir in cdirs:
                    base = args.ablation / "judging" / judge / "main6" / cdir.name
                    comp = base / "compacted" / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"
                    cond = base / mode / ("as_sent" if mode == "compacted" else view) / "judge" / "mindeval2" / "verdicts.jsonl"
                    if not (comp.is_file() and cond.is_file()):
                        continue
                    pc, pd = parts(load_verdicts(comp)), parts(load_verdicts(cond))
                    for name, f in FIELDS.items():
                        s = {(m, e): d.get(f, 0.0) for (m, e), d in pd.items() if e != "ep000"}
                        s.update({(m, e): d.get(f, 0.0) for (m, e), d in pc.items() if e == "ep000"})
                        vals[name][cdir.name] = s
                if mode in BY_COUNSELOR_MODES:
                    for c, s in vals["MQM"].items():
                        for e in eps6:
                            r = pool({c: {m: v for (m, ee), v in s.items() if ee == e}})
                            if r.get("diff") is not None:
                                by_c.append({"judge": jl, "view": view, "mode": mode, "counselor": c, "kind": "session",
                                             "session": int(e[2:]) + 1, **{k: r[k] for k in ("diff", "lo", "hi")}})
                        step_c = pool({c: {m: s[m, eps6[1]] - s[m, eps6[0]] for (m, e) in s
                                           if e == eps6[0] and (m, eps6[1]) in s}})
                        trend_c = pool({c: {m: b for m in {m for (m, _) in s}
                                            if (b := slope([(int(e[2:]) + 1, s[m, e]) for e in eps6[1:] if (m, e) in s]))
                                            is not None}})
                        for kind, r in (("step", step_c), ("trend", trend_c)):
                            if r.get("diff") is not None:
                                by_c.append({"judge": jl, "view": view, "mode": mode, "counselor": c, "kind": kind,
                                             "session": None, **{k: r[k] for k in ("diff", "lo", "hi")}})
                for name in FIELDS:
                    if not vals[name]:
                        continue
                    per_ep, step, trend = series(vals[name], eps6)
                    lines.append(f"| {mode} | {name} | " + " | ".join(lvl(r) for _, r in per_ep)
                                 + f" | {fmt(step)} | {fmt(trend, 3)} |")
                    for e, r in per_ep:
                        if r:
                            traj.append({"set": "ablation", "judge": jl, "view": view, "mode": mode, "measure": name,
                                         "session": int(e[2:]) + 1, **{k: r[k] for k in ("diff", "lo", "hi", "n_counselors")}})
                    for kind, r in (("step", step), ("trend", trend)):
                        if r:
                            steps.append({"set": "ablation", "judge": jl, "view": view, "mode": mode, "measure": name,
                                          "kind": kind, **{k: r[k] for k in ("diff", "lo", "hi", "n_counselors", "above",
                                                                             "below")}})
            lines.append("")
    # Labelers (sessions 2-6 only: session 1 has no earlier session to repeat or use).
    lines += ["## Labelers (ablation, sessions 2-6)", "",
              "| condition | measure | " + " | ".join(f"s{i + 1}" for i in range(1, 6)) + " | trend s2-6 |",
              "|---|---|" + "---|" * 6]
    for mode in MODES:
        for measure, prompt in (("member repeats", "member_side_v1"), ("use of relevant notes", "continuity_v1")):
            sc = {}
            for cdir in cdirs:
                lab = load_labels(args.labels / prompt / cdir.name / mode)
                s = {}
                for (m, e), p in lab.items():
                    if prompt == "member_side_v1":
                        s[m, e] = float(len(p.get("repeats") or []))
                    else:
                        ops = p.get("opportunities") or []
                        if ops:
                            s[m, e] = sum(o.get("status") == "used" for o in ops) / len(ops)
                if s:
                    sc[cdir.name] = s
            if not sc:
                continue
            per_ep = [(e, pooled_row({c: {m: v for (m, ee), v in s.items() if ee == e} for c, s in sc.items()}))
                      for e in eps6[1:]]
            trend = {}
            for c, s in sc.items():
                trend[c] = {m: b for m in {m for (m, _) in s}
                            if (b := slope([(int(e[2:]) + 1, s[m, e]) for e in eps6[1:] if (m, e) in s])) is not None}
            tr = pooled_row(trend)
            lines.append(f"| {mode} | {measure} | " + " | ".join(lvl(r) for _, r in per_ep) + f" | {fmt(tr, 3)} |")
            for e, r in per_ep:
                if r:
                    traj.append({"set": "ablation", "judge": "labelers", "view": "-", "mode": mode, "measure": measure,
                                 "session": int(e[2:]) + 1, **{k: r[k] for k in ("diff", "lo", "hi", "n_counselors")}})
            if tr:
                steps.append({"set": "ablation", "judge": "labelers", "view": "-", "mode": mode, "measure": measure,
                              "kind": "trend", **{k: tr[k] for k in ("diff", "lo", "hi", "n_counselors", "above", "below")}})
    lines.append("")
    # Paper runs: sessions 1-3.
    if args.runs_root:
        root = runs_root(args.runs_root)
        runs, judges = registry(root)
        eps3 = ["ep000", "ep001", "ep002"]
        lines += ["## Paper runs (sessions 1-3, 18 counselors)", "",
                  "| judge | measure | s1 | s2 | s3 | step s1→s2 | change s2→s3 |", "|---|---|---|---|---|---|---|"]
        for judge, jl in PAPER.items():
            vals = {f: {} for f in FIELDS}
            for r in runs:
                path = judge_dir(root, r, judge, judges) / "verdicts.jsonl"
                if not path.is_file():
                    continue
                p = parts(load_verdicts(path), EXCLUDE.get(r["key"], set()))
                for name, f in FIELDS.items():
                    vals[name][r["key"]] = {k: d.get(f, 0.0) for k, d in p.items()}
            for name in FIELDS:
                per_ep = [(e, pooled_row({c: {m: v for (m, ee), v in s.items() if ee == e} for c, s in vals[name].items()}))
                          for e in eps3]
                st = pooled_row({c: {m: s[m, "ep001"] - s[m, "ep000"] for (m, e) in s if e == "ep000" and (m, "ep001") in s}
                                 for c, s in vals[name].items()})
                nx = pooled_row({c: {m: s[m, "ep002"] - s[m, "ep001"] for (m, e) in s if e == "ep001" and (m, "ep002") in s}
                                 for c, s in vals[name].items()})
                lines.append(f"| {jl} | {name} | " + " | ".join(lvl(r) for _, r in per_ep) + f" | {fmt(st)} | {fmt(nx)} |")
                for e, r in per_ep:
                    if r:
                        traj.append({"set": "paper", "judge": jl, "view": "as sent", "mode": "paper", "measure": name,
                                     "session": int(e[2:]) + 1, **{k: r[k] for k in ("diff", "lo", "hi", "n_counselors")}})
                for kind, r in (("step", st), ("s2→s3", nx)):
                    if r:
                        steps.append({"set": "paper", "judge": jl, "view": "as sent", "mode": "paper", "measure": name,
                                      "kind": kind, **{k: r[k] for k in ("diff", "lo", "hi", "n_counselors", "above",
                                                                         "below")}})
        lines.append("")
    lines += ["## Per counselor: session MQM trend over sessions 2-6 (per session), with and without memory", ""]
    for jl in ("DeepSeek-V4.1", "MindJudge-27B"):
        for view in ("reference", "as_sent"):
            tr = {(r["counselor"], r["mode"]): r for r in by_c
                  if r["judge"] == jl and r["view"] == view and r["kind"] == "trend"}
            st = {(r["counselor"], r["mode"]): r for r in by_c
                  if r["judge"] == jl and r["view"] == view and r["kind"] == "step"}
            cs = sorted({c for c, _ in tr})
            if not cs:
                continue
            lines += [f"### {jl}, {view.replace('_', ' ')}", "",
                      "| counselor | with memory: step s1→s2 | trend s2-6 | without: step | trend |", "|---|---|---|---|---|"]
            for c in cs:
                lines.append(f"| {c} | " + " | ".join(fmt(d.get((c, m)), 3) for m in BY_COUNSELOR_MODES
                                                      for d in (st, tr)) + " |")
            for m in BY_COUNSELOR_MODES:
                rs = [tr[c, m] for c in cs if (c, m) in tr]
                lines.append(f"- {m}: trend above zero for {sum(r['lo'] > 0 for r in rs)} of {len(rs)} counselors, "
                             f"below for {sum(r['hi'] < 0 for r in rs)}")
            lines.append("")
    for name, rows in (("trajectory.csv", traj), ("steps.csv", steps), ("by_counselor.csv", by_c)):
        with (args.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
