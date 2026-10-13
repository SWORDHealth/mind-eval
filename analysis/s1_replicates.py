"""Session 1 run N times against N sessions of an arc: does a later session say more than another first session?

    python analysis/s1_replicates.py --ablation <memory_ablation dir> --counselors glm53_nvfp4[,...] --out <dir>
        [--arcs paper --runs_root <dir> --seeds 1,2,3]

--arcs paper: the N sessions are the paper runs' sessions 1-3 (mindeval_qwen/<c>, whose session 1 the reruns repeat:
same members and transcript), as every paper judge scored them (judge dirs from runs.yaml), all generated against one
patient server; the reruns are judged where verdicts exist (DeepSeek on our server, MindJudge, and GPT-6.1 Sol or Hy4
under judging/<judge>/s1rep/ if they are run). N runs to 3. GLM-5.3 NVFP4, Sonnet 5 and Terra repeat ablation sources
that are not paper rows: their 3-session source runs as judge_sources.sh judged them stand in (DeepSeek and MindJudge
only), and the DeepSeek ranking leaves them out (other servings).

Per counselor: its 50 members' arc under the benchmark's memory (main6/<c>/compacted, sessions 1-6) and their session 1
run again under seeds 1-5 (s1rep/<c>/seed<k>). Session 1 keeps the member's own situation, affect and concealment
whatever the seed, so a rerun changes only the conversation. Scores are session MQM; parts are drivers.py's split of it
by issue group.

The reruns were generated against another patient server than the arcs, so the rerun design is kept to itself: rerun 1
is its session 1, reruns 2-5 its repeats, and the arc's own session 1 (seed 0) only measures the shift between the two
(serving). Judges, one serving per design:
  DeepSeek-V4.1   arcs: each counselor's single-serving verdicts (judging/deepseek_v4/single/; the API counselors' were
                  made on our own server, judging/deepseek_v4/main6/); reruns: our own server (judging/deepseek_v4_own/
                  s1rep/), where the arc's session 1 is also judged again (seed0), so both shifts can be told apart.
  MindJudge-27B   one server for everything (judging/mindjudge/).
  serving    seed 0 judged on the reruns' serving minus on the arc's (the judge's shift; DeepSeek, free counselors), and
             the mean of reruns 1-5 minus seed 0 on one judge serving (the patient server's shift).
  distance   per member, the mean squared difference from session 1 of sessions 2-6, and of reruns 2-5 from rerun 1, and
             the ratio of the two means (1 if later sessions were reruns); likewise the spread among sessions 2-6 and
             among reruns 1-5 around their own mean, which leaves any step from session 1 out.
  change     by N (2-5): the mean of sessions 2..N minus session 1, beside the mean of reruns 2..N minus rerun 1 (0 in
             expectation), and their difference.
  precision  by N (1-5): the member-bootstrap standard error of the counselor's mean over N sessions and over N reruns.
  content    each issue group's part of the score as a change: sessions 2-6 minus session 1, reruns 2-5 minus rerun 1,
             and the difference.
  ranking    with three or more counselors, at N = 3 and 5: Kendall tau between the counselors' order by N sessions and
             by N reruns, against each design's agreement with itself across random halves of the members (the
             cross-design tau is across halves too, so all three carry the same noise). DeepSeek leaves out counselors
             whose arcs were judged on another serving (--rank_exclude).
Intervals: 95%, members resampled (within each counselor when pooled over counselors).
Writes s1_replicates.csv, report.md, manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import pool  # noqa: E402
from common import (  # noqa: E402
    BOOT,
    ci,
    git_describe,
    judge_dir,
    kendall_tau_b,
    load_verdicts,
    mean,
    member_draws,
    registry,
    runs_root,
)
from drivers import GROUPS, parts  # noqa: E402

JUDGES = {"deepseek_v4": "DeepSeek-V4.1", "mindjudge": "MindJudge-27B", "gpt61_sol": "GPT-6.1 Sol", "hy4": "Hy4-preview"}
#: Each judge's key in runs.yaml (paper runs) and its directory under the ablation's judging/ for the reruns.
PAPER_KEY = {"deepseek_v4": "deepseek_v4", "mindjudge": "own", "gpt61_sol": "gpt61_sol", "hy4": "hy4"}
RERUN_DIR = {"deepseek_v4": "deepseek_v4_own", "mindjudge": "mindjudge", "gpt61_sol": "gpt61_sol", "hy4": "hy4"}
#: --arcs paper for counselors whose reruns repeat an ablation source that is not a paper row (the paper's GLM-5.3 and
#: API rows are another harness): their 3-session source runs, as judge_sources.sh judged them, stand in.
SOURCE_ARC = {"glm53_nvfp4": "glm53", "claude_sonnet5": "claude_sonnet5", "gpt56terra": "gpt56terra"}
FIELDS = {"MQM": "_mqm", **{f"{g} part": f"group:{g}" for g in GROUPS}}


def verdicts(view: Path) -> Path:
    return view / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"


def paths(root: Path, judge: str, c: str, paper: dict | None = None) -> tuple[Path, Path | None, Path]:
    """The arc's verdicts (the file), session 1 judged again on the reruns' serving (None: the same serving), the
    reruns' root. `paper`: {"root", "runs", "judges"} for paper-run arcs."""
    j = root / "judging"
    rr = j / RERUN_DIR[judge] / "s1rep" / c
    if paper is not None:
        if c in SOURCE_ARC:
            arc = verdicts(j / judge / "source" / SOURCE_ARC[c]) if judge in ("deepseek_v4", "mindjudge") else Path("-")
        else:
            run = next((r for r in paper["runs"] if r["key"] == c), None)
            arc = judge_dir(paper["root"], run, PAPER_KEY[judge], paper["judges"]) / "verdicts.jsonl" if run \
                else Path("-")
        seed0 = j / "deepseek_v4_own" / "s1rep" / c / "seed0"
        return arc, seed0 if judge == "deepseek_v4" and verdicts(seed0).is_file() else None, rr
    if judge not in ("deepseek_v4", "mindjudge"):
        return Path("-"), None, rr
    if judge == "deepseek_v4":
        single = j / "deepseek_v4" / "single" / c / "compacted"
        arc = single if verdicts(single).is_file() else j / "deepseek_v4" / "main6" / c / "compacted"
        seed0 = j / "deepseek_v4_own" / "s1rep" / c / "seed0"
        return verdicts(arc), seed0 if verdicts(seed0).is_file() else None, rr
    return verdicts(j / judge / "main6" / c / "compacted"), None, rr


def load(root: Path, judge: str, c: str, seeds: list[int], paper: dict | None = None) -> tuple[dict, dict, dict, dict,
                                                                                               list[Path]]:
    """arc: member -> session (1-6) -> fields; reruns: member -> seed -> fields; s0: member -> the arc's session 1 on the
    reruns' judge serving; s1_other: the same on the arc's serving, where the two differ."""
    arc_file, seed0_dir, rr = paths(root, judge, c, paper)
    arc, reruns, s0, s1_other, used = {}, {}, {}, {}, []
    if arc_file.is_file():
        used.append(arc_file)
        for (m, e), d in parts(load_verdicts(arc_file)).items():
            arc.setdefault(m, {})[int(e[2:]) + 1] = d
    for k in seeds:
        p = verdicts(rr / f"seed{k}")
        if p.is_file():
            used.append(p)
            for (m, e), d in parts(load_verdicts(p)).items():
                if e == "ep000":
                    reruns.setdefault(m, {})[k] = d
    if seed0_dir is not None:
        used.append(verdicts(seed0_dir))
        for (m, e), d in parts(load_verdicts(verdicts(seed0_dir))).items():
            if e == "ep000":
                s0[m] = d
                if 1 in arc.get(m, {}):
                    s1_other[m] = arc[m][1]
    else:
        s0 = {m: a[1] for m, a in arc.items() if 1 in a}
    return arc, reruns, s0, s1_other, used


def f(d: dict, field: str) -> float:
    return d.get(field, 0.0)


def ratio_ci(num: dict[str, float], den: dict[str, float]) -> tuple[float, float, float]:
    ms = sorted(set(num) & set(den))
    r = lambda d: mean(num[m] for m in d) / mean(den[m] for m in d) if mean(den[m] for m in d) else math.nan  # noqa: E731
    lo, hi = ci([r(d) for d in member_draws(ms, BOOT)])
    return r(ms), lo, hi


def se(values: dict[str, float]) -> float:
    draws = [mean(values[m] for m in d) for d in member_draws(sorted(values), BOOT)]
    mu = mean(draws)
    return math.sqrt(sum((x - mu) ** 2 for x in draws) / (len(draws) - 1))


def fmt(r: dict | None, nd: int = 3) -> str:
    return "-" if not r or r.get("diff") is None else f"{r['diff']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"


def spread(xs: list[float]) -> float:
    mu = mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--counselors", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--rank_exclude", default="claude_sonnet5,gpt56terra",
                    help="left out of the DeepSeek ranking: their arcs were judged on another serving")
    ap.add_argument("--arcs", choices=("ablation", "paper"), default="ablation")
    ap.add_argument("--runs_root", help="paper runs' root (else MINDEVAL_RUNS_ROOT); for --arcs paper")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cs = args.counselors.split(",")
    seeds = [int(k) for k in args.seeds.split(",")]
    paper = None
    if args.arcs == "paper":
        root = runs_root(args.runs_root)
        runs, judges_reg = registry(root)
        paper = {"root": root, "runs": runs, "judges": judges_reg}
        args.rank_exclude = ",".join(SOURCE_ARC)  # DeepSeek: their source runs were judged on other servings
    ARC = range(1, 4) if paper else range(1, 7)
    first, later = seeds[0], seeds[1:]
    nmax = min(len(seeds), len(ARC))
    args.out.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    lines = ["# Session 1 run N times against N sessions", "",
             f"Counselors: {', '.join(cs)}. Arc: " + ("the paper runs' sessions 1-3" if paper else
                                                     "sessions 1-6 under the benchmark's memory") + ". Reruns: session 1 under "
             f"seeds {args.seeds}, against another patient server than the arcs; rerun {first} is the rerun design's "
             "session 1. Means over counselors; 95% intervals resample members within each counselor.", ""]

    def row(judge, c, analysis, measure, n, r):
        if r and r.get("diff") is not None:
            rows.append({"judge": judge, "counselor": c, "analysis": analysis, "measure": measure, "N": n,
                         "value": r["diff"], "lo": r["lo"], "hi": r["hi"], "n_counselors": r.get("n_counselors")})

    for judge, jl in JUDGES.items():
        data, s0s, others = {}, {}, {}
        for c in cs:
            arc, reruns, s0, s1_other, used = load(args.ablation, judge, c, seeds, paper)
            inputs += [str(p) for p in used]
            if arc and reruns:
                data[c], s0s[c], others[c] = (arc, reruns), s0, s1_other
        if not data:
            continue
        complete = {c: [m for m in r if all(k in r[m] for k in seeds)] for c, (_, r) in data.items()}
        if not any(complete.values()):
            continue
        lines += [f"## {jl}", "", "Members with every rerun judged: "
                  + ", ".join(f"{c} {len(ms)}" for c, ms in complete.items()), ""]

        # serving
        judge_shift = {c: {m: f(s0s[c][m], "_mqm") - f(o[m], "_mqm") for m in o} for c, o in others.items() if o}
        patient_shift = {c: {m: mean(f(r[m][k], "_mqm") for k in seeds) - f(s0s[c][m], "_mqm")
                             for m in complete[c] if m in s0s[c]} for c, (_, r) in data.items()}
        lines += ["### Serving shifts (session 1, same members and situations)", "",
                  "| counselor | judge: s1 on reruns' serving minus arc's | patient: mean of reruns minus s1 |",
                  "|---|---|---|"]
        for c in data:
            rj = pool({c: judge_shift[c]}) if judge_shift.get(c) else None
            rp = pool({c: patient_shift[c]}) if patient_shift.get(c) else None
            lines.append(f"| {c} | {fmt(rj)} | {fmt(rp)} |")
            row(jl, c, "serving", "judge shift", 1, rj)
            row(jl, c, "serving", "patient shift", 1, rp)
        rj, rp = pool({c: d for c, d in judge_shift.items() if d}), pool({c: d for c, d in patient_shift.items() if d})
        lines += [f"| mean | {fmt(rj)} | {fmt(rp)} |", ""]
        row(jl, "mean", "serving", "judge shift", 1, rj)
        row(jl, "mean", "serving", "patient shift", 1, rp)

        # distance and spread
        lines += ["### How far from session 1, and how spread out", "",
                  f"| counselor | sessions 2-{ARC[-1]}: mean sq. distance from s1 | reruns {later[0]}-{later[-1]} from "
                  f"rerun {first} | ratio | sessions 2-{ARC[-1]}: spread | reruns {seeds[0]}-{seeds[-1]}: spread | ratio |",
                  "|---|---|---|---|---|---|---|"]
        for c, (arc, reruns) in data.items():
            ms = [m for m in complete[c] if all(s in arc.get(m, {}) for s in ARC)]
            if len(ms) < 10:
                continue
            da = {m: mean((f(arc[m][s], "_mqm") - f(arc[m][1], "_mqm")) ** 2 for s in ARC[1:]) for m in ms}
            dr = {m: mean((f(reruns[m][k], "_mqm") - f(reruns[m][first], "_mqm")) ** 2 for k in later) for m in ms}
            sa = {m: spread([f(arc[m][s], "_mqm") for s in ARC[1:]]) for m in ms}
            sr = {m: spread([f(reruns[m][k], "_mqm") for k in seeds]) for m in ms}
            ra, rs = ratio_ci(da, dr), ratio_ci(sa, sr)
            lines.append(f"| {c} ({len(ms)}) | {mean(da.values()):.3f} | {mean(dr.values()):.3f} | {ra[0]:.2f} "
                         f"[{ra[1]:.2f}, {ra[2]:.2f}] | {mean(sa.values()):.3f} | {mean(sr.values()):.3f} | "
                         f"{rs[0]:.2f} [{rs[1]:.2f}, {rs[2]:.2f}] |")
            for name, r in (("distance ratio", ra), ("spread ratio", rs)):
                rows.append({"judge": jl, "counselor": c, "analysis": "distance", "measure": name, "N": ARC[-1],
                             "value": r[0], "lo": r[1], "hi": r[2], "n_counselors": 1})
        lines.append("")

        # change from session 1
        lines += ["### Change from session 1", "",
                  "| N | sessions 2..N minus s1 | reruns 2..N minus rerun 1 | difference |", "|---|---|---|---|"]
        for n in range(2, nmax + 1):
            ca, cr, cd = {}, {}, {}
            for c, (arc, reruns) in data.items():
                ms = [m for m in complete[c] if all(s in arc.get(m, {}) for s in range(1, n + 1))]
                a = {m: mean(f(arc[m][s], "_mqm") for s in range(2, n + 1)) - f(arc[m][1], "_mqm") for m in ms}
                r = {m: mean(f(reruns[m][k], "_mqm") for k in seeds[1:n]) - f(reruns[m][first], "_mqm") for m in ms}
                ca[c], cr[c], cd[c] = a, r, {m: a[m] - r[m] for m in ms}
            pa, pr, pd = pool(ca), pool(cr), pool(cd)
            lines.append(f"| {n} | {fmt(pa)} | {fmt(pr)} | {fmt(pd)} |")
            for name, r in (("sessions", pa), ("reruns", pr), ("difference", pd)):
                row(jl, "mean", "change", name, n, r)
        lines.append("")

        # precision
        lines += ["### Precision: standard error of the counselor's mean (member bootstrap)", "",
                  "| counselor | " + " | ".join(f"N={n} sessions / reruns" for n in range(1, nmax + 1)) + " |",
                  "|---|" + "---|" * nmax]
        for c, (arc, reruns) in data.items():
            cells = []
            for n in range(1, nmax + 1):
                ms = [m for m in complete[c] if all(s in arc.get(m, {}) for s in range(1, n + 1))]
                if len(ms) < 10:
                    cells.append("-")
                    continue
                a = se({m: mean(f(arc[m][s], "_mqm") for s in range(1, n + 1)) for m in ms})
                r = se({m: mean(f(reruns[m][k], "_mqm") for k in seeds[:n]) for m in ms})
                cells.append(f"{a:.3f} / {r:.3f}")
                for name, v in (("se sessions", a), ("se reruns", r)):
                    rows.append({"judge": jl, "counselor": c, "analysis": "precision", "measure": name, "N": n,
                                 "value": v, "lo": None, "hi": None, "n_counselors": 1})
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

        # content
        lines += ["### What the score is made of: change from session 1, per session", "",
                  f"| part | session 1 (arc) | sessions 2-{ARC[-1]} minus s1 | reruns {later[0]}-{later[-1]} minus rerun {first} "
                  "| difference |", "|---|---|---|---|---|"]
        for name, fld in FIELDS.items():
            s1, ss, rr, dd = {}, {}, {}, {}
            for c, (arc, reruns) in data.items():
                ms = [m for m in complete[c] if 1 in arc.get(m, {}) and any(s in arc[m] for s in ARC[1:])]
                s1[c] = {m: f(arc[m][1], fld) for m in ms}
                ss[c] = {m: mean(f(arc[m][s], fld) for s in ARC[1:] if s in arc[m]) - f(arc[m][1], fld) for m in ms}
                rr[c] = {m: mean(f(reruns[m][k], fld) for k in later) - f(reruns[m][first], fld) for m in ms}
                dd[c] = {m: ss[c][m] - rr[c][m] for m in ms}
            p1, ps, pr, pd = pool(s1), pool(ss), pool(rr), pool(dd)
            lv = "-" if p1.get("diff") is None else f"{p1['diff']:.3f}"
            lines.append(f"| {name} | {lv} | {fmt(ps)} | {fmt(pr)} | {fmt(pd)} |")
            for nm, r in (("session 1", p1), (f"sessions 2-{ARC[-1]} minus s1", ps), ("reruns minus rerun 1", pr),
                          ("difference", pd)):
                row(jl, "mean", "content", f"{name}: {nm}", 6, r)
        lines.append("")

        # ranking
        cl = [c for c in sorted(data) if not (judge == "deepseek_v4" and c in args.rank_exclude.split(","))]
        if len(cl) >= 3:
            lines += ["### Do N sessions and N runs of session 1 order the counselors alike?", "",
                      f"Counselors: {', '.join(cl)}.", "",
                      "| N | tau, sessions vs reruns (all members) | split-half tau: sessions | reruns | across designs | "
                      "across minus within |", "|---|---|---|---|---|---|"]
            for n in sorted({3, nmax}):
                common_ms = None
                for c in cl:
                    arc, _ = data[c]
                    ms = {m for m in complete[c] if all(s in arc.get(m, {}) for s in range(1, n + 1))}
                    common_ms = ms if common_ms is None else common_ms & ms
                ms = sorted(common_ms or [])
                if len(ms) < 20:
                    continue
                A = {c: {m: mean(f(data[c][0][m][s], "_mqm") for s in range(1, n + 1)) for m in ms} for c in cl}
                R = {c: {m: mean(f(data[c][1][m][k], "_mqm") for k in seeds[:n]) for m in ms} for c in cl}
                full = kendall_tau_b([mean(A[c].values()) for c in cl], [mean(R[c].values()) for c in cl])
                rng = random.Random(0)
                taa, trr, tar, gap = [], [], [], []
                for _ in range(args.splits):
                    sh = ms[:]
                    rng.shuffle(sh)
                    h1, h2 = sh[: len(sh) // 2], sh[len(sh) // 2:]
                    m_ = lambda D, h: [mean(D[c][m] for m in h) for c in cl]  # noqa: E731
                    a1, a2, r1, r2 = m_(A, h1), m_(A, h2), m_(R, h1), m_(R, h2)
                    aa, rr_ = kendall_tau_b(a1, a2), kendall_tau_b(r1, r2)
                    ar = (kendall_tau_b(a1, r2) + kendall_tau_b(r1, a2)) / 2
                    taa.append(aa)
                    trr.append(rr_)
                    tar.append(ar)
                    gap.append(ar - (aa + rr_) / 2)
                q = lambda xs: f"{mean(xs):+.2f} [{ci(xs)[0]:+.2f}, {ci(xs)[1]:+.2f}]"  # noqa: E731
                lines.append(f"| {n} | {full:+.2f} | {q(taa)} | {q(trr)} | {q(tar)} | {q(gap)} |")
                for nm, xs in (("split-half sessions", taa), ("split-half reruns", trr), ("across designs", tar),
                               ("across minus within", gap)):
                    lo, hi = ci(xs)
                    rows.append({"judge": jl, "counselor": "all", "analysis": "ranking", "measure": nm, "N": n,
                                 "value": mean(xs), "lo": lo, "hi": hi, "n_counselors": len(cl)})
                rows.append({"judge": jl, "counselor": "all", "analysis": "ranking", "measure": "tau all members",
                             "N": n, "value": full, "lo": None, "hi": None, "n_counselors": len(cl)})
            lines.append("")
        else:
            lines += [f"Ranking needs three or more counselors ({len(cl)} here).", ""]

    with (args.out / "s1_replicates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["judge", "counselor", "analysis", "measure", "N", "value", "lo", "hi",
                                           "n_counselors"])
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    runs = {}
    for c in cs:
        for rel in [f"main6/{c}/compacted", *(f"s1rep/{c}/seed{k}" for k in seeds)]:
            p = args.ablation / rel / "run.json"
            if p.is_file():
                r = json.loads(p.read_text())
                runs[rel] = {k: r.get(k) for k in ("git", "fingerprint", "seed", "sessions", "memory_mode")} | {
                    "patient": r["patient"]["model"], "patient_api_base": r["patient"]["api_base"],
                    "counselor": r["counselor"]["model"], "profiles_sha256": r["profiles"]["sha256"]}
    judges = {}
    for p in sorted(set(inputs)):
        j = Path(p).parent / "judge.json"
        if j.is_file():
            jj = json.loads(j.read_text())
            judges[p] = {k: jj.get(k) for k in ("model", "api_base", "prompt_sha256", "temperature", "max_tokens")}
    (args.out / "manifest.json").write_text(json.dumps(
        {"script": "analysis/s1_replicates.py", "analysis_git": git_describe(), "counselors": cs, "seeds": seeds,
         "runs": runs, "verdicts": judges}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
