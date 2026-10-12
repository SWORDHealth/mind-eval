"""Session 1 run N times against N sessions of an arc: does a later session say more than another first session?

    python analysis/s1_replicates.py --ablation <memory_ablation dir> --counselors glm53_nvfp4[,...] --out <dir>

Per counselor: its 50 members' arc under the benchmark's memory (main6/<c>/compacted, sessions 1-6) and their session 1
run again under seeds 1-5 (s1rep/<c>/seed<k>; seed 0 is the arc's own session 1). Session 1 keeps the member's own
situation, affect and concealment whatever the seed, so a rerun changes only the conversation. Each judge reads both
designs on one serving: DeepSeek-V4.1-Flash on our own server (judging/deepseek_v4_own/), MindJudge-27B
(judging/mindjudge/). Scores are session MQM; parts are drivers.py's split of it by issue group.
  distance   per member, the mean squared difference from session 1 of sessions 2-6 and of reruns 1-5, and the
             ratio of the two means (1 if later sessions were reruns); likewise the spread among sessions 2-6 and
             among reruns 1-5 around their own mean, which leaves the step from session 1 out.
  change     by N (2-6): the mean of sessions 2..N minus session 1, beside the mean of reruns 1..N-1 minus session 1
             (0 in expectation), and their difference.
  precision  by N (1-6): the member-bootstrap standard error of the counselor's mean over N sessions and over N runs of
             session 1.
  content    each issue group's part of the score: session 1, the reruns, sessions 2-6; and sessions 2-6 minus reruns.
  ranking    with three or more counselors, at N = 3 and 6: Kendall tau between the counselors' order by N sessions
             and by N runs of session 1, against each design's agreement with itself across random halves of the
             members (the cross-design tau is computed across halves too, so all three carry the same noise).
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
from common import BOOT, ci, git_describe, kendall_tau_b, load_verdicts, mean, member_draws  # noqa: E402
from drivers import GROUPS, parts  # noqa: E402

JUDGES = {"deepseek_v4": ("deepseek_v4_own", "DeepSeek-V4.1 (one serving)"), "mindjudge": ("mindjudge", "MindJudge-27B")}
ARC = range(1, 7)
FIELDS = {"MQM": "_mqm", **{f"{g} part": f"group:{g}" for g in GROUPS}}


def verdicts(root: Path, jroot: str, rel: str) -> Path:
    return root / "judging" / jroot / rel / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"


def load(root: Path, jroot: str, c: str, seeds: list[int]) -> tuple[dict, dict, list[Path]]:
    """arc: member -> session (1-6) -> fields; reruns: member -> seed (0 = the arc's session 1) -> fields."""
    arc, reruns, used = {}, {}, []
    p = verdicts(root, jroot, f"main6/{c}/compacted")
    if p.is_file():
        used.append(p)
        for (m, e), d in parts(load_verdicts(p)).items():
            arc.setdefault(m, {})[int(e[2:]) + 1] = d
    for k in seeds:
        p = verdicts(root, jroot, f"s1rep/{c}/seed{k}")
        if p.is_file():
            used.append(p)
            for (m, e), d in parts(load_verdicts(p)).items():
                if e == "ep000":
                    reruns.setdefault(m, {})[k] = d
    for m in reruns:
        if 1 in arc.get(m, {}):
            reruns[m][0] = arc[m][1]
    return arc, reruns, used


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--counselors", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cs = args.counselors.split(",")
    seeds = [int(k) for k in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    lines = ["# Session 1 run N times against N sessions", "",
             f"Counselors: {', '.join(cs)}. Arc: sessions 1-6 under the benchmark's memory. Reruns: session 1 under "
             f"seeds {args.seeds} (seed 0 is the arc's session 1). Means over counselors; 95% intervals resample members "
             "within each counselor.", ""]

    def row(judge, c, analysis, measure, n, r):
        if r and r.get("diff") is not None:
            rows.append({"judge": judge, "counselor": c, "analysis": analysis, "measure": measure, "N": n,
                         "value": r["diff"], "lo": r["lo"], "hi": r["hi"], "n_counselors": r.get("n_counselors")})

    for judge, (jroot, jl) in JUDGES.items():
        data = {}
        for c in cs:
            arc, reruns, used = load(args.ablation, jroot, c, seeds)
            inputs += [str(p) for p in used]
            if arc and reruns:
                data[c] = (arc, reruns)
        if not data:
            continue
        n_m = {c: sum(1 for m in r if all(k in r[m] for k in [0, *seeds])) for c, (_, r) in data.items()}
        lines += [f"## {jl}", "", "Members with every rerun judged: " + ", ".join(f"{c} {n}" for c, n in n_m.items()),
                  ""]

        # distance and spread
        lines += ["### How far from session 1, and how spread out", "",
                  "| counselor | sessions 2-6: mean sq. distance from s1 | reruns 1-5 | ratio | sessions 2-6: spread | "
                  "reruns 1-5: spread | ratio |", "|---|---|---|---|---|---|---|"]
        for c, (arc, reruns) in data.items():
            ms = [m for m in arc if all(s in arc[m] for s in ARC) and all(k in reruns.get(m, {}) for k in [0, *seeds])]
            if len(ms) < 10:
                continue
            da = {m: mean((f(arc[m][s], "_mqm") - f(arc[m][1], "_mqm")) ** 2 for s in ARC[1:]) for m in ms}
            dr = {m: mean((f(reruns[m][k], "_mqm") - f(reruns[m][0], "_mqm")) ** 2 for k in seeds) for m in ms}

            def spread(xs):
                mu = mean(xs)
                return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
            sa = {m: spread([f(arc[m][s], "_mqm") for s in ARC[1:]]) for m in ms}
            sr = {m: spread([f(reruns[m][k], "_mqm") for k in seeds]) for m in ms}
            ra, rs = ratio_ci(da, dr), ratio_ci(sa, sr)
            lines.append(f"| {c} ({len(ms)}) | {mean(da.values()):.3f} | {mean(dr.values()):.3f} | {ra[0]:.2f} "
                         f"[{ra[1]:.2f}, {ra[2]:.2f}] | {mean(sa.values()):.3f} | {mean(sr.values()):.3f} | "
                         f"{rs[0]:.2f} [{rs[1]:.2f}, {rs[2]:.2f}] |")
            for name, r in (("distance ratio", ra), ("spread ratio", rs)):
                rows.append({"judge": jl, "counselor": c, "analysis": "distance", "measure": name, "N": 6,
                             "value": r[0], "lo": r[1], "hi": r[2], "n_counselors": 1})
        lines.append("")

        # change from session 1
        lines += ["### Change from session 1", "",
                  "| N | sessions 2..N minus s1 | reruns 1..N-1 minus s1 | difference |", "|---|---|---|---|"]
        for n in range(2, 7):
            ca, cr, cd = {}, {}, {}
            for c, (arc, reruns) in data.items():
                ms = [m for m in arc if all(s in arc[m] for s in range(1, n + 1))
                      and all(k in reruns.get(m, {}) for k in range(n))]
                a = {m: mean(f(arc[m][s], "_mqm") for s in range(2, n + 1)) - f(arc[m][1], "_mqm") for m in ms}
                r = {m: mean(f(reruns[m][k], "_mqm") for k in range(1, n)) - f(reruns[m][0], "_mqm") for m in ms}
                ca[c], cr[c], cd[c] = a, r, {m: a[m] - r[m] for m in ms}
            pa, pr, pd = pool(ca), pool(cr), pool(cd)
            lines.append(f"| {n} | {fmt(pa)} | {fmt(pr)} | {fmt(pd)} |")
            for name, r in (("sessions", pa), ("reruns", pr), ("difference", pd)):
                row(jl, "mean", "change", name, n, r)
        lines.append("")

        # precision
        lines += ["### Precision: standard error of the counselor's mean (member bootstrap)", "",
                  "| counselor | " + " | ".join(f"N={n} sessions / reruns" for n in ARC) + " |",
                  "|---|" + "---|" * 6]
        for c, (arc, reruns) in data.items():
            cells = []
            for n in ARC:
                ms = [m for m in arc if all(s in arc[m] for s in range(1, n + 1))
                      and all(k in reruns.get(m, {}) for k in range(n))]
                if len(ms) < 10:
                    cells.append("-")
                    continue
                a = se({m: mean(f(arc[m][s], "_mqm") for s in range(1, n + 1)) for m in ms})
                r = se({m: mean(f(reruns[m][k], "_mqm") for k in range(n)) for m in ms})
                cells.append(f"{a:.3f} / {r:.3f}")
                for name, v in (("se sessions", a), ("se reruns", r)):
                    rows.append({"judge": jl, "counselor": c, "analysis": "precision", "measure": name, "N": n,
                                 "value": v, "lo": None, "hi": None, "n_counselors": 1})
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

        # content
        lines += ["### What the score is made of (per session)", "",
                  "| part | session 1 | reruns 1-5 | sessions 2-6 | sessions 2-6 minus reruns |", "|---|---|---|---|---|"]
        for name, fld in FIELDS.items():
            s1, rr, ss, dd = {}, {}, {}, {}
            for c, (arc, reruns) in data.items():
                ms = [m for m in arc if 1 in arc[m] and any(s in arc[m] for s in ARC[1:])
                      and any(k in reruns.get(m, {}) for k in seeds)]
                s1[c] = {m: f(arc[m][1], fld) for m in ms}
                rr[c] = {m: mean(f(reruns[m][k], fld) for k in seeds if k in reruns[m]) for m in ms}
                ss[c] = {m: mean(f(arc[m][s], fld) for s in ARC[1:] if s in arc[m]) for m in ms}
                dd[c] = {m: ss[c][m] - rr[c][m] for m in ms}
            p1, pr, ps, pd = pool(s1), pool(rr), pool(ss), pool(dd)
            lv = lambda r: "-" if not r or r.get("diff") is None else f"{r['diff']:.3f}"  # noqa: E731
            lines.append(f"| {name} | {lv(p1)} | {lv(pr)} | {lv(ps)} | {fmt(pd)} |")
            for nm, r in (("session 1", p1), ("reruns", pr), ("sessions 2-6", ps), ("sessions minus reruns", pd)):
                row(jl, "mean", "content", f"{name}: {nm}", 6, r)
        lines.append("")

        # ranking
        if len(data) >= 3:
            lines += ["### Do N sessions and N runs of session 1 order the counselors alike?", "",
                      "| N | tau, sessions vs reruns (all members) | split-half tau: sessions | reruns | across designs | "
                      "across minus within |", "|---|---|---|---|---|---|"]
            for n in (3, 6):
                common_ms = None
                for c, (arc, reruns) in data.items():
                    ms = {m for m in arc if all(s in arc[m] for s in range(1, n + 1))
                          and all(k in reruns.get(m, {}) for k in range(n))}
                    common_ms = ms if common_ms is None else common_ms & ms
                ms = sorted(common_ms or [])
                if len(ms) < 20:
                    continue
                cl = sorted(data)
                A = {c: {m: mean(f(data[c][0][m][s], "_mqm") for s in range(1, n + 1)) for m in ms} for c in cl}
                R = {c: {m: mean(f(data[c][1][m][k], "_mqm") for k in range(n)) for m in ms} for c in cl}
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
            lines += [f"Ranking needs three or more counselors ({len(data)} here).", ""]

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
