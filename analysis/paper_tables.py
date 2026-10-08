"""LaTeX tables for the later-sessions results, from the outputs of c1_ranking.py and c2_issues.py.

    python analysis/paper_tables.py --c1 <c1 dir> --c1_shared <c1_shared dir> --c2 <c2 dir> --out <dir>

Writes `tab_later_ranking.tex` (per judge: Kendall tau between the session-1 ranking and sessions 2-3, the
counselor x session interaction against its permutation null, and the growth of the spread between counselors)
and `tab_later_errors.tex` (the issue categories whose rate changes most from session 1 to sessions 2-3, under each
judge; $\dagger$ marks the later-only categories of c2_issues.py), and copies the input manifests beside them.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

JUDGES = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview"}
CATEGORIES = {"missed_shared_context": "Missed shared context", "gives_bad_advice": "Gives bad advice",
              "uncritical_endorsement": "Uncritical endorsement", "false_memory": "False memory",
              "treating_perspective_as_fact": "Perspective as fact", "anthropomorphizing": "Anthropomorphizing",
              "overstatement": "Overstatement", "critical": "Critical severity"}


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ranking(c1: Path, c1s: Path) -> str:
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Judge & $\tau$(s1, s2--3) & SD of change (null 95\%) & $p$ & Spread s2--3 / s1 \\", r"\midrule"]
    for sub, label in ((c1, "All 18 counselors"), (c1s, "11 counselors on one schedule")):
        tau = {(r["judge"], r["contrast"]): r for r in rows(sub / "taus.csv")}
        inter = {(r["judge"], r["change"]): r for r in rows(sub / "interaction.csv")}
        spread = {(r["judge"], r["window"]): r for r in rows(sub / "spread.csv")}
        out.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{label}}}}} \\")
        for j, name in JUDGES.items():
            if (j, "s1~later") not in tau:
                continue
            t, i, s = tau[j, "s1~later"], inter[j, "later-s1"], spread[j, "later"]
            out.append(f"{name} & {float(t['tau']):.2f} [{float(t['lo']):.2f}, {float(t['hi']):.2f}] & "
                       f"{float(i['sd_of_change_over_counselors']):.2f} ({float(i['null_95']):.2f}) & "
                       f"{pval(float(i['p_perm']))} & {float(s['ratio_to_s1']):.2f} "
                       f"[{float(s['ratio_lo']):.2f}, {float(s['ratio_hi']):.2f}] \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def errors(c2: Path) -> str:
    pooled = {(r["judge"], r["category"]): r for r in rows(c2 / "pooled.csv")}
    js = [j for j in JUDGES if any(k[0] == j for k in pooled)]
    out = [r"\begin{tabular}{l" + "c" * len(js) + "}", r"\toprule",
           "Category & " + " & ".join(JUDGES[j] for j in js) + r" \\", r"\midrule"]
    for cat, name in CATEGORIES.items():
        cells = []
        for j in js:
            r = pooled.get((j, cat))
            if r is None or float(r["rate_s1"]) == 0 and float(r["rate_later"]) == 0:
                cells.append("--")
                continue
            mark = r"$^\dagger$" if r["later_only"] == "True" else ""
            def fmt(x):
                return "$<$0.1" if 0 < 100 * x < 0.05 else f"{100 * x:.1f}"
            cells.append(f"{fmt(float(r['rate_s1']))} $\\to$ {fmt(float(r['rate_later']))}{mark}")
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c1", required=True, type=Path)
    ap.add_argument("--c1_shared", required=True, type=Path)
    ap.add_argument("--c2", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tab_later_ranking.tex").write_text(ranking(args.c1, args.c1_shared))
    (args.out / "tab_later_errors.tex").write_text(errors(args.c2))
    for name, d in (("c1", args.c1), ("c1_shared", args.c1_shared), ("c2", args.c2)):
        shutil.copyfile(d / "manifest.json", args.out / f"manifest_{name}.json")
    print(json.dumps(sorted(p.name for p in args.out.iterdir())))


if __name__ == "__main__":
    main()
