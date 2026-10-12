r"""LaTeX tables for the later-sessions results, from the outputs of c1_ranking.py and c2_issues.py.

    python analysis/paper_tables.py --c1 <c1 dir> --c1_shared <c1_shared dir> --c2 <c2 dir> --out <dir>

Writes `tab_c3_judges.tex` and `tab_ablation.tex` when --c3 and --ablation are given, and `tab_later_ranking.tex` (per judge: Kendall tau between the session-1 ranking and sessions 2-3, the
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

from common import ablation_counselors

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


def c3_judges(c3: Path) -> str:
    """Judges against the replies the member had to answer by repeating something recorded."""
    rows = json.loads((c3 / "judge_crosstab.json").read_text())
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Judge & \multicolumn{2}{c}{Memory issue flagged} & \multicolumn{2}{c}{Major or Critical} \\",
           r" & repeat-forcing replies & other replies & repeat-forcing & other \\", r"\midrule"]
    for r in rows:
        out.append(f"{JUDGES[r['judge']]} & {100 * r['bad_memory_flag']:.0f}\\% & {100 * r['other_memory_flag']:.0f}\\% & "
                   f"{100 * r['bad_major']:.0f}\\% & {100 * r['other_major']:.0f}\\% \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def ablation(ab: Path) -> str:
    """MQM change from compacted, per counselor, mode, judge and view, then the mean across counselors."""
    rows = {(r["counselor"], r["judge"], r["mode"], r["view"], r["field"]): r for r in csv.DictReader((ab / "ablation.csv").open())}
    pooled = ({(r["judge"], r["mode"], r["view"], r["field"]): r for r in csv.DictReader((ab / "pooled.csv").open())}
              if (ab / "pooled.csv").is_file() else {})
    names = {k: c["label"] for k, c in ablation_counselors().items()}
    present = [c for c in names if any(k[0] == c for k in rows) and c not in EXCLUDE]

    def cell(r):
        return "--" if not r or not r["diff"] else f"{float(r['diff']):+.2f} [{float(r['lo']):+.2f}, {float(r['hi']):+.2f}]"

    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"Counselor & Memory & \multicolumn{2}{c}{MindJudge-27B} & \multicolumn{2}{c}{DeepSeek-V4.1} \\",
           r" & & as sent & reference & as sent & reference \\", r"\midrule"]
    for c in present:
        for mode in ("none", "full", "split"):
            cells = [cell(rows.get((c, judge, mode, view, "mqm")))
                     for judge in ("mindjudge", "deepseek_v4") for view in ("as_sent", "reference")]
            out.append(f"{names[c]} & {mode} & " + " & ".join(cells) + r" \\")
    if pooled:
        out.append(r"\midrule")
        for mode in ("none", "full", "split"):
            cells = [cell(pooled.get((judge, mode, view, "mqm")))
                     for judge in ("mindjudge", "deepseek_v4") for view in ("as_sent", "reference")]
            out.append(f"Mean of {len(present)} & {mode} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


#: Counselors left out of the paper's ablation table: GLM-5.3 on FP8 weights, a replicate of glm53_nvfp4.
EXCLUDE = ("glm53",)


def fmt_ci(r, nd=2, key="diff"):
    return f"{float(r[key]):+.{nd}f} [{float(r['lo']):+.{nd}f}, {float(r['hi']):+.{nd}f}]"


def drivers(d: Path) -> str:
    """The change from session 1, mean over counselors, and its parts by issue group, per judge."""
    rows = [r for r in csv.DictReader((d / "drivers.csv").open()) if r["counselor"] == "mean of counselors"]
    by = {(r["judge"], r["field"]): r for r in rows}
    groups = ["memory", "exploration", "advice and validation", "style and scope"]
    out = [r"\begin{tabular}{l" + "c" * (len(groups) + 1) + "}", r"\toprule",
           "Judge & MQM change & " + " & ".join(g.capitalize() for g in groups) + r" \\", r"\midrule"]
    for j in ("GPT-6.1 Sol", "Hy4-preview", "DeepSeek-V4.1", "MindJudge-27B"):
        if (j, "_mqm") not in by:
            continue
        out.append(f"{j} & {fmt_ci(by[j, '_mqm'])} & " + " & ".join(fmt_ci(by[j, f'group:{g}']) for g in groups)
                   + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def dose(d: Path) -> str:
    """Memory failures against score, per judge: session slope per member repeat, and the reply that caused it."""
    rows = list(csv.DictReader((d / "dose_response.csv").open()))
    get = lambda j, level, m: next((r for r in rows if r["judge"] == j and r["level"] == level  # noqa: E731
                                    and r["measure"].startswith(m)), None)
    out = [r"\begin{tabular}{lccc}", r"\toprule",
           r"Judge & MQM per member repeat & MQM per 10 pts notes missed & Reply forcing a repeat vs other \\",
           r"\midrule"]
    for j in ("GPT-6.1 Sol", "Hy4-preview", "DeepSeek-V4.1", "MindJudge-27B"):
        a, b, c = get(j, "session", "repeats"), get(j, "session", "missed_share"), get(j, "reply", "repeat-forcing")
        if a:
            out.append(f"{j} & {fmt_ci(a, 3, 'slope')} & {fmt_ci(b, 3, 'slope')} & {fmt_ci(c, 2, 'slope')} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


PART = "memory's part"


def memory_share(d: Path) -> str:
    """Change from session 1 with and without the counselor's memory, mean over counselors."""
    rows = [r for r in csv.DictReader((d / "memory_share.csv").open()) if r["counselor"] == "mean of counselors"]
    out = [r"\begin{tabular}{llccc}", r"\toprule",
           r"Judge & View & Change with memory & Change without & Memory's part \\", r"\midrule"]
    for j in ("DeepSeek-V4.1", "MindJudge-27B"):
        for view in ("reference", "as_sent"):
            by = {r["measure"]: r for r in rows if r["judge"] == j and r["view"] == view}
            if by:
                out.append(f"{j} & {view.replace('_', ' ')} & {fmt_ci(by['with memory'])} & "
                           f"{fmt_ci(by['without memory'])} & {fmt_ci(by[PART])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def overlap(d: Path) -> str:
    rows = list(csv.DictReader((d / "overlap.csv").open()))
    out = [r"\begin{tabular}{lccc}", r"\toprule",
           r"Judge & Flags a memory issue & Major or Critical & On other replies: Major or Critical \\", r"\midrule"]
    for r in rows:
        out.append(f"{r['judge']} & {100 * float(r['memory_flag']):.0f}\\% & {100 * float(r['major_plus']):.0f}\\% & "
                   f"{100 * float(r['major_plus_rest']):.0f}\\% \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def trajectory(d: Path) -> str:
    """Step (session 1 to 2) and trend (slope over sessions 2-6, per session) by memory condition."""
    rows = list(csv.DictReader((d / "steps.csv").open()))
    get = lambda **kw: next((r for r in rows if all(r[k] == v for k, v in kw.items())), None)  # noqa: E731
    out = [r"\begin{tabular}{llcc}", r"\toprule", r"Memory & Measure & Step, s1 $\to$ s2 & Trend per session, s2--s6 \\",
           r"\midrule"]
    names = {"compacted": "Benchmark log", "none": "None", "full": "Full transcripts", "split": "Split notes"}
    for mode in ("compacted", "none", "full", "split"):
        a = get(set="ablation", judge="DeepSeek-V4.1", view="reference", mode=mode, measure="MQM", kind="step")
        b = get(set="ablation", judge="DeepSeek-V4.1", view="reference", mode=mode, measure="MQM", kind="trend")
        c = get(set="ablation", judge="labelers", mode=mode, measure="member repeats", kind="trend")
        if a and b:
            out.append(f"{names[mode]} & MQM (DeepSeek, reference) & {fmt_ci(a)} & {fmt_ci(b, 3)} \\\\")
        if c:
            out.append(f" & Member repeats & -- & {fmt_ci(c, 3)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c1", required=True, type=Path)
    ap.add_argument("--c1_shared", required=True, type=Path)
    ap.add_argument("--c2", required=True, type=Path)
    ap.add_argument("--c3", type=Path)
    ap.add_argument("--ablation", type=Path)
    ap.add_argument("--later", type=Path, help="later_sessions dir: drivers/, dose_response/, memory_share/, "
                    "judge_overlap/ for the tables on what drives the change")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.later:
        for name, f, sub in (("tab_drivers", drivers, "drivers"), ("tab_dose", dose, "dose_response"),
                             ("tab_memory_share", memory_share, "memory_share"), ("tab_overlap", overlap, "judge_overlap"),
                             ("tab_trajectory", trajectory, "trajectory")):
            if (args.later / sub).is_dir():
                (args.out / f"{name}.tex").write_text(f(args.later / sub))
    (args.out / "tab_later_ranking.tex").write_text(ranking(args.c1, args.c1_shared))
    (args.out / "tab_later_errors.tex").write_text(errors(args.c2))
    if args.c3:
        (args.out / "tab_c3_judges.tex").write_text(c3_judges(args.c3))
    if args.ablation:
        (args.out / "tab_ablation.tex").write_text(ablation(args.ablation))
    for name, d in (("c1", args.c1), ("c1_shared", args.c1_shared), ("c2", args.c2), ("c3", args.c3),
                    ("ablation", args.ablation)):
        if d and (d / "manifest.json").is_file():
            shutil.copyfile(d / "manifest.json", args.out / f"manifest_{name}.json")
    print(json.dumps(sorted(p.name for p in args.out.iterdir())))


if __name__ == "__main__":
    main()
