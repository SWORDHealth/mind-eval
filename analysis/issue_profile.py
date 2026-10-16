"""Issue profile of any MindEval2 runs judged with the same judge, side by side.

    python analysis/issue_profile.py --run "<label>=<run dir>" [--run ...] --out <dir> [--judge_subdir judge/mindeval2]

A run is a directory holding run.json, members/ and <judge_subdir>/verdicts.jsonl (a judging view works too: pass the
directory whose judge/mindeval2/ holds the verdicts). Per run: the share of replies the judge flags with each issue,
the severity mix, session MQM by session (mean over members, 95% member-bootstrap interval) and MQM by issue group
with every weighted reply counted once (drivers.exclusive_parts). Every run is read on the members it shares with the
others. Needs matplotlib for the figure (any Python that has it).
Writes issue_profile.csv, report.md, issue_profile.png/.pdf.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BOOT, ISSUES, WEIGHT, ci, load_verdicts, mean, member_draws  # noqa: E402
from drivers import EXCLUSIVE, exclusive_parts  # noqa: E402

LEVELS = ("No concern", "Minor", "Major", "Critical")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="label=run dir (repeat)")
    ap.add_argument("--judge_subdir", default="judge/mindeval2")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        label, d = spec.split("=", 1)
        d = Path(d)
        vs = load_verdicts(d / args.judge_subdir / "verdicts.jsonl")
        jj = json.loads((d / args.judge_subdir / "judge.json").read_text())
        rj = json.loads((d / "run.json").read_text())
        runs[label] = {"v": vs, "judge": jj, "run": rj}
    members = sorted(set.intersection(*({v["member"] for v in r["v"].values()} for r in runs.values())))
    eps = sorted({v["episode"] for r in runs.values() for v in r["v"].values()})
    args.out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], [f"# Issue profile ({len(members)} shared members, sessions {len(eps)})", "",
                       "| run | judge | prompt | harness | counselor prompt | replies |", "|---|---|---|---|---|---|"]
    prof = {}
    draws = member_draws(members, BOOT)
    for label, r in runs.items():
        vs = [v for v in r["v"].values() if v["member"] in members]
        lines.append(f"| {label} | {r['judge']['model']} | {r['judge']['prompt_sha256'][:8]} | {r['run']['fingerprint'][:8]} "
                     f"| {r['run']['counselor'].get('system_prompt_sha256', '')[:8]} | {len(vs)} |")
        n = len(vs)
        issue = {i: sum(v.get(i) == 1 for v in vs) / n for i in ISSUES}
        sev = Counter(v["severity"] for v in vs)
        sev = {lv: sev.get(lv, 0) / n for lv in LEVELS}
        # session MQM by session, per member
        per = {}
        for v in vs:
            per.setdefault((v["member"], v["episode"]), []).append(WEIGHT[v["severity"]])
        by_ep = {}
        for e in eps:
            sm = {m: mean(per[m, e]) for m in members if (m, e) in per}
            lo, hi = ci([mean(sm[m] for m in d if m in sm) for d in draws])
            by_ep[e] = (mean(sm.values()), lo, hi)
        ex = exclusive_parts({k: v for k, v in r["v"].items() if v["member"] in members})
        groups = {cat: mean(d[cat] for d in ex.values()) for cat in ["_mqm", *EXCLUSIVE]}
        prof[label] = {"issue": issue, "severity": sev, "by_session": by_ep, "groups": groups}
        for i, x in issue.items():
            rows.append({"run": label, "measure": f"flag rate {i}", "value": x})
        for lv, x in sev.items():
            rows.append({"run": label, "measure": f"severity {lv}", "value": x})
        for e, (x, lo, hi) in by_ep.items():
            rows.append({"run": label, "measure": f"MQM {e}", "value": x, "lo": lo, "hi": hi})
        for g, x in groups.items():
            rows.append({"run": label, "measure": f"group {g}", "value": x})
    labels = list(runs)
    lines += ["", "## Share of replies flagged with each issue", "", "| issue | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for i in sorted(ISSUES, key=lambda i: -max(prof[lab]["issue"][i] for lab in labels)):
        lines.append(f"| {i.replace('_', ' ')} | " + " | ".join(f"{100 * prof[lab]['issue'][i]:.1f}%" for lab in labels) + " |")
    lines += ["", "## Severity mix", "", "| severity | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for lv in LEVELS:
        lines.append(f"| {lv} | " + " | ".join(f"{100 * prof[lab]['severity'][lv]:.1f}%" for lab in labels) + " |")
    lines += ["", "## Session MQM by session [95% CI]", "", "| session | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for e in eps:
        lines.append(f"| {int(e[2:]) + 1} | " + " | ".join("{:.2f} [{:.2f}, {:.2f}]".format(*prof[lab]["by_session"][e])
                                                         for lab in labels) + " |")
    lines += ["", "## MQM by issue group, each weighted reply counted once (mean session)", "",
              "| part | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for g in ["_mqm", *EXCLUSIVE]:
        lines.append(f"| {'MQM' if g == '_mqm' else g} | " + " | ".join(f"{prof[lab]['groups'][g]:.2f}" for lab in labels) + " |")
    with (args.out / "issue_profile.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "measure", "value", "lo", "hi"])
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(prof, labels, eps, args.out)


def plot(prof: dict, labels: list[str], eps: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#e34948", "#2a78d6", "#1baf7a", "#eda100", "#9a9893", "#4a3aa7"][: len(labels)]
    muted, grid = "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": muted,
                         "xtick.color": muted, "ytick.color": muted, "legend.frameon": False, "savefig.dpi": 220,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.25, 1], hspace=0.75, wspace=0.35)
    a = fig.add_subplot(gs[:, 0])
    issues = [i for i in sorted(ISSUES, key=lambda i: max(prof[lab]["issue"][i] for lab in labels))
              if max(prof[lab]["issue"][i] for lab in labels) >= 0.005]
    h = 0.8 / len(labels)
    for k, (lab, col) in enumerate(zip(labels, colors)):
        a.barh([y + ((len(labels) - 1) / 2 - k) * h for y in range(len(issues))],
               [100 * prof[lab]["issue"][i] for i in issues], height=h * 0.92, color=col, label=lab)
    a.set_yticks(range(len(issues)), [i.replace("_", " ") for i in issues])
    a.set_xlabel("% of replies the judge flags with the issue")
    a.grid(axis="x", color=grid, lw=0.5)
    a.legend(fontsize=6, loc="lower right")
    a.set_title("(a) Issue flag rates", loc="left", fontsize=7.5)
    b = fig.add_subplot(gs[0, 1])
    sev_col = {"No concern": "#d9d8d4", "Minor": "#eda100", "Major": "#eb6834", "Critical": "#b0272d"}
    for y, lab in enumerate(labels):
        left = 0.0
        for lv in LEVELS:
            x = 100 * prof[lab]["severity"][lv]
            b.barh(y, x, left=left, color=sev_col[lv], height=0.6, edgecolor="white", linewidth=0.6,
                   label=lv if y == 0 else None)
            left += x
    b.set_yticks(range(len(labels)), [lab.replace(" (", "\n(") for lab in labels], fontsize=5.8)
    b.set_ylim(len(labels) - 0.5, -0.5)
    b.set_xlabel("% of replies")
    b.legend(fontsize=5.6, ncol=4, loc="upper center", bbox_to_anchor=(0.4, -0.3))
    b.set_title("(b) Severity mix", loc="left", fontsize=7.5)
    c = fig.add_subplot(gs[1, 1])
    for k, (lab, col) in enumerate(zip(labels, colors)):
        xs = [int(e[2:]) + 1 + (k - (len(labels) - 1) / 2) * 0.06 for e in eps]
        ys = [prof[lab]["by_session"][e] for e in eps]
        c.plot(xs, [y for y, _, _ in ys], "-o", color=col, lw=1.4, ms=3.2, mec="white", mew=0.4)
        c.vlines(xs, [lo for _, lo, _ in ys], [hi for *_, hi in ys], color=col, lw=0.9)
    c.set_xticks(range(1, len(eps) + 1))
    c.set_xlabel("session")
    c.set_ylabel("session MQM (judge)")
    c.grid(axis="y", color=grid, lw=0.5)
    c.set_title("(c) MQM by session, 95% CI", loc="left", fontsize=7.5)
    fig.savefig(out / "issue_profile.png")
    fig.savefig(out / "issue_profile.pdf")
    plt.close(fig)
    print("issue_profile: png, pdf")


if __name__ == "__main__":
    main()
