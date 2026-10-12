"""How much of the change from session 1 is the counselor's memory? Within the ablation's own runs, every member
has one session 1 (copied from the source) and sessions 2-6 under each memory condition, so per counselor:

    change with memory     = later sessions under compacted  - session 1
    change without memory  = later sessions under none       - session 1
    the memory's part      = the difference (the ablation contrast, none minus compacted, with the sign flipped)

    python analysis/memory_share.py --ablation <memory_ablation dir> --out <dir>

Judges: MindJudge-27B and DeepSeek-V4.1. Session 1 is scored from the compacted run's as-sent verdicts (the same
session in every condition); later sessions under none from both views. Means over members, 95% member bootstrap;
the mean over counselors resamples members within each. The FP8 GLM-5.3 replicate is left out.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import pool, session_scores  # noqa: E402
from common import BOOT, ci, load_verdicts, mean, member_draws  # noqa: E402

LABEL = {"mindjudge": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude_counselors", default="glm53")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], ["# The memory's part of the change from session 1 (ablation runs, sessions 2-6)", ""]
    for judge in ("mindjudge", "deepseek_v4"):
        for view in ("as_sent", "reference"):
            per = {"with memory": {}, "without memory": {}, "memory's part": {}}
            lines += [f"## {LABEL[judge]}, later sessions without memory judged {view.replace('_', ' ')}", "",
                      "| counselor | session 1 | change with memory | change without memory | memory's part |",
                      "|---|---|---|---|---|"]
            for cdir in sorted((args.ablation / "judging" / judge / "main6").iterdir()):
                if cdir.name in args.exclude_counselors.split(","):
                    continue
                vp = lambda mode, v: cdir / mode / v / "judge" / "mindeval2" / "verdicts.jsonl"  # noqa: E731
                if not (vp("compacted", "as_sent").is_file() and vp("none", view).is_file()):
                    continue
                comp = session_scores(load_verdicts(vp("compacted", "as_sent")))
                none = session_scores(load_verdicts(vp("none", view)))
                d = {"with memory": {}, "without memory": {}, "memory's part": {}}
                s1 = {}
                for m in sorted({m for (m, e) in comp}):
                    if (m, "ep000") not in comp:
                        continue
                    eps = sorted(e for (mm, e) in comp if mm == m and e != "ep000" and (m, e) in none)
                    if not eps:
                        continue
                    first = comp[m, "ep000"]["mqm"]
                    w = mean(comp[m, e]["mqm"] for e in eps) - first
                    wo = mean(none[m, e]["mqm"] for e in eps) - first
                    s1[m] = first
                    d["with memory"][m], d["without memory"][m], d["memory's part"][m] = w, wo, w - wo
                if not s1:
                    continue
                ms = sorted(s1)
                draws = member_draws(ms, BOOT)
                cells = []
                for k, vals in d.items():
                    lo, hi = ci([mean(vals[m] for m in dr) for dr in draws])
                    per[k][cdir.name] = vals
                    rows.append({"judge": LABEL[judge], "view": view, "counselor": cdir.name, "measure": k,
                                 "diff": mean(vals.values()), "lo": lo, "hi": hi, "members": len(ms)})
                    cells.append(f"{mean(vals.values()):+.2f} [{lo:+.2f}, {hi:+.2f}]")
                lines.append(f"| {cdir.name} | {mean(s1.values()):.2f} | " + " | ".join(cells) + " |")
            pooled = {k: pool(v) for k, v in per.items() if len(v) >= 2}
            if pooled:
                lines.append("| **mean of counselors** | | " + " | ".join(
                    f"**{x['diff']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]**" for x in pooled.values()) + " |")
                for k, x in pooled.items():
                    rows.append({"judge": LABEL[judge], "view": view, "counselor": "mean of counselors", "measure": k,
                                 **{kk: x[kk] for kk in ("diff", "lo", "hi", "n_counselors", "above", "below")}})
            lines.append("")
    with (args.out / "memory_share.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
