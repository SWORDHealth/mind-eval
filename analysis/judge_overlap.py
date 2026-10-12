"""When GPT-6.1 Sol flags a memory issue on a reply, what do the other judges say about the same reply?

    python analysis/judge_overlap.py --runs_root <dir> --out <dir> [--exclude <leaks.jsonl>]

Paper runs, sessions 2-3. For the replies Sol flags with missed_shared_context or false_memory, and for all other
replies: each judge's share flagging a memory issue, its share at Major or Critical, its mean MQM weight, and its most
frequent flags on those replies. Writes overlap.csv and report.md.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from common import EXCLUDE, ISSUES, WEIGHT, judge_dir, load_verdicts, mean, registry, runs_root, set_exclusions

MEMORY = ("missed_shared_context", "false_memory")
OTHERS = {"own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", type=Path)
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    acc = {j: {"sol_memory": [], "rest": []} for j in ["gpt61_sol", *OTHERS]}
    for r in runs:
        drop = EXCLUDE.get(r["key"], set())
        vs = {j: load_verdicts(judge_dir(root, r, j, judges) / "verdicts.jsonl") for j in acc
              if (judge_dir(root, r, j, judges) / "verdicts.jsonl").is_file()}
        if "gpt61_sol" not in vs:
            continue
        for k, sol in vs["gpt61_sol"].items():
            if k in drop or sol["episode"] == "ep000":
                continue
            group = "sol_memory" if any(sol.get(f) == 1 for f in MEMORY) else "rest"
            for j, v in vs.items():
                if k in v:
                    acc[j][group].append(v[k])
    rows, lines = [], ["# Replies GPT-6.1 Sol flags with a memory issue, as the other judges see them "
                       "(paper runs, sessions 2-3)", "",
                       "| judge | replies | memory issue flagged | Major or Critical | mean MQM weight | "
                       "on the other replies: memory flag / Major+ / weight | its most frequent flags on them |",
                       "|---|---|---|---|---|---|---|"]
    for j, d in acc.items():
        a, b = d["sol_memory"], d["rest"]
        if not a:
            continue

        def stats(xs):
            return (mean(any(v.get(f) == 1 for f in MEMORY) for v in xs),
                    mean(v["severity"] in ("Major", "Critical") for v in xs), mean(WEIGHT[v["severity"]] for v in xs))
        sa, sb = stats(a), stats(b)
        top = Counter(i for v in a for i in ISSUES if v.get(i) == 1).most_common(3)
        name = "GPT-6.1 Sol" if j == "gpt61_sol" else OTHERS[j]
        rows.append({"judge": name, "replies": len(a), "memory_flag": sa[0], "major_plus": sa[1], "weight": sa[2],
                     "memory_flag_rest": sb[0], "major_plus_rest": sb[1], "weight_rest": sb[2],
                     "top_flags": "; ".join(f"{i} {c / len(a):.0%}" for i, c in top)})
        lines.append(f"| {name} | {len(a)} | {sa[0]:.0%} | {sa[1]:.0%} | {sa[2]:.2f} | {sb[0]:.0%} / {sb[1]:.0%} / "
                     f"{sb[2]:.2f} | " + ", ".join(f"{i} {c / len(a):.0%}" for i, c in top) + " |")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "overlap.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
