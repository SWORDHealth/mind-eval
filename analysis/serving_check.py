"""Do the servings of one judge score alike? The same replies judged on every endpoint the ablation's verdicts came from.

    python analysis/serving_check.py prepare --run <run dir> --members 6 --seed 20261010 --endpoints <url> ... --out <dir>
    python analysis/serving_check.py report --out <dir>

prepare writes <out>/<tag>/as_sent/ for each endpoint (run.json, and members/ holding only a fixed random draw of
members, all symlinks: the run is not touched) and <out>/endpoints.json; each is then judged as usual with
MINDEVAL_JUDGE_API_BASE set to its endpoint. report gives, per endpoint, the session MQM over sessions 2 onward, and per
pair of endpoints the paired difference (95% CI resampling members) and the share of replies given the same severity.
The ablation spreads views over endpoints, so a level difference between endpoints would enter its contrasts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import session_scores  # noqa: E402
from common import BOOT, ci, load_verdicts, mean, member_draws  # noqa: E402


def tag(url: str) -> str:
    return url.split("//")[1].split("/")[0].replace(":", "_")


def prepare(args) -> None:
    run = args.run.resolve()
    members = json.loads((run / "run.json").read_text())["member_ids"]
    picked = sorted(random.Random(args.seed).sample(members, args.members))
    tags = {}
    for url in args.endpoints:  # an endpoint listed twice is judged twice: the within-serving baseline (T = 1.0)
        t = tag(url)
        t = t if t not in tags else f"{t}_repeat"
        tags[t] = url
    for t, url in tags.items():
        d = args.out / t / "as_sent"
        (d / "members").mkdir(parents=True, exist_ok=True)
        for link, target in [(d / "run.json", run / "run.json")] + [(d / "members" / m, run / "members" / m) for m in picked]:
            if not link.is_symlink():
                link.symlink_to(target)
    (args.out / "endpoints.json").write_text(json.dumps({"run": str(run), "members": picked, "seed": args.seed,
                                                         "endpoints": tags}, indent=1) + "\n")
    print(f"{len(picked)} members of {run}; {len(args.endpoints)} endpoints under {args.out}")


def report(args) -> None:
    meta = json.loads((args.out / "endpoints.json").read_text())
    verdicts = {t: load_verdicts(args.out / t / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl")
                for t in meta["endpoints"] if (args.out / t / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl").is_file()}
    scores = {t: session_scores(v) for t, v in verdicts.items()}
    members = meta["members"]
    lines = [f"# Serving check: {len(members)} members of {meta['run']}, sessions 2 onward", "",
             "| endpoint | replies | session MQM |", "|---|---|---|"]
    for t, s in scores.items():
        xs = [x["mqm"] for (m, e), x in s.items() if e != "ep000"]
        lines.append(f"| {t} | {len(verdicts[t])} | {mean(xs):.3f} |")
    lines += ["", "| pair | MQM difference [95% CI] | same severity |", "|---|---|---|"]
    rows = []
    for a, b in combinations(sorted(scores), 2):
        per = {}
        for m in members:
            d = [scores[a][m, e]["mqm"] - scores[b][m, e]["mqm"] for (mm, e) in scores[a]
                 if mm == m and e != "ep000" and (m, e) in scores[b]]
            if d:
                per[m] = mean(d)
        draws = [mean(per[m] for m in dr) for dr in member_draws(sorted(per), BOOT)] if per else []
        lo, hi = ci(draws) if draws else (float("nan"), float("nan"))
        shared = set(verdicts[a]) & set(verdicts[b])
        same = mean(verdicts[a][k]["severity"] == verdicts[b][k]["severity"] for k in shared) if shared else float("nan")
        rows.append({"pair": f"{a} - {b}", "diff": mean(per.values()) if per else None, "lo": lo, "hi": hi,
                     "same_severity": same, "replies": len(shared)})
        lines.append(f"| {a} − {b} | {rows[-1]['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {100 * same:.0f}% of {len(shared)} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--members", type=int, default=6)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--endpoints", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else report(args)


if __name__ == "__main__":
    main()
