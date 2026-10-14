"""Weight sensitivity and rank uncertainty of the paper's scores (paper runs, sessions 1-3, every judge).

    python analysis/weights_ranks.py --runs_root <dir> --out <dir> [--exclude <leaks.jsonl>] [--boot 2000]

Score: per member, the mean over its three sessions of session MQM (the mean severity weight of the session's replies);
per counselor, the mean over members. Only members with all three sessions judged for every counselor are used (the
same patient IDs for every model), so every score, rank and interval is over one member set.
  weights   the score under 0/1/5/10 (No concern/Minor/Major/Critical; the current weights), 0/1/3/10 and 0/1/5/25 (an
            MQM scoring example, not a clinical calibration), each counselor's rank (1 = lowest MQM, best) under each,
            Kendall tau between the rankings, and the session-averaged Major and Critical rates (per member the mean over
            its sessions of the share of replies at that level; mean over members).
  ranks     a bootstrap of the member IDs, the same draw for every counselor, each member keeping its three sessions;
            ranks computed jointly in each draw: each counselor's median rank and 95% rank range, how often it holds
            each rank, how often each counselor is ordered above each other (pairs.csv), and for close comparisons
            (adjacent in the point ranking, or ordered either way in at least 10% of draws) the paired score difference
            with its 95% interval.
Writes weights.csv, ranks.csv, pairs.csv, report.md, manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import (
    EPISODES,
    ci,
    git_describe,
    kendall_tau_b,
    load_judge,
    mean,
    member_draws,
    registry,
    runs_root,
    set_exclusions,
)

JUDGES = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview"}
LEVELS = ("No concern", "Minor", "Major", "Critical")
WEIGHTINGS = {"0/1/5/10": (0, 1, 5, 10), "0/1/3/10": (0, 1, 3, 10), "0/1/5/25": (0, 1, 5, 25)}
CURRENT = "0/1/5/10"


def ranks_of(scores: dict[str, float]) -> dict[str, int]:
    """1 = lowest MQM (best); ties share the better rank."""
    order = sorted(scores.values())
    return {k: order.index(v) + 1 for k, v in scores.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--close", type=float, default=0.10, help="a pair is close when ordered either way this often")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    label = {r["key"]: r["label"] for r in runs}
    harness = {r["key"]: r["fingerprint"][:8] for r in runs}
    args.out.mkdir(parents=True, exist_ok=True)
    wrows, rrows, prows = [], [], []
    lines = ["# Weight sensitivity and rank uncertainty (paper runs, sessions 1-3)", ""]
    for j, jl in JUDGES.items():
        S = load_judge(root, runs, j, judges, check_coverage=False)["sessions"]
        cs = [r["key"] for r in runs if r["key"] in S]
        members = sorted(set.intersection(*({m for m in r["run"]["member_ids"] if all((m, e) in S[r["key"]]
                                                                                     for e in EPISODES)}
                                             for r in runs if r["key"] in S)))
        # per counselor, member: the member's score under each weighting, and its Major and Critical rates
        per = {}
        for c in cs:
            per[c] = {}
            for m in members:
                ss = [S[c][m, e] for e in EPISODES]
                d = {w: mean(sum(wt * s["severity"].get(lv, 0) for wt, lv in zip(ws, LEVELS)) / s["n"] for s in ss)
                     for w, ws in WEIGHTINGS.items()}
                d["Major"] = mean(s["severity"].get("Major", 0) / s["n"] for s in ss)
                d["Critical"] = mean(s["severity"].get("Critical", 0) / s["n"] for s in ss)
                per[c][m] = d
        lines += [f"## {jl} ({len(cs)} counselors, {len(members)} members with all three sessions for every one)", ""]

        # weights
        score = {w: {c: mean(per[c][m][w] for m in members) for c in cs} for w in WEIGHTINGS}
        rank = {w: ranks_of(score[w]) for w in WEIGHTINGS}
        rate = {lv: {c: mean(per[c][m][lv] for m in members) for c in cs} for lv in ("Major", "Critical")}
        lines += ["### Scores under three weightings (rank in brackets) and session-averaged Major and Critical rates",
                  "", "| counselor | harness | " + " | ".join(WEIGHTINGS) + " | Major | Critical |",
                  "|---|---|" + "---|" * (len(WEIGHTINGS) + 2)]
        for c in sorted(cs, key=lambda c: score[CURRENT][c]):
            lines.append(f"| {label[c]} | {harness[c]} | " + " | ".join(f"{score[w][c]:.2f} ({rank[w][c]})"
                                                                         for w in WEIGHTINGS)
                         + f" | {100 * rate['Major'][c]:.1f}% | {100 * rate['Critical'][c]:.2f}% |")
            wrows.append({"judge": jl, "counselor": label[c], "harness": harness[c], "members": len(members),
                          **{f"score {w}": score[w][c] for w in WEIGHTINGS}, **{f"rank {w}": rank[w][c] for w in WEIGHTINGS},
                          "major_rate": rate["Major"][c], "critical_rate": rate["Critical"][c]})
        lines.append("")
        for w in WEIGHTINGS:
            if w == CURRENT:
                continue
            tau = kendall_tau_b([score[CURRENT][c] for c in cs], [score[w][c] for c in cs])
            moved = sorted(((rank[w][c] - rank[CURRENT][c], c) for c in cs), key=lambda x: -abs(x[0]))
            mv = ", ".join(f"{label[c]} {d:+d}" for d, c in moved if d)
            lines.append(f"- {w} against {CURRENT}: Kendall tau {tau:+.3f}; rank changes: {mv or 'none'}")
        lines.append("")

        # ranks: joint bootstrap of member IDs
        draws = member_draws(members, args.boot, seed=0)
        for w in WEIGHTINGS:
            bs = []
            for d in draws:
                bs.append({c: mean(per[c][m][w] for m in d) for c in cs})
            bran = [ranks_of(b) for b in bs]
            if w == CURRENT:
                lines += [f"### Rank uncertainty ({w}; {args.boot} joint bootstrap draws of the {len(members)} members)",
                          "", "| counselor | score [95% CI] | rank | median rank | 95% rank range | holds its rank |",
                          "|---|---|---|---|---|---|"]
            for c in sorted(cs, key=lambda c: score[w][c]):
                rs = sorted(br[c] for br in bran)
                lo_s, hi_s = ci([b[c] for b in bs])
                lo_r, hi_r = rs[int(0.025 * len(rs))], rs[min(len(rs) - 1, int(0.975 * len(rs)))]
                hold = sum(1 for r in rs if r == rank[w][c]) / len(rs)
                freq = {k: sum(1 for r in rs if r == k) / len(rs) for k in range(1, len(cs) + 1)}
                rrows.append({"judge": jl, "weighting": w, "counselor": label[c], "score": score[w][c], "score_lo": lo_s,
                              "score_hi": hi_s, "rank": rank[w][c], "rank_median": rs[len(rs) // 2], "rank_lo": lo_r,
                              "rank_hi": hi_r, "holds_rank": hold,
                              "rank_freq": json.dumps({k: round(v, 4) for k, v in freq.items() if v})})
                if w == CURRENT:
                    lines.append(f"| {label[c]} | {score[w][c]:.2f} [{lo_s:.2f}, {hi_s:.2f}] | {rank[w][c]} | "
                                 f"{rs[len(rs) // 2]} | {lo_r}-{hi_r} | {100 * hold:.0f}% |")
            # pairs
            order = sorted(cs, key=lambda c: score[w][c])
            close = []
            for i, a in enumerate(order):
                for b in order[i + 1:]:
                    p = sum(1 for bb in bs if bb[a] < bb[b]) / len(bs)
                    diffs = [bb[a] - bb[b] for bb in bs]
                    lo, hi = ci(diffs)
                    adjacent = order.index(b) == i + 1
                    row = {"judge": jl, "weighting": w, "better": label[a], "worse": label[b],
                           "diff": score[w][a] - score[w][b], "lo": lo, "hi": hi, "p_order": p, "adjacent": adjacent}
                    prows.append(row)
                    if adjacent or p < 1 - args.close:
                        close.append(row)
            if w == CURRENT:
                lines += ["", f"### Close comparisons ({w}): adjacent in the ranking, or ordered either way in at least "
                          f"{100 * args.close:.0f}% of draws", "",
                          "| better | worse | score difference [95% CI] | ordered so in | adjacent |", "|---|---|---|---|---|"]
                for r in close:
                    lines.append(f"| {r['better']} | {r['worse']} | {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                                 f"{100 * r['p_order']:.0f}% | {'yes' if r['adjacent'] else ''} |")
                n_sep = sum(1 for r in prows if r["judge"] == jl and r["weighting"] == w and r["hi"] < 0)
                n_all = sum(1 for r in prows if r["judge"] == jl and r["weighting"] == w)
                lines += ["", f"- {n_sep} of {n_all} pairs have a paired difference whose 95% interval excludes 0.", ""]
        # rank ranges under the alternative weightings, summary
        for w in WEIGHTINGS:
            if w == CURRENT:
                continue
            rr = [r for r in rrows if r["judge"] == jl and r["weighting"] == w]
            lines.append(f"- {w}: median 95% rank range width {sorted(r['rank_hi'] - r['rank_lo'] for r in rr)[len(rr) // 2]}"
                         f" (current {sorted(r['rank_hi'] - r['rank_lo'] for r in rrows if r['judge'] == jl and r['weighting'] == CURRENT)[len(cs) // 2]})")
        lines.append("")
    for name, rows in (("weights.csv", wrows), ("ranks.csv", rrows), ("pairs.csv", prows)):
        with (args.out / name).open("w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
            w_.writeheader()
            w_.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"script": "analysis/weights_ranks.py", "analysis_git": git_describe(),
                                                        "boot": args.boot, "weightings": WEIGHTINGS,
                                                        "exclude": str(args.exclude)}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
