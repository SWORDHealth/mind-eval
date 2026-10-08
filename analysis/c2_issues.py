"""C2, errors that appear later: the judge's own issue categories and severities by session index and counselor.

A rate is the share of a session's counselor replies that carry the flag (or the severity), averaged over the
session's members, so every session counts once as in MQM. Two views:
  - pooled over counselors: each member's rate is averaged over counselors first, so the member stays the unit;
  - per counselor.
Changes from session 1 are within member, with 95% intervals from resampling members.

A category is called "later-only" when it is rare in session 1 (pooled rate below --rare, 2% of replies by default)
and its pooled rate in sessions 2-3 is at least --rise (1.5x) the session-1 rate and at least --min_rise (0.2
points) above it, with the change's interval excluding 0. Every category is reported, flagged or not.

    python analysis/c2_issues.py --runs_root <dir> --out <dir>/c2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import (
    BOOT,
    EPISODES,
    ISSUES,
    ci,
    judge_identity,
    load_judge,
    mean,
    member_draws,
    registry,
    runs_root,
    write_manifest,
)

SEV = {"major_plus": ("Major", "Critical"), "critical": ("Critical",), "minor_plus": ("Minor", "Major", "Critical")}
CATEGORIES = ISSUES + tuple(SEV)


def session_rate(s: dict, cat: str) -> float:
    if cat in SEV:
        return sum(s["severity"][x] for x in SEV[cat]) / s["n"]
    return s["flags"][cat] / s["n"]


def analyse(root: Path, runs: list[dict], judges: dict, judge: str, members: list[str], rare: float,
            rise: float, min_rise: float) -> dict:
    loaded = load_judge(root, runs, judge, judges)
    keys = [r["key"] for r in runs if r["key"] in loaded["sessions"]
            and all((m, ep) in loaded["sessions"][r["key"]] for m in members for ep in EPISODES)]
    dropped = [r["key"] for r in runs if r["key"] not in keys]
    # rate[c][cat][s][m]
    rate = {c: {cat: {f"s{i + 1}": {m: session_rate(loaded["sessions"][c][(m, ep)], cat) for m in members}
                      for i, ep in enumerate(EPISODES)} for cat in CATEGORIES} for c in keys}
    replies = {c: {f"s{i + 1}": sum(loaded["sessions"][c][(m, ep)]["n"] for m in members)
                   for i, ep in enumerate(EPISODES)} for c in keys}
    draws = member_draws(members)

    def lv(c, cat, w, ms):
        if w == "later":
            return mean((rate[c][cat]["s2"][m] + rate[c][cat]["s3"][m]) / 2 for m in ms)
        return mean(rate[c][cat][w][m] for m in ms)

    # Every counselor plays the same members, so the pooled mean is the mean over members of each member's mean
    # over counselors: computed once per member, then resampled.
    per_member = {cat: {w: {m: mean(rate[c][cat][w][m] for c in keys) for m in members} for w in ("s1", "s2", "s3")}
                  for cat in CATEGORIES}

    def pooled(cat, w, ms):
        if w == "later":
            return mean((per_member[cat]["s2"][m] + per_member[cat]["s3"][m]) / 2 for m in ms)
        return mean(per_member[cat][w][m] for m in ms)

    pooled_rows, per_rows = [], []
    for cat in CATEGORIES:
        boot = [{w: pooled(cat, w, ms) for w in ("s1", "s2", "s3", "later")} for ms in draws]
        pt = {w: pooled(cat, w, members) for w in ("s1", "s2", "s3", "later")}
        row = {"judge": judge, "category": cat, "counselors": len(keys), "members": len(members),
               **{f"rate_{w}": pt[w] for w in pt}}
        for w in ("s2", "s3", "later"):
            lo, hi = ci([b[w] - b["s1"] for b in boot])
            row.update({f"change_{w}": pt[w] - pt["s1"], f"change_{w}_lo": lo, f"change_{w}_hi": hi})
        row["ratio_later"] = pt["later"] / pt["s1"] if pt["s1"] else float("inf")
        row["later_only"] = (pt["s1"] < rare and row["ratio_later"] >= rise and row["change_later"] >= min_rise
                             and row["change_later_lo"] > 0)
        # Counselors whose own later rate rises with an interval excluding 0.
        rising = []
        for c in keys:
            cb = [lv(c, cat, "later", ms) - lv(c, cat, "s1", ms) for ms in draws]
            lo, hi = ci(cb)
            d = lv(c, cat, "later", members) - lv(c, cat, "s1", members)
            per = {"judge": judge, "category": cat, "counselor": c,
                   **{f"rate_{w}": lv(c, cat, w, members) for w in ("s1", "s2", "s3", "later")},
                   "change_later": d, "lo": lo, "hi": hi,
                   "replies_s1": replies[c]["s1"], "replies_s2": replies[c]["s2"], "replies_s3": replies[c]["s3"]}
            per_rows.append(per)
            if lo > 0:
                rising.append(c)
        row["counselors_rising"] = len(rising)
        row["counselors_falling"] = sum(1 for p in per_rows if p["category"] == cat and p["hi"] < 0)
        pooled_rows.append(row)
    return {"keys": keys, "pooled": pooled_rows, "per": per_rows, "problems": loaded["problems"],
            "dropped": dropped}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def report(results: dict, labels: dict, rare: float, rise: float) -> str:
    lines = ["# C2: issue categories by session", "",
             f"Rates are % of replies (session-weighted). Later-only = pooled s1 rate < {100 * rare:.0f}%, "
             f"later/s1 >= {rise}, a rise of at least 0.2 points, and the change's 95% CI > 0.", ""]
    for judge, res in results.items():
        lines += [f"## Judge: {judge} ({len(res['keys'])} counselors x 50 members)", ""]
        if res["problems"] or res["dropped"]:
            lines += [f"Coverage: {res['problems']}; dropped: {res['dropped']}", ""]
        lines += ["| category | s1 | s2 | s3 | later - s1 [95% CI] | later/s1 | counselors up / down | later-only |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(res["pooled"], key=lambda r: -r["change_later"]):
            lines.append(f"| {r['category']} | {pct(r['rate_s1'])} | {pct(r['rate_s2'])} | {pct(r['rate_s3'])} | "
                         f"{100 * r['change_later']:+.1f} [{100 * r['change_later_lo']:+.1f}, "
                         f"{100 * r['change_later_hi']:+.1f}] | {r['ratio_later']:.2f} | "
                         f"{r['counselors_rising']} / {r['counselors_falling']} | {'**yes**' if r['later_only'] else ''} |")
        # Per counselor: later - s1 for the categories that move most, marked where the CI excludes 0.
        top = [r["category"] for r in sorted(res["pooled"], key=lambda r: -abs(r["change_later"]))[:8]]
        per = {(p["counselor"], p["category"]): p for p in res["per"]}
        lines += ["", "Per counselor, later - s1 in points (* = 95% CI excludes 0):", "",
                  "| counselor | " + " | ".join(top) + " |", "|---|" + "---|" * len(top)]
        for c in res["keys"]:
            cells = []
            for cat in top:
                p = per[c, cat]
                star = "*" if p["lo"] > 0 or p["hi"] < 0 else ""
                cells.append(f"{100 * p['change_later']:+.1f}{star}")
            lines.append(f"| {labels[c]} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", default="gpt61_sol,own,deepseek_v4,hy4")
    ap.add_argument("--subset", choices=("all", "shared"), default="all")
    ap.add_argument("--rare", type=float, default=0.02)
    ap.add_argument("--rise", type=float, default=1.5)
    ap.add_argument("--min_rise", type=float, default=0.002)
    args = ap.parse_args()
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    if args.subset == "shared":
        runs = [r for r in runs if r["fingerprint"] == "7962e7caed157571"]
    members = runs[0]["run"]["member_ids"]
    labels = {r["key"]: r["label"] for r in runs}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for judge in args.judges.split(","):
        results[judge] = analyse(root, runs, judges, judge, members, args.rare, args.rise, args.min_rise)
        print(f"{judge}: {len(results[judge]['keys'])} counselors", flush=True)
    write_csv(out / "pooled.csv", [r for res in results.values() for r in res["pooled"]])
    write_csv(out / "per_counselor.csv", [r for res in results.values() for r in res["per"]])
    (out / "report.md").write_text(report(results, labels, args.rare, args.rise))
    write_manifest(out, "analysis/c2_issues.py", runs, {j: judge_identity(root, runs, j, judges) for j in results},
                   {"subset": args.subset, "bootstrap": BOOT, "rare": args.rare, "rise": args.rise,
                    "min_rise": args.min_rise})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
