"""Two checks on the change from session 1 to sessions 2-3 (paper runs, every judge, leaked replies left out):

    python analysis/schedule_checks.py --runs_root <dir> --out <dir> [--exclude <leaks.jsonl>]

severity   the change split by severity level: MQM = 1 x Minor share + 5 x Major share + 10 x Critical share, so the
           change is the sum of each level's weighted share change; per judge, mean over counselors.
schedule   every counselor of one harness version meets each member with the same drawn schedule (concealment and
           affect per session, the beats between sessions, the gaps), so the schedule can move every counselor's
           change for a member alike but cannot make counselors differ. Per judge and harness version: the share of
           the variance of change (counselor x member) that is the member's shared part, the counselor's part and the
           rest; and, for each schedule draw, how much of the members' shared change it accounts for (eta squared,
           with a permutation p over members).
Writes severity.csv, schedule.csv, report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from common import EPISODES, SEVERITIES, WEIGHT, load_judge, mean, registry, runs_root, set_exclusions

JUDGES = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview"}


def shape(run_dir: Path, member: str) -> dict | None:
    p = run_dir / "members" / member / "arc" / "arc.json"
    return json.loads(p.read_text()).get("shape") if p.is_file() else None


def draws(sh: dict) -> dict[str, str]:
    """The schedule draws as categories: session-1 and later concealment and affect, the two beats, the gaps."""
    later_conc = sorted(sh["concealment"][1:3])
    gaps = sh.get("gaps_hours") or []
    return {"concealment, session 1": sh["concealment"][0], "concealment, sessions 2-3": "/".join(later_conc),
            "affect, session 1": sh["affect"][0], "affect, session 2": sh["affect"][1],
            "beat before session 2": sh["beats"][0], "beat before session 3": sh["beats"][1],
            "gap before session 2": "under 1 day" if gaps and gaps[0] < 24 else "1 day or more"}


def eta2(groups: dict[str, list[float]]) -> float:
    xs = [x for g in groups.values() for x in g]
    m = mean(xs)
    tot = sum((x - m) ** 2 for x in xs)
    between = sum(len(g) * (mean(g) - m) ** 2 for g in groups.values() if g)
    return between / tot if tot else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    args.out.mkdir(parents=True, exist_ok=True)
    sev_rows, sch_rows = [], []
    lines = ["# Checks on the change from session 1 (paper runs, sessions 2-3 against session 1)", "",
             "## By severity: each level's weighted share change (mean over counselors)", "",
             "| judge | MQM change | " + " | ".join(SEVERITIES[1:]) + " |", "|---|---|" + "---|" * 3]
    loaded = {j: load_judge(root, runs, j, judges, check_coverage=False)["sessions"] for j in JUDGES}
    for j, sess in loaded.items():
        per_c = defaultdict(list)
        for r in runs:
            s = sess.get(r["key"], {})
            for m in r["run"]["member_ids"]:
                if not all((m, e) in s for e in EPISODES):
                    continue
                n = lambda e: s[m, e]["n"]  # noqa: E731
                share = {lvl: {e: s[m, e]["severity"].get(lvl, 0) / n(e) for e in EPISODES} for lvl in SEVERITIES}
                per_c[r["key"]].append({lvl: WEIGHT[lvl] * (mean(share[lvl][e] for e in EPISODES[1:]) - share[lvl]["ep000"])
                                        for lvl in SEVERITIES[1:]})
        parts = {lvl: mean(mean(x[lvl] for x in xs) for xs in per_c.values()) for lvl in SEVERITIES[1:]}
        total = sum(parts.values())
        lines.append(f"| {JUDGES[j]} | {total:+.3f} | " + " | ".join(f"{parts[lvl]:+.3f}" for lvl in SEVERITIES[1:]) + " |")
        sev_rows.append({"judge": JUDGES[j], "change": total, **{lvl: parts[lvl] for lvl in SEVERITIES[1:]}})
    lines += ["", "## The schedule", ""]
    rng = random.Random(0)
    for j, sess in loaded.items():
        for fp in sorted({r["fingerprint"] for r in runs}):
            group = [r for r in runs if r["fingerprint"] == fp]
            change = {}
            for r in group:
                s = sess.get(r["key"], {})
                for m in r["run"]["member_ids"]:
                    if all((m, e) in s for e in EPISODES):
                        change[r["key"], m] = mean(s[m, e]["score"] for e in EPISODES[1:]) - s[m, "ep000"]["score"]
            members = sorted({m for (_, m) in change if all((r["key"], m) in change for r in group)})
            cs = [r["key"] for r in group]
            if len(members) < 10 or len(cs) < 2:
                continue
            grand = mean(change[c, m] for c in cs for m in members)
            m_eff = {m: mean(change[c, m] for c in cs) - grand for m in members}
            c_eff = {c: mean(change[c, m] for m in members) - grand for c in cs}
            tot = sum((change[c, m] - grand) ** 2 for c in cs for m in members)
            ss_m = len(cs) * sum(v ** 2 for v in m_eff.values())
            ss_c = len(members) * sum(v ** 2 for v in c_eff.values())
            row = {"judge": JUDGES[j], "fingerprint": fp[:8], "counselors": len(cs), "members": len(members),
                   "member_share": ss_m / tot, "counselor_share": ss_c / tot, "rest_share": 1 - (ss_m + ss_c) / tot}
            lines.append(f"- {JUDGES[j]}, harness {fp[:8]} ({len(cs)} counselors x {len(members)} members): variance "
                         f"of change: members' shared part {100 * row['member_share']:.0f}%, counselors' part "
                         f"{100 * row['counselor_share']:.0f}%, counselor x member and noise "
                         f"{100 * row['rest_share']:.0f}%")
            sh = {m: draws(shape(group[0]["dir"], m)) for m in members if shape(group[0]["dir"], m)}
            shared = {m: m_eff[m] for m in sh}
            feats = []
            for f in next(iter(sh.values())):
                groups = defaultdict(list)
                for m, v in shared.items():
                    groups[sh[m][f]].append(v)
                if len(groups) < 2:
                    continue
                e = eta2(groups)
                labels = [sh[m][f] for m in shared]
                vals = list(shared.values())
                null = []
                for _ in range(args.perms):
                    rng.shuffle(labels)
                    g = defaultdict(list)
                    for lab, v in zip(labels, vals):
                        g[lab].append(v)
                    null.append(eta2(g))
                p = (1 + sum(x >= e for x in null)) / (1 + len(null))
                feats.append((f, e, p, len(groups)))
                sch_rows.append({**row, "draw": f, "eta2": e, "p_perm": p, "levels": len(groups)})
            lines.append("  members' shared change by schedule draw (eta², permutation p): " + "; ".join(
                f"{f} {e:.2f} (p {p:.2f})" for f, e, p, _ in sorted(feats, key=lambda x: -x[1])))
    for name, rows in (("severity.csv", sev_rows), ("schedule.csv", sch_rows)):
        if rows:
            with (args.out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
