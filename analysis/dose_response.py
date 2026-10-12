"""Do memory failures cost score? Paper runs, sessions 2-3: the labelers' continuity failures against each judge's
session MQM, within counselor, and at the reply where the failure happens.

    python analysis/dose_response.py --runs_root <dir> --labels <c3_labels dir> --out <dir> [--exclude <leaks.jsonl>]

Per session, from the frozen labelers: missed and contradicted notes and re-asks (continuity_v1), member repeats and
corrections (member_side_v1). Reported per judge:
  session   the slope of session MQM on the session's count of continuity failures, within counselor (both demeaned by
            counselor), and session MQM by failure count (0, 1, 2, 3+), each re-centred on the grand mean; the same for
            member repeats. 95% intervals resample members (every counselor's sessions of a drawn member together).
  reply     the mean MQM weight of the replies that fail (a missed or contradicted note, a re-ask, or the reply a member
            answers by repeating a recorded fact) against every other reply of sessions 2-3, within counselor.
Writes dose_response.csv and report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from common import (
    BOOT,
    EXCLUDE,
    WEIGHT,
    ci,
    judge_dir,
    load_verdicts,
    mean,
    member_draws,
    registry,
    runs_root,
    set_exclusions,
)

JUDGES = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview"}
LATER = ("ep001", "ep002")


def load_labels(d: Path) -> dict[tuple[str, str, str], dict]:
    keys = {k["id"]: k for k in map(json.loads, (d / "keys.jsonl").read_text().splitlines())}
    out = {}
    for line in (d / "labels.jsonl").read_text().splitlines():
        x = json.loads(line)
        if "parsed" in x and x["id"] in keys:
            k = keys[x["id"]]
            out[k["name"], k["member"], k["episode"]] = x["parsed"]
    return out


def failures(cont: dict | None, side: dict | None) -> tuple[dict, dict[str, set[int]]]:
    """Session measures, and the failing reply turns of each kind: the reply a member answers by repeating a recorded
    fact (member_side_v1, validated), and the reply where a relevant note is missed (continuity_v1, validated)."""
    ops = (cont or {}).get("opportunities") or []
    missed = [o for o in ops if o.get("status") == "missed"]
    repeats = (side or {}).get("repeats") or []
    c = {"repeats": len(repeats), "missed_share": (len(missed) / len(ops)) if ops else None}
    turns = {"repeat-forcing": {x["turn"] for x in repeats if isinstance(x.get("turn"), int)},
             "missed-note": {o["turn"] for o in missed if isinstance(o.get("turn"), int)}}
    return c, turns


def slope(points: list[tuple[str, str, float, float]], draws: list[list[str]]) -> tuple[float, float, float]:
    """Within-counselor OLS slope of y on x: points are (counselor, member, x, y); demeaned by counselor."""
    def fit(pts):
        by_c = defaultdict(list)
        for c, _, x, y in pts:
            by_c[c].append((x, y))
        num = den = 0.0
        for xs in by_c.values():
            mx, my = mean(x for x, _ in xs), mean(y for _, y in xs)
            num += sum((x - mx) * (y - my) for x, y in xs)
            den += sum((x - mx) ** 2 for x, _ in xs)
        return num / den if den else float("nan")
    by_m = defaultdict(list)
    for p in points:
        by_m[p[1]].append(p)
    est = fit(points)
    lo, hi = ci([fit([p for m in d for p in by_m[m]]) for d in draws])
    return est, lo, hi


def binned(points, draws, bins=(0, 1, 2, 3)) -> list[dict]:
    """Session MQM by count bin (the last bin is 'or more'), re-centred: y minus its counselor mean plus the grand mean."""
    by_c = defaultdict(list)
    for c, _, x, y in points:
        by_c[c].append(y)
    cm = {c: mean(v) for c, v in by_c.items()}
    grand = mean(y for *_, y in points)
    adj = [(m, min(x, bins[-1]), y - cm[c] + grand) for c, m, x, y in points]
    by_m = defaultdict(list)
    for a in adj:
        by_m[a[0]].append(a)
    out = []
    for b in bins:
        ys = [y for _, x, y in adj if x == b]
        if not ys:
            continue
        boots = []
        for d in draws:
            v = [y for m in d for _, x, y in by_m[m] if x == b]
            if v:
                boots.append(mean(v))
        lo, hi = ci(boots)
        out.append({"bin": f"{b}+" if b == bins[-1] else str(b), "sessions": len(ys), "mqm": mean(ys), "lo": lo, "hi": hi})
    return out


def analyse(label: str, sources: list, cont: dict, side: dict, members: list[str], draws: list, rows: list,
            lines: list, unit_of: str = "counselor", units: str = "counselors") -> None:
    """One judge (or judge view): sources are (label key, verdicts path, replies to drop, episodes) per run."""
    sess_pts = {"repeats": [], "missed_share": []}
    reply = {kind: defaultdict(lambda: {"bad": [], "other": []}) for kind in ("repeat-forcing", "missed-note")}
    for key, path, drop, episodes in sources:
        vs = defaultdict(dict)
        for k, v in load_verdicts(path).items():
            if k not in drop and v["episode"] in episodes:
                vs[v["member"], v["episode"]][v["turn"]] = WEIGHT[v["severity"]]
        for (m, e), turns in vs.items():
            c_lab, s_lab = cont.get((key, m, e)), side.get((key, m, e))
            if c_lab is None or s_lab is None:
                continue
            counts, bad = failures(c_lab, s_lab)
            y = mean(turns.values())
            sess_pts["repeats"].append((key, m, counts["repeats"], y))
            if counts["missed_share"] is not None:
                sess_pts["missed_share"].append((key, m, 10 * counts["missed_share"], y))
            for kind, kturns in bad.items():
                for t, w in turns.items():
                    reply[kind][key]["bad" if t in kturns else "other"].append((m, w))
    if not sess_pts["repeats"]:
        return
    lines += [f"## {label}", ""]
    for measure, pts in sess_pts.items():
        s, lo, hi = slope(pts, draws)
        rows.append({"judge": label, "level": "session", "measure": measure, "slope": s, "lo": lo, "hi": hi,
                     "sessions": len(pts)})
        unit = "member repeat" if measure == "repeats" else "10 points of relevant notes missed"
        lines.append(f"- Session MQM per {unit} (within {unit_of}): {s:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                     f"over {len(pts)} sessions")
        if measure == "repeats":
            bs = binned(pts, draws)
            for b in bs:
                rows.append({"judge": label, "level": "session_bin", "measure": measure, **b})
            lines.append("  by member repeats: " + ", ".join(
                f"{b['bin']}: {b['mqm']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}] (n {b['sessions']})" for b in bs))
    # Reply level: failing replies against the rest, within counselor, then averaged over counselors.
    for kind, rep in reply.items():
        by_m = defaultdict(lambda: defaultdict(lambda: {"bad": [], "other": []}))
        for c, d in rep.items():
            for k2 in ("bad", "other"):
                for m, w in d[k2]:
                    by_m[m][c][k2].append(w)

        def stat(ms, rep=rep, by_m=by_m):
            per_c = []
            for c in rep:
                bad = [w for m in ms for w in by_m[m][c]["bad"]]
                oth = [w for m in ms for w in by_m[m][c]["other"]]
                if bad and oth:
                    per_c.append(mean(bad) - mean(oth))
            return mean(per_c) if per_c else float("nan")
        est = stat(members)
        lo, hi = ci([stat(d) for d in draws[:500]])
        nb = sum(len(d["bad"]) for d in rep.values())
        above = sum(1 for d in rep.values() if d["bad"] and d["other"]
                    and mean(w for _, w in d["bad"]) > mean(w for _, w in d["other"]))
        rows.append({"judge": label, "level": "reply", "measure": f"{kind} minus other replies", "slope": est,
                     "lo": lo, "hi": hi, "sessions": nb})
        lines.append(f"- Reply MQM weight, {kind} replies minus the rest: {est:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                     f"({nb} replies; {above} of {len(rep)} {units} above zero)")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--labels", required=True, type=Path, help="dir holding continuity_v1/ and member_side_v1/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--ablation", type=Path, help="memory_ablation dir: the same analysis over its sessions 2-6")
    ap.add_argument("--ablation_labels", type=Path, help="later_sessions/ablation_labels")
    ap.add_argument("--ablation_exclude", default="glm53")
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    cont, side = load_labels(args.labels / "continuity_v1"), load_labels(args.labels / "member_side_v1")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], ["# Do memory failures cost score? (paper runs, sessions 2-3, within counselor)", ""]
    members = sorted({m for (_, m, _) in cont})
    draws = member_draws(members, BOOT)
    for judge, label in JUDGES.items():
        sources = [(r["key"], judge_dir(root, r, judge, judges) / "verdicts.jsonl", EXCLUDE.get(r["key"], set()), LATER)
                   for r in runs if (judge_dir(root, r, judge, judges) / "verdicts.jsonl").is_file()]
        analyse(label, sources, cont, side, members, draws, rows, lines)
    if args.ablation:
        # The ablation's sessions 2-6, every counselor x memory condition a unit of its own (labels keyed "c/mode").
        acont, aside = {}, {}
        for d in sorted((args.ablation_labels / "continuity_v1").glob("*/*/")):
            acont.update(load_labels(d))
        for d in sorted((args.ablation_labels / "member_side_v1").glob("*/*/")):
            aside.update(load_labels(d))
        eps = {f"ep{i:03d}" for i in range(1, 6)}
        lines += ["# The memory ablation (sessions 2-6, within counselor x memory condition)", ""]
        for judge, jl in (("mindjudge", "MindJudge-27B"), ("deepseek_v4", "DeepSeek-V4.1")):
            for view in ("reference", "as_sent"):
                sources = []
                for cdir in sorted((args.ablation / "judging" / judge / "main6").iterdir()):
                    if cdir.name in args.ablation_exclude.split(","):
                        continue
                    for mode in ("compacted", "none", "full", "split"):
                        v = "as_sent" if mode == "compacted" else view
                        path = cdir / mode / v / "judge" / "mindeval2" / "verdicts.jsonl"
                        if path.is_file():
                            sources.append((f"{cdir.name}/{mode}", path, set(), eps))
                analyse(f"{jl}, {view.replace('_', ' ')}", sources, acont, aside,
                        sorted({m for (_, m, _) in acont}), draws, rows, lines,
                        unit_of="counselor x condition", units="counselor x conditions")
    with (args.out / "dose_response.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
