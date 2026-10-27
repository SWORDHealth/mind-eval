"""What the simulated patient does, session 1 against later sessions and by the beat before the session (paper runs).

    python analysis/patient_moves.py --runs_root <dir> --out <dir>

Every member turn after the opening message: the committed move (final_move.move) and its class (move_class). Per
session index: each move's share of turns (members pooled over the runs), with a 95% interval that resamples members;
the Jensen-Shannon divergence between the session-1 and the later-session move distributions, against the divergence
between two random halves of the members within session 1 (its noise floor). Sessions 2 and 3 by the beat scheduled
before them (arc.json): the move-class mix, turns per session, the share the member ends and words per turn. Runs at
both harness versions are pooled (each run's own schedule); the move-class mix is also given per version.
Writes patient_moves.csv, report.md, patient_moves.png/.pdf (needs matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BOOT, ci, member_draws, registry, runs_root  # noqa: E402

CLASSES = ("engaging", "disclosing", "expressive", "concealing")


def jsd(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    js = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        mm = (a + b) / 2
        if a:
            js += 0.5 * a * math.log2(a / mm)
        if b:
            js += 0.5 * b * math.log2(b / mm)
    return js


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = runs_root(args.runs_root)
    runs, _ = registry(root)
    args.out.mkdir(parents=True, exist_ok=True)
    # per (member, session): move counts, class counts; per session record: dynamics
    moves = defaultdict(Counter)   # (member, s) -> Counter of moves, pooled over runs
    classes = defaultdict(Counter)
    by_harness = defaultdict(Counter)  # (fingerprint, s) -> class counts
    sess = []                       # dicts per (run, member, s)
    for r in runs:
        for m in r["run"]["member_ids"]:
            arc_dir = Path(r["dir"]) / "members" / m / "arc"
            if not (arc_dir / "arc.json").is_file():
                continue
            shape = json.loads((arc_dir / "arc.json").read_text())["shape"]
            for s in (1, 2, 3):
                t = arc_dir / f"ep{s - 1:03d}" / "trace.jsonl"
                if not t.is_file():
                    continue
                recs = [json.loads(x) for x in t.read_text().split("\n") if x.strip()]
                if not recs or recs[-1].get("type") != "session":
                    continue
                n = 0
                words = []
                cc = Counter()
                for x in recs:
                    if x.get("type") != "turn" or x.get("turn", 0) == 0:
                        continue
                    fm = x.get("final_move") or {}
                    if fm.get("move"):
                        moves[m, s][fm["move"].removeprefix("do_")] += 1
                    if x.get("move_class"):
                        classes[m, s][x["move_class"]] += 1
                        cc[x["move_class"]] += 1
                        by_harness[r["fingerprint"][:8], s][x["move_class"]] += 1
                    words.append(len((x.get("utterance") or "").split()))
                    n += 1
                sess.append({"run": r["key"], "member": m, "session": s, "beat": "none" if s == 1 else shape["beats"][s - 2],
                             "turns": n, "member_ended": recs[-1].get("termination") == "member_terminated",
                             "words": st.mean(words) if words else 0.0, **{c: cc[c] / n if n else 0.0 for c in CLASSES}})
    members = sorted({m for m, _ in moves})
    draws = member_draws(members, BOOT)
    rows, lines = [], ["# What the simulated patient does, by session and by beat (paper runs, 18 counselors)", ""]

    def pooled(sel, d=None):
        c = Counter()
        for m in (d if d is not None else members):
            for s in sel:
                c.update(moves.get((m, s), {}))
        return c

    s1, later = pooled([1]), pooled([2, 3])
    n1, n2 = sum(s1.values()), sum(later.values())
    top = [k for k, _ in (s1 + later).most_common(14)]
    lines += [f"## Moves: session 1 ({n1} turns) against sessions 2-3 ({n2} turns), share of turns [95% CI]", "",
              "| move | session 1 | sessions 2-3 | change (points) |", "|---|---|---|---|"]
    boots = [(pooled([1], d), pooled([2, 3], d)) for d in draws[:500]]
    for k in top:
        a, b = s1[k] / n1, later[k] / n2
        diffs = [bb[1][k] / sum(bb[1].values()) - bb[0][k] / sum(bb[0].values()) for bb in boots]
        lo, hi = ci(diffs)
        lines.append(f"| {k.replace('_', ' ')} | {100 * a:.1f}% | {100 * b:.1f}% | {100 * (b - a):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
        rows.append({"analysis": "move", "measure": k, "session 1": a, "sessions 2-3": b, "diff": b - a, "lo": lo, "hi": hi})
    js = jsd(s1, later)
    js_b = [jsd(b0, b1) for b0, b1 in boots]
    rng = random.Random(0)
    noise = []
    for _ in range(200):
        sh = members[:]
        rng.shuffle(sh)
        noise.append(jsd(pooled([1], sh[: len(sh) // 2]), pooled([1], sh[len(sh) // 2:])))
    lines += ["", f"- Jensen-Shannon divergence, session 1 against sessions 2-3: {js:.4f} [{ci(js_b)[0]:.4f}, "
              f"{ci(js_b)[1]:.4f}] bits; two halves of the members within session 1: {st.mean(noise):.4f} "
              f"(95th percentile {sorted(noise)[189]:.4f}).", ""]
    rows.append({"analysis": "jsd", "measure": "s1 vs s2-3", "diff": js, "lo": ci(js_b)[0], "hi": ci(js_b)[1]})
    rows.append({"analysis": "jsd", "measure": "split-half noise, s1", "diff": st.mean(noise)})
    # classes by harness and session
    lines += ["## Move classes by session and harness version", "", "| harness | session | " + " | ".join(CLASSES) + " |",
              "|---|---|" + "---|" * len(CLASSES)]
    for (fp, s) in sorted(by_harness):
        c = by_harness[fp, s]
        n = sum(c.values())
        lines.append(f"| {fp} | {s} | " + " | ".join(f"{100 * c[k] / n:.1f}%" for k in CLASSES) + " |")
    # by beat
    beats = ["none"] + sorted({x["beat"] for x in sess if x["beat"] != "none"})
    lines += ["", "## By the beat before the session (sessions 2-3; 'none' = session 1)", "",
              "| beat | sessions | " + " | ".join(CLASSES) + " | turns | member ends it | words per turn |",
              "|---|---|" + "---|" * (len(CLASSES) + 3)]
    for b in beats:
        xs = [x for x in sess if x["beat"] == b]
        if not xs:
            continue
        vals = {c: st.mean(x[c] for x in xs) for c in CLASSES}
        lines.append(f"| {b} | {len(xs)} | " + " | ".join(f"{100 * vals[c]:.1f}%" for c in CLASSES)
                     + f" | {st.mean(x['turns'] for x in xs):.1f} | {100 * st.mean(x['member_ended'] for x in xs):.0f}% | "
                     f"{st.mean(x['words'] for x in xs):.1f} |")
        rows.append({"analysis": "beat", "measure": b, "sessions": len(xs), **{c: vals[c] for c in CLASSES},
                     "turns": st.mean(x["turns"] for x in xs), "member_ended": st.mean(x["member_ended"] for x in xs),
                     "words": st.mean(x["words"] for x in xs)})
    lines += ["", "## Session dynamics by session", "", "| session | turns | member ends it | words per turn |",
              "|---|---|---|---|"]
    for s in (1, 2, 3):
        xs = [x for x in sess if x["session"] == s]
        lines.append(f"| {s} | {st.mean(x['turns'] for x in xs):.1f} | {100 * st.mean(x['member_ended'] for x in xs):.0f}% | "
                     f"{st.mean(x['words'] for x in xs):.1f} |")
    with (args.out / "patient_moves.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(rows, sess, CLASSES, args.out)


def plot(rows, sess, classes, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 220, "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    mv = [r for r in rows if r["analysis"] == "move"][::-1]
    ys = range(len(mv))
    a.barh([y + 0.2 for y in ys], [100 * r["session 1"] for r in mv], height=0.38, color="#9a9893", label="session 1")
    a.barh([y - 0.2 for y in ys], [100 * r["sessions 2-3"] for r in mv], height=0.38, color="#2a78d6",
           label="sessions 2-3")
    a.set_xscale("log")
    a.set_xticks([1, 3, 10, 30, 60], ["1", "3", "10", "30", "60"])
    a.set_yticks(list(ys), [r["measure"].replace("_", " ") for r in mv])
    a.set_xlabel("% of member turns (log scale)")
    a.legend(fontsize=6, loc="lower right")
    a.set_title("(a) Moves, session 1 against sessions 2-3", loc="left", fontsize=7.5)
    br = [r for r in rows if r["analysis"] == "beat"]
    cols = {"engaging": "#d9d8d4", "disclosing": "#2a78d6", "expressive": "#eda100", "concealing": "#4a3aa7"}
    for y, r in enumerate(br):
        left = 0.0
        for c in classes:
            b.barh(y, 100 * r[c], left=left, color=cols[c], height=0.62, edgecolor="white", linewidth=0.6,
                   label=c if y == 0 else None)
            left += 100 * r[c]
        b.text(101, y, f"{r['turns']:.0f} turns, {100 * r['member_ended']:.0f}% ended by member", va="center",
               fontsize=5.6, color="#52514e")
    b.set_yticks(range(len(br)), [("session 1, no beat" if r["measure"] == "none" else r["measure"].replace("_", " "))
                                  + f" ({r['sessions']})" for r in br])
    b.set_ylim(len(br) - 0.5, -0.5)
    b.set_xlim(0, 100)
    b.set_xlabel("% of member turns")
    b.legend(fontsize=5.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    b.set_title("(b) Move classes by the beat before the session", loc="left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "patient_moves.png")
    fig.savefig(out / "patient_moves.pdf")
    plt.close(fig)
    print("patient_moves: png, pdf")


if __name__ == "__main__":
    main()
