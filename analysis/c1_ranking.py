"""C1, ranking stability: would session 1 alone rank the counselors as the whole arc does?

For each judge:
  - every counselor's mean MQM in sessions 1, 2 and 3 and over the arc, with its rank (1 = best);
  - Kendall's tau-b between the session-1 ranking and each later one, and the arc's, with 95% intervals from
    resampling members;
  - a permutation null for tau: within each member x counselor the three session labels are shuffled after
    removing each session's mean over counselors, so the null holds the counselor order fixed across sessions and
    keeps all of the member-level noise. p is the share of null taus at or below the observed one;
  - each counselor's within-member change from session 1, with 95% intervals;
  - every counselor pair: the gap in session 1 against the gap later, and whether it flips sign.

`--subset shared` keeps only the runs that share the schedule (one fingerprint); the default keeps all.

    python analysis/c1_ranking.py --runs_root <dir> --out <dir>/c1
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from common import (
    BOOT,
    EPISODES,
    ci,
    judge_identity,
    kendall_tau_b,
    load_judge,
    mean,
    member_draws,
    ranks,
    registry,
    runs_root,
    set_exclusions,
    write_manifest,
)

PERM = 2000
CONTRASTS = (("s1", "s2"), ("s1", "s3"), ("s2", "s3"), ("s1", "arc"), ("s1", "later"))


def table(sess: dict, runs: list[dict], members: list[str]) -> dict:
    """score[c][s][m] for s in s1, s2, s3; missing sessions are absent."""
    out = {}
    for r in runs:
        out[r["key"]] = {f"s{i + 1}": {m: sess[r["key"]][(m, ep)]["score"] for m in members
                                       if (m, ep) in sess[r["key"]]} for i, ep in enumerate(EPISODES)}
    return out


def level(score_c: dict, which: str, ms: list[str]) -> float:
    """A counselor's mean over the members `ms` (a resample may repeat them) for a session, the arc
    (each member's mean over sessions 1-3) or later (sessions 2-3)."""
    if which in ("s1", "s2", "s3"):
        return mean(score_c[which][m] for m in ms if m in score_c[which])
    parts = ("s1", "s2", "s3") if which == "arc" else ("s2", "s3")
    return mean(mean(score_c[p][m] for p in parts if m in score_c[p]) for m in ms
                if all(m in score_c[p] for p in parts))


def permuted(scores: dict, keys: list[str], members: list[str], rng: random.Random) -> dict:
    """One draw of the null: session labels shuffled within member x counselor, after centring each session on
    its mean over counselors (so a common session shift is not mistaken for a reordering)."""
    centre = {s: mean(mean(scores[k][s].values()) for k in keys) for s in ("s1", "s2", "s3")}
    out = {}
    for k in keys:
        out[k] = {"s1": {}, "s2": {}, "s3": {}}
        for m in members:
            if not all(m in scores[k][s] for s in ("s1", "s2", "s3")):
                continue
            vals = [scores[k][s][m] - centre[s] for s in ("s1", "s2", "s3")]
            rng.shuffle(vals)
            for s, v in zip(("s1", "s2", "s3"), vals):
                out[k][s][m] = v
    return out


def analyse(root: Path, runs: list[dict], judges: dict, judge: str, members: list[str]) -> dict:
    loaded = load_judge(root, runs, judge, judges)
    present = [r for r in runs if r["key"] in loaded["sessions"]]
    keys = [r["key"] for r in present]
    scores = table(loaded["sessions"], present, members)
    # Every statistic below is a difference of member means over the same resample, which is the member-paired
    # difference only when no session is missing; so missing sessions are refused rather than silently unpaired.
    incomplete = [k for k in keys if any(len(scores[k][s]) != len(members) for s in ("s1", "s2", "s3"))]
    if incomplete:
        loaded["problems"].append(f"dropped for missing sessions: {incomplete}")
        keys = [k for k in keys if k not in incomplete]
    windows = ("s1", "s2", "s3", "arc", "later")
    draws = member_draws(members)

    def levels(ms):
        return {k: {w: level(scores[k], w, ms) for w in windows} for k in keys}

    point = levels(members)
    boot = [levels(ms) for ms in draws]
    rk = {w: ranks({k: point[k][w] for k in keys}) for w in windows}
    n = {k: {s: len(scores[k][s]) for s in ("s1", "s2", "s3")} for k in keys}

    def taus_of(lv):
        return {f"{a}~{b}": kendall_tau_b([lv[k][a] for k in keys], [lv[k][b] for k in keys]) for a, b in CONTRASTS}

    # The interaction itself: how much counselors' changes from session 1 differ from each other (the SD over
    # counselors of later - s1). Zero if every counselor moved by the same amount; the permutation null gives what
    # member-level noise alone produces.
    def spread_of_change(lv, w="later"):
        xs = [lv[k][w] - lv[k]["s1"] for k in keys]
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    tau_point = taus_of(point)
    tau_boot = [taus_of(lv) for lv in boot]
    rng = random.Random(1)
    tau_null, inter_null = [], []
    for _ in range(PERM):
        perm = permuted(scores, keys, members, rng)
        lv = {k: {w: level(perm[k], w, members) for w in windows} for k in keys}
        tau_null.append(taus_of(lv))
        inter_null.append({w: spread_of_change(lv, w) for w in ("s2", "s3", "later")})
    interaction = []
    for w in ("s2", "s3", "later"):
        obs = spread_of_change(point, w)
        lo, hi = ci([spread_of_change(lv, w) for lv in boot])
        null = sorted(x[w] for x in inter_null)
        interaction.append({"judge": judge, "change": f"{w}-s1", "sd_of_change_over_counselors": obs, "lo": lo,
                            "hi": hi, "null_median": null[len(null) // 2], "null_95": null[int(0.95 * len(null))],
                            "p_perm": (1 + sum(1 for x in null if x >= obs)) / (1 + len(null))})
    tau_rows = []
    for name in tau_point:
        lo, hi = ci([t[name] for t in tau_boot])
        null = [t[name] for t in tau_null]
        p = (1 + sum(1 for x in null if x <= tau_point[name])) / (1 + len(null))
        tau_rows.append({"judge": judge, "contrast": name, "tau": tau_point[name], "lo": lo, "hi": hi,
                         "null_median": sorted(null)[len(null) // 2], "p_perm": p, "counselors": len(keys)})

    # Within-member change from session 1, and its excess over the mean change of all counselors.
    change_rows = []
    for k in keys:
        for later in ("s2", "s3"):
            def delta(lv, k=k, later=later):
                return lv[k][later] - lv[k]["s1"]

            def excess(lv, k=k, later=later):
                return delta(lv) - mean(lv[j][later] - lv[j]["s1"] for j in keys)

            d_lo, d_hi = ci([delta(lv) for lv in boot])
            x_lo, x_hi = ci([excess(lv) for lv in boot])
            change_rows.append({"judge": judge, "counselor": k, "later": later, "n_members": len(members),
                                "s1": point[k]["s1"], "later_mean": point[k][later],
                                "change": delta(point), "lo": d_lo, "hi": d_hi,
                                "excess_over_mean_change": excess(point), "x_lo": x_lo, "x_hi": x_hi})

    # Every pair: gap (a - b) in session 1 and later; a flip is a change of sign; it is credible when the
    # interaction (gap later - gap in session 1) has an interval excluding 0 and the later gap's interval does too.
    pair_rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            row = {"judge": judge, "a": a, "b": b}
            for w in ("s1", "s2", "s3", "later"):
                lo, hi = ci([lv[a][w] - lv[b][w] for lv in boot])
                row.update({f"gap_{w}": point[a][w] - point[b][w], f"gap_{w}_lo": lo, f"gap_{w}_hi": hi})
            for w in ("s2", "s3", "later"):
                inter = [(lv[a][w] - lv[b][w]) - (lv[a]["s1"] - lv[b]["s1"]) for lv in boot]
                lo, hi = ci(inter)
                g1, gw = row["gap_s1"], row[f"gap_{w}"]
                flips = (g1 > 0) != (gw > 0) and g1 != 0 and gw != 0
                row.update({f"interaction_{w}": gw - g1, f"interaction_{w}_lo": lo, f"interaction_{w}_hi": hi,
                            f"flip_{w}": flips,
                            f"credible_flip_{w}": flips and (lo > 0 or hi < 0)
                            and (row[f"gap_{w}_lo"] > 0 or row[f"gap_{w}_hi"] < 0)})
            pair_rows.append(row)

    # How far apart the counselors are, and how many pairs a window separates (gap CI excluding 0). s2 and s3 are
    # single sessions like s1, so they compare with it at equal size; `later` averages two.
    def sd(lv, w):
        xs = [lv[k][w] for k in keys]
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    spread = []
    for w in ("s1", "s2", "s3", "later", "arc"):
        lo, hi = ci([sd(lv, w) for lv in boot])
        r_lo, r_hi = ci([sd(lv, w) / sd(lv, "s1") for lv in boot])
        separated = sum(1 for r in pair_rows if r.get(f"gap_{w}_lo", 0) > 0 or r.get(f"gap_{w}_hi", 0) < 0) \
            if w != "arc" else None
        spread.append({"judge": judge, "window": w, "sd_between_counselors": sd(point, w), "lo": lo, "hi": hi,
                       "ratio_to_s1": sd(point, w) / sd(point, "s1"), "ratio_lo": r_lo, "ratio_hi": r_hi,
                       "pairs_separated": separated, "pairs": len(pair_rows)})
    # Fan-out: do counselors that start better improve more? Spearman between the session-1 level and the change.
    # Noise in session 1 biases this toward negative (regression to the mean), so a positive value is conservative.
    def spearman(xs, ys):
        rx, ry = ranks(dict(enumerate(xs))), ranks(dict(enumerate(ys)))
        a, b = [rx[i] for i in range(len(xs))], [ry[i] for i in range(len(ys))]
        ma, mb = mean(a), mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else float("nan")

    fan = []
    for w in ("s2", "s3", "later"):
        def rho(lv, w=w):
            return spearman([lv[k]["s1"] for k in keys], [lv[k][w] - lv[k]["s1"] for k in keys])
        lo, hi = ci([rho(lv) for lv in boot])
        fan.append({"judge": judge, "change": f"{w}-s1", "spearman_s1_vs_change": rho(point), "lo": lo, "hi": hi})

    return {"keys": keys, "point": point, "ranks": rk, "n": n, "taus": tau_rows, "changes": change_rows,
            "pairs": pair_rows, "spread": spread, "fan": fan, "interaction": interaction,
            "problems": loaded["problems"]}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def report(results: dict, labels: dict, subset: str) -> str:
    lines = [f"# C1: ranking stability ({subset} runs)", ""]
    for judge, res in results.items():
        keys = res["keys"]
        lines += [f"## Judge: {judge}", ""]
        if res["problems"]:
            lines += ["Coverage problems: " + "; ".join(res["problems"]), ""]
        lines += ["| counselor | s1 | rank | s2 | rank | s3 | rank | arc | rank | n (s1/s2/s3) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(keys, key=lambda k: res["point"][k]["arc"]):
            p, rk, n = res["point"][k], res["ranks"], res["n"][k]
            lines.append(f"| {labels[k]} | {p['s1']:.2f} | {rk['s1'][k]} | {p['s2']:.2f} | {rk['s2'][k]} | "
                         f"{p['s3']:.2f} | {rk['s3'][k]} | {p['arc']:.2f} | {rk['arc'][k]} | "
                         f"{n['s1']}/{n['s2']}/{n['s3']} |")
        lines += ["", "| rankings compared | Kendall tau-b | 95% CI | null median | p (perm) |", "|---|---|---|---|---|"]
        for t in res["taus"]:
            lines.append(f"| {t['contrast']} | {t['tau']:.3f} | [{t['lo']:.3f}, {t['hi']:.3f}] | "
                         f"{t['null_median']:.3f} | {t['p_perm']:.3f} |")
        moved = Counter(abs(res["ranks"]["s1"][k] - res["ranks"]["arc"][k]) for k in keys)
        lines += ["", "Rank positions moved between session 1 and the arc: "
                  + ", ".join(f"{d}: {moved[d]}" for d in sorted(moved)), ""]
        for w in ("s2", "s3", "later"):
            flips = [r for r in res["pairs"] if r[f"flip_{w}"]]
            cred = [r for r in res["pairs"] if r[f"credible_flip_{w}"]]
            inter = [r for r in res["pairs"] if r[f"interaction_{w}_lo"] > 0 or r[f"interaction_{w}_hi"] < 0]
            lines.append(f"- s1 vs {w}: {len(flips)} of {len(res['pairs'])} pairs flip sign; {len(inter)} have a "
                         f"gap change whose CI excludes 0; {len(cred)} flip with both CIs excluding 0"
                         + (": " + ", ".join(f"{labels[r['a']]} vs {labels[r['b']]}" for r in cred) if cred else ""))
        lines += ["", "| window | SD between counselors [95% CI] | ratio to s1 [95% CI] | pairs separated |",
                  "|---|---|---|---|"]
        for x in res["spread"]:
            sep = f"{x['pairs_separated']} of {x['pairs']}" if x["pairs_separated"] is not None else "-"
            lines.append(f"| {x['window']} | {x['sd_between_counselors']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] | "
                         f"{x['ratio_to_s1']:.2f} [{x['ratio_lo']:.2f}, {x['ratio_hi']:.2f}] | {sep} |")
        lines += [""] + [f"- Spearman(session-1 MQM, {x['change']}): {x['spearman_s1_vs_change']:+.2f} "
                         f"[{x['lo']:+.2f}, {x['hi']:+.2f}]" for x in res["fan"]]
        lines += [""] + [f"- Counselor x session interaction, SD over counselors of {x['change']}: "
                         f"{x['sd_of_change_over_counselors']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]; noise alone "
                         f"(permutation) median {x['null_median']:.3f}, 95th pct {x['null_95']:.3f}; "
                         f"p = {x['p_perm']:.4f}" for x in res["interaction"]]
        lines += ["", "| counselor | s2 - s1 [95% CI] | excess over mean change | s3 - s1 [95% CI] | excess |",
                  "|---|---|---|---|---|"]
        ch = {(r["counselor"], r["later"]): r for r in res["changes"]}
        for k in sorted(keys, key=lambda k: res["point"][k]["arc"]):
            a, b = ch[k, "s2"], ch[k, "s3"]
            lines.append(f"| {labels[k]} | {a['change']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}] | "
                         f"{a['excess_over_mean_change']:+.2f} [{a['x_lo']:+.2f}, {a['x_hi']:+.2f}] | "
                         f"{b['change']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] | "
                         f"{b['excess_over_mean_change']:+.2f} [{b['x_lo']:+.2f}, {b['x_hi']:+.2f}] |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", default="gpt61_sol,own,deepseek_v4,hy4")
    ap.add_argument("--subset", choices=("all", "shared"), default="all")
    ap.add_argument("--exclude", type=Path, help="replies to leave out of every score (artifacts.py's leaks.jsonl)")
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    if args.subset == "shared":
        runs = [r for r in runs if r["fingerprint"] == "7962e7caed157571"]
    members = runs[0]["run"]["member_ids"]
    assert all(r["run"]["member_ids"] == members for r in runs), "runs play different members"
    labels = {r["key"]: r["label"] for r in runs}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for judge in args.judges.split(","):
        res = analyse(root, runs, judges, judge, members)
        results[judge] = res
        rows = [{"judge": judge, "counselor": k, **{f"mqm_{w}": res["point"][k][w] for w in res["point"][k]},
                 **{f"rank_{w}": res["ranks"][w][k] for w in res["ranks"]},
                 **{f"n_{s}": res["n"][k][s] for s in res["n"][k]}} for k in res["keys"]]
        write_csv(out / f"scores_{judge}.csv", rows)
        write_csv(out / f"pairs_{judge}.csv", res["pairs"])
        write_csv(out / f"changes_{judge}.csv", res["changes"])
        print(f"{judge}: done ({len(res['keys'])} counselors)" + (f"; {res['problems']}" if res["problems"] else ""),
              flush=True)
    write_csv(out / "taus.csv", [t for res in results.values() for t in res["taus"]])
    write_csv(out / "spread.csv", [t for res in results.values() for t in res["spread"]])
    write_csv(out / "fan.csv", [t for res in results.values() for t in res["fan"]])
    write_csv(out / "interaction.csv", [t for res in results.values() for t in res["interaction"]])
    (out / "report.md").write_text(report(results, labels, args.subset))
    write_manifest(out, "analysis/c1_ranking.py", runs,
                   {j: judge_identity(root, runs, j, judges) for j in results},
                   {"subset": args.subset, "bootstrap": BOOT, "permutations": PERM})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
