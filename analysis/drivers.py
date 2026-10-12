"""What moves the score: each reply's MQM weight split equally over the issues the judge flagged on it (a weighted
reply with no flag counts as "(no flag)"), so a session's MQM is the sum of its issues' parts.

    python analysis/drivers.py --runs_root <dir> --out <dir> [--exclude <leaks.jsonl>] [--ablation <memory_ablation dir>]

Paper runs, per judge: each counselor's change from session 1 to sessions 2-3 split by issue and by issue group,
mean over members with a member-bootstrap 95% interval, the mean over counselors (members resampled within each), and
how far the counselors' total change follows their change in the memory group (Spearman across counselors).
Ablation (--ablation): each memory condition minus the compacted rerun, sessions 2-6, split the same way, per
counselor and judge view, and the mean over counselors (the FP8 GLM-5.3 replicate left out).
Each reply counted once (the figure's split, `exclusive_parts`): a reply flagged for memory and for exploration is a
category of its own instead of half in each, with the share of replies flagged for each and for both.
Writes drivers.csv, drivers_exclusive.csv, ablation_drivers.csv, report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import pool  # noqa: E402
from common import (  # noqa: E402
    BOOT,
    EXCLUDE,
    ISSUES,
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

#: Issue groups, for the figure and the headline table.
GROUPS = {
    "memory": ("missed_shared_context", "false_memory"),
    "exploration": ("over_exploring", "under_guiding"),
    "advice and validation": ("gives_bad_advice", "uncritical_endorsement", "treating_perspective_as_fact",
                              "overstatement", "false_fact"),
    "style and scope": ("excessive_verbosity", "formulaic_responding", "anthropomorphizing", "out_of_scope_responding"),
    "other": ("other", "(no flag)"),
}
GROUP_OF = {i: g for g, xs in GROUPS.items() for i in xs}
PAPER_JUDGES = ("gpt61_sol", "own", "deepseek_v4", "hy4")
LABEL = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview",
         "mindjudge": "MindJudge-27B"}


def parts(verdicts: dict, drop: set[str] = frozenset()) -> dict[tuple[str, str], dict[str, float]]:
    """(member, episode) -> issue -> its share of the session's MQM (summing to the session's MQM), plus "_mqm"."""
    acc: dict = defaultdict(lambda: defaultdict(float))
    n: dict = defaultdict(int)
    for key, v in verdicts.items():
        if key in drop:
            continue
        k = (v["member"], v["episode"])
        n[k] += 1
        w = WEIGHT[v["severity"]]
        if not w:
            continue
        flagged = [i for i in ISSUES if v.get(i) == 1] or ["(no flag)"]
        for i in flagged:
            acc[k][i] += w / len(flagged)
    out = {}
    for k, c in n.items():
        d = {i: x / c for i, x in acc[k].items()}
        for g, xs in GROUPS.items():
            d[f"group:{g}"] = sum(d.get(i, 0.0) for i in xs)
        d["_mqm"] = sum(v for i, v in d.items() if not i.startswith("group:"))
        out[k] = d
    return out


#: Mutually exclusive categories (the figure's): every weighted reply counts once, in full. A reply flagged for both
#: memory and exploration (asking again about what the member already said is both) is a category of its own, so
#: neither group's part moves only because the other's flag joined the same reply, as it does under `parts`. Replies
#: with neither split their weight over their issues as in `parts`.
EXCLUSIVE = ("memory only", "memory and exploration", "exploration only", "advice and validation", "style and scope",
             "other")


def exclusive_parts(verdicts: dict, drop: set[str] = frozenset()) -> dict[tuple[str, str], dict[str, float]]:
    """(member, episode) -> category -> its share of the session's MQM (summing to the session's MQM), plus "_mqm"."""
    acc: dict = defaultdict(lambda: defaultdict(float))
    n: dict = defaultdict(int)
    for key, v in verdicts.items():
        if key in drop:
            continue
        k = (v["member"], v["episode"])
        n[k] += 1
        w = WEIGHT[v["severity"]]
        if not w:
            continue
        flagged = [i for i in ISSUES if v.get(i) == 1] or ["(no flag)"]
        mem = any(GROUP_OF[i] == "memory" for i in flagged)
        exp = any(GROUP_OF[i] == "exploration" for i in flagged)
        if mem or exp:
            acc[k]["memory and exploration" if mem and exp else "memory only" if mem else "exploration only"] += w
        else:
            for i in flagged:
                acc[k][GROUP_OF[i]] += w / len(flagged)
    out = {}
    for k, c in n.items():
        d = {cat: acc[k].get(cat, 0.0) / c for cat in EXCLUSIVE}
        d["_mqm"] = sum(d.values())
        out[k] = d
    return out


def per_member_change(p: dict, early: tuple[str, ...], late: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """member -> field -> mean over `late` sessions minus mean over `early` sessions (members with both)."""
    out = {}
    for m in sorted({m for (m, e) in p}):
        a = [p[m, e] for e in early if (m, e) in p]
        b = [p[m, e] for e in late if (m, e) in p]
        if a and b:
            fields = {f for d in a + b for f in d}
            out[m] = {f: mean(d.get(f, 0.0) for d in b) - mean(d.get(f, 0.0) for d in a) for f in fields}
    return out


def summarise(per: dict[str, dict[str, float]], fields: list[str], seed: int = 0) -> dict[str, dict]:
    ms = sorted(per)
    draws = member_draws(ms, BOOT, seed=seed)
    out = {}
    for f in fields:
        vals = {m: per[m].get(f, 0.0) for m in ms}
        lo, hi = ci([mean(vals[m] for m in d) for d in draws])
        out[f] = {"diff": mean(vals.values()), "lo": lo, "hi": hi, "_per": vals, "n_members": len(ms)}
    return out


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def fields_of() -> list[str]:
    return ["_mqm", *[f"group:{g}" for g in GROUPS], *ISSUES, "(no flag)"]


def flag_rates(verdicts: dict, drop: set[str] = frozenset()) -> dict[str, dict[str, float]]:
    """Session 1 and sessions 2-3: the share of replies flagged for exploration, for memory, and for both."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for key, v in verdicts.items():
        if key in drop:
            continue
        s = "session 1" if v["episode"] == "ep000" else "sessions 2-3"
        exp = any(v.get(i) == 1 for i in GROUPS["exploration"])
        mem = any(v.get(i) == 1 for i in GROUPS["memory"])
        acc[s]["exploration"].append(exp)
        acc[s]["memory"].append(mem)
        acc[s]["both"].append(exp and mem)
    return {s: {f: mean(xs) for f, xs in d.items()} for s, d in acc.items()}


def paper(root: Path, lines: list[str], rows: list[dict], ex_rows: list[dict]) -> None:
    runs, judges = registry(root)
    for judge in PAPER_JUDGES:
        per_c, per_x, rates = {}, {}, {}
        for r in runs:
            path = judge_dir(root, r, judge, judges) / "verdicts.jsonl"
            if not path.is_file():
                continue
            v, drop = load_verdicts(path), EXCLUDE.get(r["key"], set())
            p = parts(v, drop)
            per_c[r["label"]] = summarise(per_member_change(p, ("ep000",), ("ep001", "ep002")), fields_of())
            per_x[r["label"]] = summarise(per_member_change(exclusive_parts(v, drop), ("ep000",), ("ep001", "ep002")),
                                          ["_mqm", *EXCLUSIVE])
            rates[r["label"]] = flag_rates(v, drop)
        if not per_c:
            continue
        for c, s in per_x.items():
            for f, x in s.items():
                ex_rows.append({"judge": LABEL[judge], "counselor": c, "field": f,
                                **{k: v for k, v in x.items() if k != "_per"}})
        pooled_x = {f: pool({c: s[f]["_per"] for c, s in per_x.items()}) for f in ["_mqm", *EXCLUSIVE]}
        for f, x in pooled_x.items():
            ex_rows.append({"judge": LABEL[judge], "counselor": "mean of counselors", "field": f,
                            **{k: v for k, v in x.items() if k in ("diff", "lo", "hi", "n_counselors", "above", "below")}})
        rate = {s: {f: mean(rates[c][s][f] for c in rates if s in rates[c]) for f in ("exploration", "memory", "both")}
                for s in ("session 1", "sessions 2-3")}
        lines += [f"## {LABEL[judge]}: each reply counted once", "",
                  "Share of replies flagged (mean over counselors), session 1 → sessions 2-3: exploration "
                  + " → ".join(f"{100 * rate[s]['exploration']:.1f}%" for s in rate) + "; memory "
                  + " → ".join(f"{100 * rate[s]['memory']:.1f}%" for s in rate) + "; both "
                  + " → ".join(f"{100 * rate[s]['both']:.1f}%" for s in rate) + ".",
                  "Change from session 1 with every weighted reply in one category (mean over counselors): "
                  + ", ".join(f"{f} {pooled_x[f]['diff']:+.3f} [{pooled_x[f]['lo']:+.3f}, {pooled_x[f]['hi']:+.3f}]"
                              for f in EXCLUSIVE), ""]
        lines += [f"## {LABEL[judge]}: change from session 1 to sessions 2-3, by issue group", "",
                  "| counselor | MQM change | " + " | ".join(GROUPS) + " |", "|---|---|" + "---|" * len(GROUPS)]
        for c, s in sorted(per_c.items(), key=lambda kv: kv[1]["_mqm"]["diff"]):
            lines.append(f"| {c} | {s['_mqm']['diff']:+.2f} [{s['_mqm']['lo']:+.2f}, {s['_mqm']['hi']:+.2f}] | "
                         + " | ".join(f"{s[f'group:{g}']['diff']:+.2f}" for g in GROUPS) + " |")
            for f, x in s.items():
                rows.append({"judge": LABEL[judge], "counselor": c, "field": f,
                             **{k: v for k, v in x.items() if k != "_per"}})
        pooled = {f: pool({c: s[f]["_per"] for c, s in per_c.items()}) for f in fields_of()}
        lines += ["", f"Mean over {len(per_c)} counselors: MQM {pooled['_mqm']['diff']:+.3f} "
                  f"[{pooled['_mqm']['lo']:+.3f}, {pooled['_mqm']['hi']:+.3f}]; by issue (largest first):"]
        for f, x in sorted(((f, x) for f, x in pooled.items() if not f.startswith(("_", "group:"))),
                           key=lambda kv: -abs(kv[1]["diff"]))[:6]:
            lines.append(f"- {f}: {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]")
        for f, x in pooled.items():
            rows.append({"judge": LABEL[judge], "counselor": "mean of counselors", "field": f,
                         **{k: v for k, v in x.items() if k in ("diff", "lo", "hi", "n_counselors", "above", "below")}})
        tot = [s["_mqm"]["diff"] for s in per_c.values()]
        lines += ["", "Across counselors, the total change against each group's change (Spearman ρ): "
                  + ", ".join(f"{g} {spearman(tot, [s[f'group:{g}']['diff'] for s in per_c.values()]):+.2f}"
                              for g in GROUPS), ""]


def ablation(root: Path, lines: list[str], rows: list[dict], exclude_counselors: set[str]) -> None:
    batch = root / "judging"
    for judge in ("mindjudge", "deepseek_v4"):
        for view in ("as_sent", "reference"):
            for mode in ("none", "full", "split"):
                per_c = {}
                for cdir in sorted((batch / judge / "main6").iterdir()):
                    if cdir.name in exclude_counselors:
                        continue
                    comp = cdir / "compacted" / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"
                    cond = cdir / mode / view / "judge" / "mindeval2" / "verdicts.jsonl"
                    if not (comp.is_file() and cond.is_file()):
                        continue
                    pc, pd = parts(load_verdicts(comp)), parts(load_verdicts(cond))
                    eps = sorted({e for (m, e) in pd if e != "ep000"})
                    diff = {}
                    for m in sorted({m for (m, e) in pd}):
                        shared = [e for e in eps if (m, e) in pd and (m, e) in pc]
                        if shared:
                            fields = {f for e in shared for f in list(pd[m, e]) + list(pc[m, e])}
                            diff[m] = {f: mean(pd[m, e].get(f, 0.0) - pc[m, e].get(f, 0.0) for e in shared)
                                       for f in fields}
                    if diff:
                        per_c[cdir.name] = summarise(diff, fields_of())
                if not per_c:
                    continue
                pooled = {f: pool({c: s[f]["_per"] for c, s in per_c.items()}) for f in fields_of()}
                top = sorted(((f, x) for f, x in pooled.items() if not f.startswith(("_", "group:"))),
                             key=lambda kv: -abs(kv[1]["diff"]))[:4]
                lines.append(f"- {LABEL[judge]}, {view.replace('_', ' ')}, {mode} minus compacted "
                             f"({len(per_c)} counselors): MQM {pooled['_mqm']['diff']:+.2f}; groups "
                             + ", ".join(f"{g} {pooled[f'group:{g}']['diff']:+.2f}" for g in GROUPS)
                             + "; issues " + ", ".join(f"{f} {x['diff']:+.2f}" for f, x in top))
                for f, x in pooled.items():
                    rows.append({"judge": LABEL[judge], "view": view, "mode": mode, "field": f,
                                 **{k: v for k, v in x.items() if k in ("diff", "lo", "hi", "n_counselors", "above",
                                                                        "below")}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", type=Path, help="replies to leave out (artifacts.py's leaks.jsonl)")
    ap.add_argument("--ablation", type=Path)
    ap.add_argument("--ablation_exclude", default="glm53", help="ablation counselors left out (replicates)")
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# What moves the score", "",
             "Each reply's MQM weight is split equally over the issues flagged on it; a session's MQM is the sum of "
             "its issues' parts. Changes are means over members [95% member-bootstrap interval]."
             + (f" Replies in {args.exclude.name} left out." if args.exclude else ""), ""]
    rows: list[dict] = []
    ex_rows: list[dict] = []
    paper(root, lines, rows, ex_rows)
    for name, rs in (("drivers.csv", rows), ("drivers_exclusive.csv", ex_rows)):
        with (args.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rs for k in r}))
            w.writeheader()
            w.writerows(rs)
    if args.ablation:
        ab_rows: list[dict] = []
        lines += ["## Memory ablation: what each condition moves (mean over counselors, sessions 2-6)", ""]
        ablation(args.ablation, lines, ab_rows, set(args.ablation_exclude.split(",")))
        with (args.out / "ablation_drivers.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in ab_rows for k in r}))
            w.writeheader()
            w.writerows(ab_rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"script": "analysis/drivers.py", "exclude": str(args.exclude),
                                                        "ablation": str(args.ablation)}, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
