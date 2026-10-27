"""Does the counselor's session 1 shape the next scenario, or is the change sampling? (paper runs, session 2)

    python analysis/situation_variation.py --runs_root <dir> --resample <situation_resample dir> --out <dir>

Inputs: the session-2 situations situation_resample.py wrote again from each run's own session-1 transcript (replicate
0 with the arc's seed, 1..N with other seeds; one serving for all), the paper runs' original session-2 situations, the
shared session-1 situation, and GPT-6.1 Sol's session-1 MQM of each run. A situation's text is its event, opening
message, goal, checkpoints, automatic thoughts, behaviours and physical signs; similarity is TF-IDF cosine (English stop
words, fitted on every situation here).
  similarity   mean cosine between: replicates of the same transcript (sampling alone); regenerations from different
               counselors' transcripts of the same member (counselor + sampling); the paper's original situations of the
               same member across counselors; a regeneration and the original of the same transcript (serving +
               sampling); situations of different members (the floor). The counselor's share of a member's scenario
               variation: (within-transcript - across-counselor) / (within-transcript - different-member), resampling
               members for the interval: 0 when another counselor's session changes the next scenario no more than
               sampling does.
  quality      within member, across counselors: Spearman between session-1 MQM (Sol) and the next scenario's features
               (its similarity to session 1, its number of distressing items: automatic thoughts, behaviours, physical
               signs), averaged over members, with a permutation p (MQM shuffled within member); regenerations averaged
               over replicates 1..N, and the original situations.
Writes situation_variation.csv, report.md, situation_variation.png/.pdf. Needs scikit-learn and matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BOOT, WEIGHT, ci, judge_dir, load_verdicts, member_draws, registry, runs_root  # noqa: E402


def text_of(s: dict) -> str:
    parts = [s.get("event") or "", s.get("opening_message") or "", s.get("goal") or ""]
    for k in ("checkpoints", "automatic_thoughts", "behaviors", "physical"):
        parts += [str(x) for x in (s.get(k) or [])]
    return " ".join(parts)


def distress(s: dict) -> int:
    return sum(len(s.get(k) or []) for k in ("automatic_thoughts", "behaviors", "physical"))


def situation_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").split("\n", 1)[1])


def spearman(x, y) -> float:
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = float(k)
        return r
    rx, ry = rk(x), rk(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--resample", required=True, type=Path)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    rkeys = sorted(d.name for d in args.resample.iterdir() if d.is_dir())
    byk = {r["key"]: r for r in runs}
    regen = defaultdict(dict)   # (run, member) -> replicate -> situation
    for k in rkeys:
        for f in (args.resample / k).glob("*/r*.json"):
            d = json.loads(f.read_text())
            regen[k, d["member"]][d["replicate"]] = d["situation"]
    orig, s1, mqm, beat = {}, {}, {}, {}
    for k in rkeys:
        r = byk[k]
        v = load_verdicts(judge_dir(root, r, "gpt61_sol", judges) / "verdicts.jsonl")
        per = defaultdict(list)
        for x in v.values():
            if x["episode"] == "ep000":
                per[x["member"]].append(WEIGHT[x["severity"]])
        for m in r["run"]["member_ids"]:
            arc = Path(r["dir"]) / "members" / m / "arc"
            if (arc / "ep001" / "situation.yaml").is_file():
                orig[k, m] = situation_file(arc / "ep001" / "situation.yaml")
            if m not in s1 and (arc / "ep000" / "situation.yaml").is_file():
                s1[m] = situation_file(arc / "ep000" / "situation.yaml")
                beat[m] = json.loads((arc / "arc.json").read_text())["shape"]["beats"][0]
            if per.get(m):
                mqm[k, m] = st.mean(per[m])
    reps = sorted({rep for d in regen.values() for rep in d if rep > 0})
    members = sorted({m for (k, m), d in regen.items() if all(rep in d for rep in reps)
                      and sum(1 for kk in rkeys if all(rr in regen.get((kk, m), {}) for rr in reps)) >= 3})
    # TF-IDF over every situation
    docs, idx = [], {}
    def add(key, s):
        idx[key] = len(docs)
        docs.append(text_of(s))
    for (k, m), d in regen.items():
        for rep, s in d.items():
            add(("regen", k, m, rep), s)
    for (k, m), s in orig.items():
        add(("orig", k, m), s)
    for m, s in s1.items():
        add(("s1", m), s)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    X = X.multiply(1 / np.sqrt(X.multiply(X).sum(axis=1))).tocsr()
    def cos(a, b) -> float:
        return float(X[idx[a]].multiply(X[idx[b]]).sum())

    per_m = {}
    rng = random.Random(0)
    for m in members:
        ks = [k for k in rkeys if all(rep in regen.get((k, m), {}) for rep in reps)]
        within = [cos(("regen", k, m, a), ("regen", k, m, b)) for k in ks for i, a in enumerate(reps) for b in reps[i + 1:]]
        across = [cos(("regen", k1, m, a), ("regen", k2, m, b)) for i, k1 in enumerate(ks) for k2 in ks[i + 1:]
                  for a in reps for b in reps]
        oo = [cos(("orig", k1, m), ("orig", k2, m)) for i, k1 in enumerate(ks) for k2 in ks[i + 1:]
              if (k1, m) in orig and (k2, m) in orig]
        ro = [cos(("regen", k, m, 0), ("orig", k, m)) for k in ks if 0 in regen[k, m] and (k, m) in orig]
        others = [x for x in members if x != m]
        floor = [cos(("regen", k, m, reps[0]), ("regen", k, x, reps[0])) for k in ks
                 for x in rng.sample(others, 3) if reps[0] in regen.get((k, x), {})]
        cont = [cos(("regen", k, m, a), ("s1", m)) for k in ks for a in reps]
        per_m[m] = {"within": st.mean(within), "across": st.mean(across), "orig_across": st.mean(oo) if oo else None,
                    "regen_vs_orig": st.mean(ro) if ro else None, "floor": st.mean(floor), "continuity": st.mean(cont)}
    draws = member_draws(members, BOOT)
    rows, lines = [], [f"# Does session 1 shape the next scenario? (session 2; {len(members)} members, {len(rkeys)} "
                       f"counselors, regenerations: replicate 0 and {len(reps)} more per transcript)", ""]
    lines += ["## Similarity of session-2 situations (TF-IDF cosine, mean over members [95% CI])", "",
              "| comparison | cosine |", "|---|---|"]
    labels = {"within": "same session-1 transcript, regenerated (sampling alone)",
              "across": "different counselors' transcripts, same member, regenerated",
              "orig_across": "the paper's original situations, same member, different counselors",
              "regen_vs_orig": "regeneration (arc's own seed) against the original, same transcript",
              "floor": "different members", "continuity": "regenerated session 2 against the session-1 situation"}
    stats = {}
    for key, lab in labels.items():
        vals = {m: per_m[m][key] for m in members if per_m[m][key] is not None}
        mu = st.mean(vals.values())
        lo, hi = ci([st.mean(vals[m] for m in d if m in vals) for d in draws])
        stats[key] = (mu, lo, hi)
        lines.append(f"| {lab} | {mu:.3f} [{lo:.3f}, {hi:.3f}] |")
        rows.append({"analysis": "similarity", "measure": key, "value": mu, "lo": lo, "hi": hi})
    share = lambda ms: (st.mean(per_m[m]["within"] for m in ms) - st.mean(per_m[m]["across"] for m in ms)) / (  # noqa: E731
        st.mean(per_m[m]["within"] for m in ms) - st.mean(per_m[m]["floor"] for m in ms))
    sh = share(members)
    lo, hi = ci([share(d) for d in draws])
    lines += ["", f"- The counselor's share of a member's scenario variation: {100 * sh:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] "
              "(0: another counselor's session 1 changes the next scenario no more than sampling does; 100%: as much as "
              "another member).", ""]
    rows.append({"analysis": "share", "measure": "counselor share", "value": sh, "lo": lo, "hi": hi})
    # quality: within member, across counselors
    lines += ["## Session-1 quality (Sol MQM) against the next scenario, within member (mean Spearman over members)", "",
              "| feature | regenerated (mean of replicates) | permutation p | original | permutation p |",
              "|---|---|---|---|---|"]
    feats = {"similarity to the session-1 situation": lambda s, m: cos(s, ("s1", m)),
             "distressing items (thoughts, behaviours, physical)": None}
    for fname, f in feats.items():
        res = []
        for source in ("regen", "orig"):
            per_mem = {}
            for m in members:
                ks = [k for k in rkeys if (k, m) in mqm and (((k, m) in regen and all(r in regen[k, m] for r in reps))
                                                            if source == "regen" else (k, m) in orig)]
                if len(ks) < 4:
                    continue
                if source == "regen":
                    ys = [st.mean((f(("regen", k, m, r), m) if f else distress(regen[k, m][r])) for r in reps) for k in ks]
                else:
                    ys = [(f(("orig", k, m), m) if f else distress(orig[k, m])) for k in ks]
                per_mem[m] = ([mqm[k, m] for k in ks], ys)
            obs = st.mean(spearman(x, y) for x, y in per_mem.values())
            null = []
            prng = random.Random(1)
            for _ in range(args.perms):
                tot = []
                for x, y in per_mem.values():
                    xx = x[:]
                    prng.shuffle(xx)
                    tot.append(spearman(xx, y))
                null.append(st.mean(tot))
            p = (1 + sum(abs(n_) >= abs(obs) for n_ in null)) / (1 + len(null))
            res.append((obs, p))
            rows.append({"analysis": "quality", "measure": f"{fname} ({source})", "value": obs, "p": p,
                         "members": len(per_mem)})
        lines.append(f"| {fname} | {res[0][0]:+.3f} | {res[0][1]:.3f} | {res[1][0]:+.3f} | {res[1][1]:.3f} |")
    # by beat: within vs across
    lines += ["", "## By the beat before session 2", "", "| beat | members | within transcript | across counselors | "
              "counselor share |", "|---|---|---|---|---|"]
    for b in sorted({beat[m] for m in members}):
        ms = [m for m in members if beat[m] == b]
        if len(ms) < 3:
            continue
        lines.append(f"| {b} | {len(ms)} | {st.mean(per_m[m]['within'] for m in ms):.3f} | "
                     f"{st.mean(per_m[m]['across'] for m in ms):.3f} | {100 * share(ms):.0f}% |")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "situation_variation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(stats, labels, args.out)


def plot(stats: dict, labels: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    order = ["within", "regen_vs_orig", "across", "orig_across", "continuity", "floor"]
    colors = {"within": "#2a78d6", "regen_vs_orig": "#9cc3ef", "across": "#eb6834", "orig_across": "#f4a582",
              "continuity": "#1baf7a", "floor": "#9a9893"}
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ys = range(len(order))[::-1]
    for y, k in zip(ys, order):
        mu, lo, hi = stats[k]
        ax.barh(y, mu, color=colors[k], height=0.6)
        ax.hlines(y, lo, hi, color="#0b0b0b", lw=0.8)
    ax.set_yticks(list(ys), [labels[k] for k in order], fontsize=6.2)
    ax.set_xlabel("TF-IDF cosine between session-2 situations (mean over members)")
    ax.grid(axis="x", color="#e4e3df", lw=0.5)
    ax.set_title("Does the counselor's session 1 shape the next scenario?", loc="left", fontsize=7.5)
    fig.savefig(out / "situation_variation.png")
    fig.savefig(out / "situation_variation.pdf")
    plt.close(fig)
    print("situation_variation: png, pdf")


if __name__ == "__main__":
    main()
