"""C3 with strict re-asks and contradictions, coded by reference coders on a sample (c3_strict_sample.py).

    python analysis/c3_strict.py --runs_root <dir> --strict <c3_strict dir> --prior <c3_validation dir>
        --ablation <memory_ablation dir> --out <dir>

Reports:
  - the reference coders' own agreement, on the sessions two coders coded;
  - re-asks and contradictions per session, and the share of sessions with either: paper runs (sessions 2-3), and
    the ablation's sessions 2-6 without counselor memory against compacted; 95% intervals resample sessions;
  - the cross-tab: of the replies coded as a re-ask or a contradiction, the share each judge flags with a memory issue
    (missed_shared_context or false_memory) and with Major or Critical, against every other reply of the same sessions;
    and of the sessions with such a reply, the share where the judge flags any memory issue, against the rest.
Paper sessions are judged by GPT-6.1 Sol, MindJudge-27B, DeepSeek-V4.1 and Hy4; ablation sessions by MindJudge-27B and
DeepSeek-V4.1, as sent and against the reference memory (judge_views.py).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import ci, judge_dir, load_verdicts, mean, registry, runs_root

MEMORY = ("missed_shared_context", "false_memory")
JUDGE_LABEL = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview",
               "mindjudge": "MindJudge-27B"}


def load_codes(paths) -> dict[str, dict]:
    out = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                out[x["id"]] = {"reasks": [r for r in x.get("reasks", []) if isinstance(r.get("turn"), int)],
                                "contradictions": [c for c in x.get("contradictions", []) if isinstance(c.get("turn"), int)]}
    return out


def kappa(pairs) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_mean(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    m = mean(xs)
    lo, hi = ci([mean(rng.choice(xs) for _ in xs) for _ in range(n)]) if xs else (float("nan"), float("nan"))
    return m, lo, hi


def verdicts_by_turn(path: Path) -> dict[tuple[str, str], dict[int, dict]]:
    out: dict = defaultdict(dict)
    if path.is_file():
        for v in load_verdicts(path).values():
            out[v["member"], v["episode"]][v["turn"]] = v
    return out


def crosstab(sessions: list[dict], vt: dict) -> dict | None:
    """sessions: [{member, episode, bad_turns}] -> shares for the judge whose verdicts are vt."""
    hit = {"bad": [0, 0, 0], "other": [0, 0, 0]}
    sess = {"bad": [0, 0], "other": [0, 0]}
    for s in sessions:
        vs = vt.get((s["member"], s["episode"]))
        if not vs:
            continue
        any_mem = False
        for turn, v in vs.items():
            mem = any(v.get(f) == 1 for f in MEMORY)
            any_mem |= mem
            h = hit["bad" if turn in s["bad_turns"] else "other"]
            h[0] += 1
            h[1] += mem
            h[2] += v["severity"] in ("Major", "Critical")
        k = "bad" if s["bad_turns"] else "other"
        sess[k][0] += 1
        sess[k][1] += any_mem
    b, o = hit["bad"], hit["other"]
    if not b[0]:
        return None
    return {"failure_replies": b[0], "memory_flag_failure": b[1] / b[0], "major_failure": b[2] / b[0],
            "other_replies": o[0], "memory_flag_other": o[1] / o[0] if o[0] else None,
            "major_other": o[2] / o[0] if o[0] else None,
            "sessions_with_failure": sess["bad"][0],
            "session_memory_flag_failure": sess["bad"][1] / sess["bad"][0] if sess["bad"][0] else None,
            "sessions_without": sess["other"][0],
            "session_memory_flag_other": sess["other"][1] / sess["other"][0] if sess["other"][0] else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--strict", required=True, type=Path)
    ap.add_argument("--prior", required=True, type=Path)
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    keys = {k["id"]: k for k in map(json.loads, (args.strict / "keys.jsonl").read_text().splitlines())}
    codes = load_codes(sorted((args.strict / "codes").glob("*.jsonl")) + sorted((args.prior / "rc_ref").glob("*.jsonl")))
    second = load_codes(sorted((args.strict / "codes2").glob("*.jsonl")))
    lines = ["# C3 with strict re-asks and contradictions (reference coders, sampled)", ""]

    # Coders' own agreement.
    both = sorted(set(codes) & set(second))
    agree = {}
    for kind in ("reasks", "contradictions"):
        pairs = [(bool(codes[i][kind]), bool(second[i][kind])) for i in both]
        t1 = {(i, x["turn"]) for i in both for x in codes[i][kind]}
        t2 = {(i, x["turn"]) for i in both for x in second[i][kind]}
        agree[kind] = {"sessions": len(both), "kappa_any": kappa(pairs), "coder1": len(t1), "coder2": len(t2),
                       "same_turn": len(t1 & t2)}
    lines += [f"Coder agreement on {len(both)} double-coded sessions: " + "; ".join(
        f"{k}: any-{k[:-1]} kappa {v['kappa_any']:.2f}, turns {v['coder1']} vs {v['coder2']} ({v['same_turn']} the same)"
        for k, v in agree.items()), ""]

    # Rates per group.
    groups = defaultdict(list)
    for sid, c in codes.items():
        k = keys.get(sid)
        if k is None:
            continue
        groups[k["group"]].append({"id": sid, **k, "reasks": len(c["reasks"]), "contra": len(c["contradictions"]),
                                   "bad_turns": {x["turn"] for x in c["reasks"] + c["contradictions"]}})
    rate_rows = []
    lines += ["| sessions | n | re-asks / session [95% CI] | contradictions / session | sessions with either |",
              "|---|---|---|---|---|"]
    for g in sorted(groups):
        ss = groups[g]
        r = boot_mean([s["reasks"] for s in ss], seed=1)
        c = boot_mean([s["contra"] for s in ss], seed=2)
        e = boot_mean([float(bool(s["bad_turns"])) for s in ss], seed=3)
        rate_rows.append({"group": g, "n": len(ss), "reasks": r, "contradictions": c, "either": e})
        lines.append(f"| {g} | {len(ss)} | {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}] | {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}] | "
                     f"{100 * e[0]:.0f}% [{100 * e[1]:.0f}, {100 * e[2]:.0f}] |")

    # Cross-tab: paper sessions, four judges.
    by_key = {r["key"]: r for r in runs}
    cross = []
    for j in ("gpt61_sol", "own", "deepseek_v4", "hy4"):
        bad, other_sess = [], []
        vt_all = {}
        for s in groups.get("paper", []):
            r = by_key[s["name"]]
            if r["key"] not in vt_all:
                vt_all[r["key"]] = verdicts_by_turn(judge_dir(root, r, j, judges) / "verdicts.jsonl")
            vs = vt_all[r["key"]].get((s["member"], s["episode"]))
            (bad if s["bad_turns"] else other_sess).append((s, vs))
        merged = {}
        sess = []
        for s, vs in bad + other_sess:
            if vs:
                merged[(s["name"], s["member"], s["episode"])] = vs
                sess.append({"member": (s["name"], s["member"], s["episode"]), "episode": "", "bad_turns": s["bad_turns"]})
        vt = {(k, ""): v for k, v in merged.items()}
        x = crosstab(sess, vt)
        if x:
            cross.append({"set": "paper runs, sessions 2–3", "judge": JUDGE_LABEL[j], "view": "as sent", **x})
    # Cross-tab: ablation sessions, two judges, two views.
    for g, ss in groups.items():
        if not g.startswith("ablation"):
            continue
        _, c, mode = g.split()
        for j in ("mindjudge", "deepseek_v4"):
            for view in ("as_sent", "reference"):
                if mode == "compacted" and view == "reference":
                    continue
                vp = args.ablation / "judging" / j / "main6" / c / mode / view / "judge" / "mindeval2" / "verdicts.jsonl"
                x = crosstab([{"member": s["member"], "episode": s["episode"], "bad_turns": s["bad_turns"]} for s in ss],
                             verdicts_by_turn(vp))
                if x:
                    cross.append({"set": f"ablation {c} {mode}", "judge": JUDGE_LABEL[j],
                                  "view": "as sent" if view == "as_sent" else "reference memory", **x})

    def pct(v):
        return "-" if v is None else f"{100 * v:.0f}%"
    lines += ["", "| sessions | judge | view | coded failure replies | memory issue flagged | on other replies | "
              "Major+ on failures | on other replies | sessions with a failure: any memory flag | other sessions |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for x in cross:
        lines.append(f"| {x['set']} | {x['judge']} | {x['view']} | {x['failure_replies']} | {pct(x['memory_flag_failure'])} | "
                     f"{pct(x['memory_flag_other'])} | {pct(x['major_failure'])} | {pct(x['major_other'])} | "
                     f"{pct(x['session_memory_flag_failure'])} of {x['sessions_with_failure']} | "
                     f"{pct(x['session_memory_flag_other'])} of {x['sessions_without']} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "results.json").write_text(json.dumps({"agreement": agree, "rates": rate_rows, "crosstab": cross},
                                                      indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
