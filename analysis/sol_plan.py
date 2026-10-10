"""What GPT-6.1 Sol would cost to judge parts of the free-model ablation, from measured data. Nothing calls a model.

    python analysis/sol_plan.py --ablation <memory_ablation dir> --sol <dir of Sol's verdicts of the paper runs>
        --counselors glm53_nvfp4 qwen35_27b gemma4_12b [--members 25] [--seed 20261010]

Per view: the replies in sessions 2 onward and their judge-prompt size (MindJudge-27B's recorded prompt tokens on the
same prompts), priced with Sol's own profile (its output tokens per verdict and the cached share of its input, from
its verdicts of the paper runs) at --prices (input, cached input, output, per 1M tokens). A member subset is a fixed
random draw, made before any Sol verdict exists.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
from pathlib import Path

PLANS = {"core": [("compacted", "as_sent"), ("none", "reference")],
         "core + full as sent": [("compacted", "as_sent"), ("none", "reference"), ("full", "as_sent")]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--sol", required=True, type=Path)
    ap.add_argument("--counselors", nargs="+", required=True)
    ap.add_argument("--members", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--prices", default="2.00,0.20,10.00")
    args = ap.parse_args()
    p_in, p_cached, p_out = map(float, args.prices.split(","))
    out_toks, cached = [], []
    for f in glob.glob(f"{args.sol}/*/judge/mindeval2/verdicts.jsonl"):
        for line in open(f):
            u = json.loads(line).get("usage") or {}
            if u.get("prompt_tokens"):
                out_toks.append(u.get("completion_tokens") or 0)
                cached.append(((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) / u["prompt_tokens"])
    out, cache = st.mean(out_toks), st.mean(cached)
    print(f"Sol: {out:.0f} output tokens a verdict, {100 * cache:.0f}% of input cached ({len(out_toks)} verdicts)")
    members = json.loads((args.ablation / "main6" / args.counselors[0] / "compacted" / "run.json").read_text())["member_ids"]
    subset = set(random.Random(args.seed).sample(members, args.members))

    def view(c: str, m: str, v: str, keep: set | None) -> tuple[int, float, float]:
        toks = []
        for line in open(args.ablation / "judging" / "mindjudge" / "main6" / c / m / v / "judge" / "mindeval2" / "verdicts.jsonl"):
            x = json.loads(line)
            if "_error" in x or x.get("episode") == "ep000" or (keep and x["member"] not in keep):
                continue
            toks.append(((x.get("usage") or {}).get("prompt_tokens")) or 0)
        pin = st.mean(toks) if toks else 0.0
        return len(toks), pin, len(toks) * (pin * (1 - cache) * p_in + pin * cache * p_cached + out * p_out) / 1e6

    for name, views in PLANS.items():
        for label, keep in ((f"{args.members} members", subset), ("all members", None)):
            parts = [(c, m, v, *view(c, m, v, keep)) for c in args.counselors for m, v in views]
            print(f"\n{name}, {label}: ${sum(p[5] for p in parts):.0f}")
            for c, m, v, n, pin, cost in parts:
                print(f"    {c}/{m}/{v}: {n} replies, {pin:.0f} prompt tokens, ${cost:.0f}")


if __name__ == "__main__":
    main()
