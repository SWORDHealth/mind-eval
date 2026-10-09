"""What the memory ablation would cost with API models: counselors, and GPT-6.1 Sol as the judge.

    python analysis/api_cost.py --prices <estimate.py with MODELS> --ablation <ablation run dir> --api_run <run dir>
        [--members 50 --sessions 5]

Nothing here calls a model. Token profiles are measured:
  - the counselor's reply count, input and output tokens per reply, from an API counselor's existing run
    (`--api_run`, sessions 2-3);
  - the extra system-prompt tokens each memory mode adds, from the free-model ablation's traces (`--ablation`:
    the first reply's prompt tokens under each mode minus compacted, averaged over sessions 2 onward);
  - the judge's cost per verdict, from GPT-6.1 Sol's recorded usage on the paper runs (`--judge_dir`): its input
    tokens, how much of it was cached, and its output, priced at --judge_prices.
Prices per 1M tokens come from the MODELS table of `--prices` (input, cached input, output, cache-write multiplier).
Assumptions: conversation history is cached after its first use (the provider's prefix cache), memory text is cached
within a session, every non-compacted mode is judged in two views (as sent, and the reference memory), and the
patient (DeepSeek, self-hosted) is free.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import statistics as st
from pathlib import Path


def load_prices(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("prices", path)
    src = path.read_text().split("print(", 1)[0]  # the table, not the report it prints
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod.MODELS


def counselor_profile(run: Path) -> dict:
    replies, pin, pout, cached = [], [], [], []
    for f in glob.glob(f"{run}/members/*/arc/ep00[12]/trace.jsonl"):
        rows = [json.loads(x) for x in open(f)]
        u = [r["meta"]["counselor"].get("usage") or {} for r in rows
             if r.get("type") == "turn" and (r.get("meta") or {}).get("counselor")]
        replies.append(len(u))
        for x in u:
            pin.append(x.get("prompt_tokens") or 0)
            pout.append(x.get("completion_tokens") or 0)
            cached.append((x.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {"replies": st.mean(replies), "in": st.mean(pin), "out": st.mean(pout),
            "cached_share": sum(cached) / max(1, sum(pin))}


def memory_extra(ablation: Path) -> dict:
    """Extra counselor prompt tokens over compacted, per mode, sessions 2 onward (first reply of each session)."""
    first = {}
    for mode in ("compacted", "none", "full", "split"):
        xs = []
        for f in glob.glob(f"{ablation}/{mode}/members/*/arc/ep00[1-9]/trace.jsonl"):
            for line in open(f):
                r = json.loads(line)
                u = ((r.get("meta") or {}).get("counselor") or {}).get("usage")
                if r.get("type") == "turn" and u:
                    xs.append(u.get("prompt_tokens") or 0)
                    break
        first[mode] = st.mean(xs) if xs else 0
    return {m: first[m] - first["compacted"] for m in first}


def judge_profile(judge_dir: Path) -> dict:
    us = []
    for f in sorted(glob.glob(f"{judge_dir}/*/judge/mindeval2/verdicts.jsonl"))[:6]:
        for line in open(f):
            v = json.loads(line)
            u = v.get("usage")
            if u and v.get("episode") in ("ep001", "ep002"):
                us.append((u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                           u["completion_tokens"]))
    return {"in": st.mean(x[0] for x in us), "cached": st.mean(x[1] for x in us), "out": st.mean(x[2] for x in us)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--ablation", type=Path, required=True, help="one counselor's ablation runs, <dir>/<mode>/")
    ap.add_argument("--api_run", type=Path, required=True, help="an API counselor's existing run")
    ap.add_argument("--counselor", default="Claude Opus 5", help="its row in the price table")
    ap.add_argument("--judge_dir", type=Path, required=True)
    ap.add_argument("--judge_prices", default="2.00,0.20,10.00", help="GPT-6.1 Sol input, cached, output per 1M")
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--sessions", type=int, default=5, help="forked sessions per arc (2 to 6 = 5)")
    ap.add_argument("--new_session1", action="store_true", help="the counselor has no run at this fingerprint yet")
    args = ap.parse_args()
    prices = load_prices(args.prices)
    p_in, p_cache, p_out, write = prices[args.counselor]
    prof, extra, jp = counselor_profile(args.api_run), memory_extra(args.ablation), judge_profile(args.judge_dir)
    j_in, j_cache, j_out = (float(x) for x in args.judge_prices.split(","))
    print(f"counselor profile ({args.api_run.name}): {prof['replies']:.1f} replies/session, {prof['in']:.0f} input and "
          f"{prof['out']:.0f} output tokens/reply, cached share {prof['cached_share']:.2f}")
    print("extra memory tokens over compacted: " + ", ".join(f"{m} {x:+.0f}" for m, x in extra.items()))
    print(f"judge profile: {jp['in']:.0f} input ({jp['cached']:.0f} cached), {jp['out']:.0f} output tokens/verdict")
    total_c = total_j = 0.0
    print(f"\n{'mode':10} {'sessions':>8} {'counselor $':>12} {'judge views':>11} {'judge $':>9}")
    for mode in ("compacted", "none", "full", "split"):
        sessions = args.members * (args.sessions + (1 if args.new_session1 and mode == "compacted" else 0))
        replies = sessions * prof["replies"]
        tok_in = prof["in"] + extra[mode]
        new_in = tok_in * (1 - prof["cached_share"]) if prof["cached_share"] < 1 else tok_in * 0.1
        c = replies * (new_in * p_in * write + (tok_in - new_in) * p_cache + prof["out"] * p_out) / 1e6
        views = 1 if mode == "compacted" else 2
        jin = jp["in"] + extra[mode] / 2  # the as-sent view carries the mode's memory; the reference view does not
        j = views * replies * ((jin - jp["cached"]) * j_in + jp["cached"] * j_cache + jp["out"] * j_out) / 1e6
        total_c += c
        total_j += j
        print(f"{mode:10} {sessions:>8} {c:>12.2f} {views:>11} {j:>9.2f}")
    print(f"{'total':10} {'':>8} {total_c:>12.2f} {'':>11} {total_j:>9.2f}   = {total_c + total_j:.2f} USD")


if __name__ == "__main__":
    main()
