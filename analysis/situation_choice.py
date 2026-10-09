"""Does the counselor's memory in session 2 change what session 3 starts from?

    python analysis/situation_choice.py --root <memory_ablation dir> --batch main6 --out <dir>
        [--choice --model hosted_vllm/<name> --api_base <url>]

Session 3's situation is written by the patient model from the session-2 conversation, before the session-3
counselor says anything. For every member, three pairs of session-3 situations are compared:
  condition vs compacted   (none, full, split: the session-2 counselor had a different memory)
  rerun vs source          (the compacted rerun against the source run: the same memory, resampled) - the floor.

Similarity (no model): TF-IDF cosine between the two situations (event, thoughts, behaviours, opening message).
If memory changes where session 3 starts beyond resampling, a condition's situation sits further from compacted's
than two compacted resamples sit from each other. Paired within member, intervals over members.

Forced choice (--choice): the A1 question at this boundary. A model reads run A's session-2 conversation and two
session-3 situations, A's and B's, in a seeded random order, and says which was written after it; both directions
of each pair are asked. Accuracy above the rerun-vs-source floor means the session-2 conversation under a memory
condition leads somewhere resampling alone does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BOOT, ablation_counselors, ci, git_describe, mean, member_draws, runs_root  # noqa: E402

from mindeval import llm  # noqa: E402

WORD = re.compile(r"[a-z']+")
CHOICE = """Below is a conversation between a person and an AI counsellor, and two short descriptions of what happened
to that person afterwards, before their next conversation. Both are plausible. Exactly one of them was written as
the continuation of this particular conversation. Which one?

## The conversation

{transcript}

## Description A

{a}

## Description B

{b}

Answer with the letter only, A or B, on the last line."""


def situation_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    s = json.loads(path.read_text().split("\n", 1)[1])
    return "\n".join([s["event"], "; ".join(s["automatic_thoughts"]), "; ".join(s["behaviors"]),
                      s["opening_message"]])


def transcript(trace: Path) -> str | None:
    if not trace.is_file():
        return None
    rows = [json.loads(x) for x in trace.read_text().split("\n") if x.strip()]
    if rows[-1].get("type") != "session":
        return None
    return "\n".join(f"{'Counsellor' if m['role'] == 'user' else 'Person'}: {m['content']}"
                     for m in rows[-1]["member_messages"])


def tfidf_cosine(docs: dict[str, str]) -> callable:
    toks = {k: Counter(WORD.findall(v.lower())) for k, v in docs.items()}
    df = Counter(w for c in toks.values() for w in c)
    n = len(toks)
    vec = {k: {w: tf * math.log((1 + n) / (1 + df[w])) for w, tf in c.items()} for k, c in toks.items()}
    norm = {k: math.sqrt(sum(x * x for x in v.values())) or 1.0 for k, v in vec.items()}

    def cos(a: str, b: str) -> float:
        va, vb = vec[a], vec[b]
        return sum(x * vb.get(w, 0.0) for w, x in va.items()) / (norm[a] * norm[b])
    return cos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--batch", default="main6")
    ap.add_argument("--episode", default="ep002", help="the situation compared (ep002 = session 3)")
    ap.add_argument("--choice", action="store_true")
    ap.add_argument("--model", default="hosted_vllm/nemotron-3.5-super-ea")
    ap.add_argument("--api_base", default="http://worker-11:8067/v1")
    ap.add_argument("--max_tokens", type=int, default=48000, help="cut a runaway reasoning loop; retried on rerun")
    ap.add_argument("--runs_root", help="the source runs' root (else MINDEVAL_RUNS_ROOT); sources from runs.yaml")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    sources = {k: runs_root(args.runs_root) / c["path"] for k, c in ablation_counselors().items()}
    args.out.mkdir(parents=True, exist_ok=True)
    prev = f"ep{int(args.episode[2:]) - 1:03d}"
    lines, rows = [f"# Session-{int(args.episode[2:]) + 1} situations across memory conditions ({args.batch})", ""], []
    for cdir in sorted(p for p in (args.root / args.batch).iterdir() if p.is_dir()):
        c = cdir.name
        runs = {m.name: m for m in cdir.iterdir() if (m / "run.json").is_file()}
        runs["source"] = sources[c]
        members = json.loads((runs["compacted"] / "run.json").read_text())["member_ids"]
        docs = {}
        for name, run in runs.items():
            for m in members:
                t = situation_text(run / "members" / m / "arc" / args.episode / "situation.yaml")
                if t:
                    docs[f"{name}/{m}"] = t
        cos = tfidf_cosine(docs)
        pairs = [(x, "compacted") for x in ("none", "full", "split") if x in runs] + [("compacted", "source")]
        sims = {}
        for a, b in pairs:
            sims[a, b] = {m: cos(f"{a}/{m}", f"{b}/{m}") for m in members
                          if f"{a}/{m}" in docs and f"{b}/{m}" in docs}
        lines += [f"## {c}", "", "| pair | members | TF-IDF cosine | minus the floor (rerun vs source) [95% CI] |",
                  "|---|---|---|---|"]
        floor = sims["compacted", "source"]
        for (a, b), s in sims.items():
            shared = sorted(set(s) & set(floor))
            if (a, b) == ("compacted", "source"):
                lines.append(f"| rerun vs source (floor) | {len(s)} | {mean(s.values()):.3f} | - |")
                continue
            d = {m: s[m] - floor[m] for m in shared}
            lo, hi = ci([mean(d[m] for m in dr) for dr in member_draws(shared, BOOT)]) if shared else (None, None)
            lines.append(f"| {a} vs compacted | {len(s)} | {mean(s.values()):.3f} | "
                         + (f"{mean(d.values()):+.3f} [{lo:+.3f}, {hi:+.3f}] |" if shared else "- |"))
            rows.append({"counselor": c, "pair": f"{a}-compacted", "members": len(shared),
                         "cosine": mean(s.values()), "minus_floor": mean(d.values()) if shared else None,
                         "lo": lo, "hi": hi})

        if args.choice:
            cache_path = args.out / f"choice_{c}.jsonl"
            cache = {}
            if cache_path.is_file():  # answered choices are kept; failed ones (cut off, errors) are asked again, and so
                # is a choice whose prompt has changed since (a rerun condition); the earliest answers carry no hash
                cache = {x["key"]: x for x in map(json.loads, cache_path.read_text().splitlines())
                         if x["said"] in ("A", "B")}
            jobs = []
            for a, b in pairs:
                for m in members:
                    for x, y in ((a, b), (b, a)):
                        tr = transcript(runs[x] / "members" / m / "arc" / prev / "trace.jsonl")
                        if not tr or f"{x}/{m}" not in docs or f"{y}/{m}" not in docs:
                            continue
                        key = f"{x}|{y}|{m}"
                        flip = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)).random() < 0.5
                        da, db = (docs[f"{y}/{m}"], docs[f"{x}/{m}"]) if flip else (docs[f"{x}/{m}"], docs[f"{y}/{m}"])
                        prompt = CHOICE.format(transcript=tr, a=da, b=db)
                        sha = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                        if key in cache and cache[key].get("prompt_sha", sha) == sha:
                            continue
                        cache.pop(key, None)
                        jobs.append((key, a, b, m, "B" if flip else "A", prompt))

            def ask(job):
                key, a, b, m, right, prompt = job
                try:
                    text, meta = llm.call_messages([{"role": "user", "content": prompt}], args.model, args.api_base,
                                                   0.6, args.max_tokens, 3600, params={}, api_key="EMPTY")
                    if meta.get("finish_reason") == "length":
                        raise ValueError(f"cut at {args.max_tokens} tokens")
                    said = (re.findall(r"\b([AB])\b", text.split("</think>")[-1]) or ["?"])[-1]
                except Exception as e:  # noqa: BLE001
                    said = f"error: {e}"[:200]
                return {"key": key, "pair": f"{a}|{b}", "member": m, "right": right, "said": said,
                        "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16]}

            llm._lib()
            with ThreadPoolExecutor(max_workers=96) as pool, cache_path.open("a") as f:
                for fut in as_completed([pool.submit(ask, j) for j in jobs]):
                    res = fut.result()
                    f.write(json.dumps(res) + "\n")
                    f.flush()
                    cache[res["key"]] = res
            lines += ["", "| pair | asked | forced-choice accuracy [95% CI] |", "|---|---|---|"]
            for a, b in pairs:
                per = {}
                for x in cache.values():
                    if x["pair"] == f"{a}|{b}" and x["said"] in ("A", "B"):
                        per.setdefault(x["member"], []).append(float(x["said"] == x["right"]))
                ms = sorted(per)
                if not ms:
                    continue
                acc = mean(mean(v) for v in per.values())
                lo, hi = ci([mean(mean(per[m]) for m in d) for d in member_draws(ms, BOOT)])
                name = "rerun vs source (floor)" if b == "source" else f"{a} vs compacted"
                lines.append(f"| {name} | {sum(len(v) for v in per.values())} | {acc:.3f} [{lo:.3f}, {hi:.3f}] |")
                rows.append({"counselor": c, "pair": f"{a}-{b}", "choice_accuracy": acc, "lo": lo, "hi": hi,
                             "members": len(ms)})
        lines.append("")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"script": "analysis/situation_choice.py",
                                                        "analysis_git": git_describe(), "batch": args.batch,
                                                        "episode": args.episode, "sources": {k: str(v) for k, v in sources.items()},
                                                        "choice_model": args.model if args.choice else None},
                                                       indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
