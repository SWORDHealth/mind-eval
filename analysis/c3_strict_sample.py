"""The sample for strict re-ask and contradiction coding by reference coders, fixed by seed before any coding.

    python analysis/c3_strict_sample.py --runs_root <dir> --ablation <memory_ablation dir> --prior <c3_validation dir>
        --out <dir>

The LLM labelers do not find strict re-asks and contradictions reliably (c3_agreement.py), so these are coded by
blind reference coders with the same manual (`prompts/reask_contradict_v1.txt`), on a sample:
  paper      every paper counselor, sessions 2-3: the 2 sessions per counselor already coded for the labeler's
             validation plus 6 more, drawn here;
  ablation   sessions 2-6 of the memory ablation: 24 GLM-5.3 and 24 Gemma 4 12B sessions without counselor memory,
             where re-asks should be common, and 24 GLM-5.3 compacted sessions to compare;
  double     24 of the new sessions, coded a second time by another coder, for the coders' own agreement.
Coders see an opaque id only. Writes cases/<id>.md, keys.jsonl and batches.json (12 sessions a batch).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c3_label import opaque, session_input  # noqa: E402

PROMPT = Path(__file__).resolve().parent / "prompts" / "reask_contradict_v1.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root", required=True, type=Path)
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--prior", required=True, type=Path, help="the validation sample already coded (sample_keys.jsonl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tpl = PROMPT.read_text()
    prior = [json.loads(x) for x in (args.prior / "sample_keys.jsonl").read_text().splitlines()]
    prior_ids = {(p["name"], p["member"], p["episode"]) for p in prior}
    keys = [{**p, "group": "paper", "prior": True} for p in prior]
    new = []
    for r in yaml.safe_load((Path(__file__).resolve().parent / "runs.yaml").read_text())["runs"]:
        run = args.runs_root / r["path"]
        members = json.loads((run / "run.json").read_text())["member_ids"]
        cands = [(m, ep) for m in members for ep in ("ep001", "ep002") if (r["key"], m, ep) not in prior_ids]
        for m, ep in rng.sample(cands, 6):
            new.append({"id": opaque(r["key"], m, ep), "name": r["key"], "run": str(run), "member": m, "episode": ep,
                        "group": "paper", "prior": False})
    for c, mode in (("glm53", "none"), ("gemma4_12b", "none"), ("glm53", "compacted")):
        run = args.ablation / "main6" / c / mode
        members = json.loads((run / "run.json").read_text())["member_ids"]
        cands = [(m, f"ep{i:03d}") for m in members for i in range(1, 6)]
        for m, ep in rng.sample(cands, 24):
            name = f"{c}/{mode}"
            new.append({"id": opaque(name, m, ep), "name": name, "run": str(run), "member": m, "episode": ep,
                        "group": f"ablation {c} {mode}", "prior": False})
    keys += new
    double = rng.sample([k["id"] for k in new], 24)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cases").mkdir(exist_ok=True)
    for k in new:
        inp = session_input(Path(k["run"]) / "members" / k["member"] / "arc" / k["episode"] / "trace.jsonl")
        if inp is None:
            raise SystemExit(f"unfinished session in the sample: {k}")
        (args.out / "cases" / f"{k['id']}.md").write_text(
            f"# Session {k['id']}\n\n" + tpl.replace("{memory}", inp["memory"]).replace("{transcript}", inp["transcript"]))
    (args.out / "keys.jsonl").write_text("".join(json.dumps(k) + "\n" for k in keys))
    order = [k["id"] for k in new]
    rng.shuffle(order)
    batches = [order[i:i + 12] for i in range(0, len(order), 12)]
    # Second codes: each double-coded session goes to a batch other than the one that holds it.
    dbl = list(double)
    batches += [dbl[i:i + 12] for i in range(0, len(dbl), 12)]
    (args.out / "batches.json").write_text(json.dumps({"batches": batches, "double": double}, indent=1) + "\n")
    print(f"{len(new)} new sessions ({sum(1 for k in new if k['group'] == 'paper')} paper), {len(prior)} already coded, "
          f"{len(double)} double-coded; {len(batches)} batches")


if __name__ == "__main__":
    main()
