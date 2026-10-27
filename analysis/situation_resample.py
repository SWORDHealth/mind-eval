"""Write the session-2 situation again from the same session-1 transcript, several times: the situation writer's sampling.

    python analysis/situation_resample.py --runs_root <dir> --runs qwen35_27b,... --out <dir> [--replicates 3]
        --patient_model hosted_vllm/<name> --patient_base <url> [--patient_key_file <path>] [--workers 32]

For each run (paper runs at the current harness, where every counselor meets each member with the same schedule) and
member: the session-1 situation, the session-1 record (its trace's last line), the gap, the beat and the clock come from
the run's own files, and the patient model writes the session-2 situation with the same prompt as the harness
(arc.evolve_situation). Replicate 0 uses the arc's own seed; replicates 1..N change only the seed (a suffix on the
arc id, which nothing but the seed reads). The across-counselor variation of these situations, set against their
within-transcript variation, separates what the counselor's session changed from what sampling changes.
Writes <out>/<run>/<member>/r<k>.json; reruns skip what exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import registry, runs_root  # noqa: E402

from mindeval.arc import Anchor, arc_id_of, describe_ts, evolve_situation  # noqa: E402
from mindeval.config import CallSpec, load_config  # noqa: E402
from mindeval.models import MemberRow, Situation  # noqa: E402
from mindeval.session import read_trace  # noqa: E402


def situation_from_file(path: Path) -> Situation:
    return Situation.model_validate(json.loads(path.read_text(encoding="utf-8").split("\n", 1)[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--patient_model", required=True)
    ap.add_argument("--patient_base", required=True)
    ap.add_argument("--patient_key_file")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--members", type=int, help="only the first N members of each run (a test)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config()
    root = runs_root(args.runs_root)
    runs, _ = registry(root)
    key = Path(args.patient_key_file).read_text().strip() if args.patient_key_file else "EMPTY"
    jobs = []
    for r in runs:
        if r["key"] not in args.runs.split(","):
            continue
        if r["run"]["fingerprint"] != cfg.fingerprint:
            raise SystemExit(f"{r['key']} is at {r['run']['fingerprint']}, this code at {cfg.fingerprint}")
        run = r["run"]
        p = run["patient"]
        patient = CallSpec(model=args.patient_model, api_base=args.patient_base, api_key=key,
                           temperature=p["temperature"], max_tokens=p["max_tokens"], timeout=p["timeout"],
                           max_retries=p.get("max_retries", 3), params=p.get("params") or {})
        rows = {}
        for line in Path(run["profiles"]["path"]).read_text().splitlines():
            row = MemberRow.model_validate_json(line)
            rows[row.member_id] = row
        for m in run["member_ids"][: args.members]:
            arc_dir = Path(r["dir"]) / "members" / m / "arc"
            if not (arc_dir / "ep000" / "trace.jsonl").is_file() or not (arc_dir / "arc.json").is_file():
                continue
            for k in range(args.replicates + 1):
                dest = args.out / r["key"] / m / f"r{k}.json"
                if not dest.is_file():
                    jobs.append((r["key"], m, k, arc_dir, rows[m], patient, dest))
    print(f"{len(jobs)} situations to write", flush=True)

    def one(job):
        key_, m, k, arc_dir, row, patient, dest = job
        arc = json.loads((arc_dir / "arc.json").read_text())
        shape, ep1 = arc["shape"], arc["episodes"][1]
        arc_id = arc_id_of(row, arc["arc_seed"], cfg.fingerprint)
        assert arc_id == arc["arc_id"], f"{key_}/{m}: arc id {arc_id} != {arc['arc_id']}"
        _, record = read_trace(arc_dir / "ep000" / "trace.jsonl")
        prev = situation_from_file(arc_dir / "ep000" / "situation.yaml")
        anchor = Anchor(shape["anchor_weekday"], shape["anchor_hour"])
        s = evolve_situation(cfg, row, prev, record, shape["gaps_hours"][0], patient, arc_seed=arc["arc_seed"],
                             arc_id=arc_id if k == 0 else f"{arc_id}#resample{k}", episode_id=1,
                             now_line=f"It is now {describe_ts(anchor, ep1['opened_ts'])}.", beat=shape["beats"][0])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"run": key_, "member": m, "replicate": k, "beat": shape["beats"][0],
                                    "situation": s.model_dump()}, ensure_ascii=False, indent=1) + "\n")
        return key_, m, k

    done = failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
            except Exception as e:  # noqa: BLE001 — reported; a rerun retries it
                failed += 1
                print(f"failed: {str(e)[:200]}", flush=True)
            if (done + failed) % 50 == 0:
                print(f"[{done + failed}/{len(jobs)}] {failed} failed", flush=True)
    print(f"done: {done} written, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
