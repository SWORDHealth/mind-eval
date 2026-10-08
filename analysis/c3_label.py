"""C3 labelers: code sessions 2 onward with a frozen prompt from `prompts/`:
`continuity_v1` (the counselor: continuity use per opportunity, and re-asking) or `member_side_v1` (the member:
corrections of the counselor, and repeats of what was already told).

    python analysis/c3_label.py --out <dir> [--prompt continuity_v1] [--model ... --api_base <url>] name=<run dir> ...

For every finished session after the first, the labeler gets the shared log written after the session before
(the benchmark's memory, which every memory mode keeps, numbered M1, M2, ...) and the session's transcript, with
the trace's turn numbers so codes line up with the judge's per-reply verdicts. It sees an opaque session id only:
no counselor, condition or run name. `<out>/keys.jsonl` maps ids back; `<out>/labels.jsonl` holds every answer
(raw and parsed) and is appended to, so a rerun resumes. The prompt's sha256 is recorded on each label and a
label made with another prompt or model is not reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindeval import llm  # noqa: E402
from mindeval.arc import parse_log  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "prompts"
STATUSES = ("used", "missed", "contradicted")


def opaque(name: str, member: str, ep: str) -> str:
    return "S" + hashlib.sha256(f"{name}\x00{member}\x00{ep}".encode()).hexdigest()[:12]


def session_input(trace: Path) -> dict | None:
    """The labeler's two blocks for one finished session after the first, or None."""
    lines = [json.loads(x) for x in trace.read_text(encoding="utf-8").split("\n") if x.strip()]
    if not lines or lines[-1].get("type") != "session" or lines[-1].get("termination") == "error":
        return None
    ep = int(trace.parent.name[2:])
    if ep == 0:
        return None
    log = parse_log((trace.parents[1] / f"ep{ep - 1:03d}" / "log.md").read_text())
    memory = "\n".join(f"M{i + 1}. {e}" for i, e in enumerate(log))
    rows = []
    for t in lines[:-1]:
        if t.get("counselor_utterance"):
            rows.append(f"[{t['turn']}] Counsellor: {t['counselor_utterance']}")
        if t.get("utterance"):
            rows.append(f"[{t['turn']}] Member: {t['utterance']}")
    return {"memory": memory, "transcript": "\n\n".join(rows), "n_notes": len(log),
            "counselor_turns": [t["turn"] for t in lines[:-1] if t.get("counselor_utterance")],
            "member_turns": [t["turn"] for t in lines[:-1] if t.get("utterance")]}


def _answer(text: str, first_key: str) -> dict:
    body = text.split("</think>")[-1]
    decoder, obj, start = json.JSONDecoder(), None, body.rfind('{"' + first_key + '"')
    if start < 0:
        start = body.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(body[start:])
            break
        except ValueError:
            start = body.find("{", start + 1)
    if not isinstance(obj, dict):
        raise ValueError("no answer object")
    return obj


def parse_member_side(text: str, inp: dict) -> dict:
    obj = _answer(text, "corrections")
    if "corrections" not in obj or "repeats" not in obj:
        raise ValueError("no answer object")
    notes = {f"M{i + 1}" for i in range(inp["n_notes"])}

    def clean(items):
        out, seen = [], set()
        for x in items:
            turn = x.get("turn")
            if turn not in inp["member_turns"] or turn in seen:
                continue
            seen.add(turn)
            note = x.get("note")
            out.append({"turn": turn, "note": note if note in notes else None, "quote": str(x.get("quote", ""))[:300]})
        return out

    return {"corrections": clean(obj["corrections"]), "repeats": clean(obj["repeats"])}


def parse(text: str, n_notes: int, turns: list[int]) -> dict:
    """The last JSON object in the answer, checked: notes exist, turns are counselor turns, statuses known."""
    obj = _answer(text, "opportunities")
    if "opportunities" not in obj or "reasks" not in obj:
        raise ValueError("no answer object")
    notes = {f"M{i + 1}" for i in range(n_notes)}
    ops, seen = [], set()
    for o in obj["opportunities"]:
        note, status = str(o.get("note", "")).strip(), str(o.get("status", "")).strip().lower()
        if note not in notes or status not in STATUSES or note in seen:
            continue
        seen.add(note)
        turn = o.get("turn")
        ops.append({"note": note, "status": status, "turn": turn if turn in turns else None,
                    "evidence": str(o.get("evidence", ""))[:300]})
    reasks = [{"note": str(r.get("note", "")), "turn": r.get("turn"), "question": str(r.get("question", ""))[:300]}
              for r in obj["reasks"] if r.get("turn") in turns]
    return {"opportunities": ops, "reasks": reasks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompt", default="continuity_v1", choices=("continuity_v1", "member_side_v1"))
    ap.add_argument("--model", default="hosted_vllm/nemotron-3.5-super-ea")
    ap.add_argument("--api_base", default="http://worker-11:8067/v1")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="label at most this many new sessions (0 = all)")
    ap.add_argument("--members", type=int, default=0, help="only the first N members of each run (0 = all)")
    ap.add_argument("--ids", type=Path, help="only these session ids (one per line), e.g. a validation sample")
    ap.add_argument("runs", nargs="+", help="name=<run dir>")
    args = ap.parse_args()
    template = (PROMPTS / f"{args.prompt}.txt").read_text()
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()[:16]
    args.out.mkdir(parents=True, exist_ok=True)
    keys_path, labels_path = args.out / "keys.jsonl", args.out / "labels.jsonl"
    keys = {}
    if keys_path.is_file():
        keys = {k["id"]: k for k in map(json.loads, keys_path.read_text().splitlines())}
    done = set()
    if labels_path.is_file():
        for line in labels_path.read_text().splitlines():
            x = json.loads(line)
            if "parsed" in x and x["prompt_sha"] == prompt_sha and x["model"] == args.model:
                done.add(x["id"])

    todo = []
    for spec in args.runs:
        name, run = spec.split("=", 1)
        run_dir = Path(run)
        member_ids = json.loads((run_dir / "run.json").read_text())["member_ids"]
        if args.members:
            member_ids = member_ids[:args.members]
        for member in member_ids:
            for trace in sorted((run_dir / "members" / member / "arc").glob("ep*/trace.jsonl")):
                ep = trace.parent.name
                if ep == "ep000":
                    continue
                sid = opaque(name, member, ep)
                if sid not in keys:
                    keys[sid] = {"id": sid, "name": name, "run": str(run_dir), "member": member, "episode": ep}
                if sid in done:
                    continue
                todo.append((sid, trace))
    keys_path.write_text("".join(json.dumps(k) + "\n" for k in keys.values()))
    if args.ids:
        wanted = set(args.ids.read_text().split())
        todo = [(sid, trace) for sid, trace in todo if sid in wanted]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} labelled, {len(todo)} to label with {args.model} (prompt {prompt_sha})", flush=True)

    lock = threading.Lock()
    counts = {"ok": 0, "error": 0}

    def one(sid: str, trace: Path) -> None:
        inp = session_input(trace)
        if inp is None:
            return
        prompt = template.replace("{memory}", inp["memory"]).replace("{transcript}", inp["transcript"])
        row = {"id": sid, "model": args.model, "prompt_sha": prompt_sha}
        last = None
        for attempt in range(3):
            try:
                text, meta = llm.call_messages([{"role": "user", "content": prompt}], args.model, args.api_base,
                                               args.temperature, None, 1800, params={}, api_key="EMPTY")
                parsed = (parse(text, inp["n_notes"], inp["counselor_turns"]) if args.prompt == "continuity_v1"
                          else parse_member_side(text, inp))
                row.update({"raw": text[-6000:], "parsed": parsed, "prompt": args.prompt, "n_notes": inp["n_notes"],
                            "usage": meta.get("usage")})
                last = None
                break
            except Exception as e:  # noqa: BLE001 — retried, then recorded
                last = f"{type(e).__name__}: {e}"[:500]
        if last:
            row["error"] = last
        with lock:
            with labels_path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["error" if last else "ok"] += 1
            n = counts["ok"] + counts["error"]
            if n % 25 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {counts['error']} errors", flush=True)

    llm._lib()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one, sid, trace) for sid, trace in todo]):
            f.result()


if __name__ == "__main__":
    main()
