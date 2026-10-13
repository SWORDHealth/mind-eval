"""Is GPT-6.1 Sol over-flagging memory? Paper runs (every judge on the same replies), against the labelers.

    python analysis/memory_flags.py --runs_root <dir> --labels <c3_labels dir> --out <dir> [--exclude <leaks.jsonl>]

  rates      per judge, the share of replies flagged for a memory issue (missed shared context, false memory), in
             session 1 and in sessions 2-3.
  what       Sol's justification of each reply it flags for memory, sorted by what it cites: an earlier session or
             the notes, or something said earlier in the same session (keyword rules; the rest is unclear).
  labelers   sessions 2-3: of the replies a judge flags for memory, the share at a turn where the validated labelers
             found a memory failure (the reply a member answers by repeating a recorded fact, or a relevant note
             missed), against the share among the replies it does not flag (lift); and of the repeat-forcing replies,
             the share each judge flags for memory (recall).
  others     how each other judge rates the replies Sol flags for memory: Major or worse, and the issues it flags.
  examples   replies Sol flags for missed shared context, at Major or worse, that the labelers confirm (the member
             repeats a recorded fact in answer) and MindJudge-27B does not flag for memory: the note, the member's turn
             before, the reply, the member's answer, and both judges' justifications.
Writes memory_flags.csv, report.md, examples.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXCLUDE, ISSUES, judge_dir, load_verdicts, mean, registry, runs_root, set_exclusions  # noqa: E402
from dose_response import load_labels  # noqa: E402

from mindeval.arc import parse_log  # noqa: E402

JUDGES = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview"}
MEMORY = ("missed_shared_context", "false_memory")
CROSS = re.compile(r"\b(last|previous|prior|past|earlier|first) (session|sessions|time|week|conversation|check-?in|chat)"
                   r"|\bsession [123]\b|\b(notes?|memory|log|history)\b|\bfrom before\b|\bpreviously (shared|discussed|"
                   r"established|mentioned|reported)|\bfollow(ing)?[- ]up on\b|\blast (week|time)\b|\bbetween sessions\b"
                   r"|\bsince (we|they|last)\b", re.I)
WITHIN = re.compile(r"\b(earlier|already|just|again)\b|\brepeat(s|ing|ed)?\b|\balready (said|stated|explained|"
                    r"described|established|asked|answered|covered|shared|told)", re.I)


def memory_flag(v: dict) -> bool:
    return any(v.get(i) == 1 for i in MEMORY)


def cites(text: str) -> str:
    if CROSS.search(text or ""):
        return "an earlier session or the notes"
    if WITHIN.search(text or ""):
        return "earlier in the same session"
    return "unclear"


def session_turns(trace: Path) -> dict[int, dict]:
    return {r["turn"]: r for r in map(json.loads, filter(None, trace.read_text(encoding="utf-8").split("\n")))
            if r.get("type") == "turn"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    cont, side = load_labels(args.labels / "continuity_v1"), load_labels(args.labels / "member_side_v1")
    args.out.mkdir(parents=True, exist_ok=True)
    V = {}  # judge -> run key -> verdicts
    for j in JUDGES:
        V[j] = {}
        for r in runs:
            p = judge_dir(root, r, j, judges) / "verdicts.jsonl"
            if p.is_file():
                drop = EXCLUDE.get(r["key"], set())
                V[j][r["key"]] = {k: v for k, v in load_verdicts(p).items() if k not in drop}
    rows, lines = [], ["# Is GPT-6.1 Sol over-flagging memory? (paper runs)", ""]

    # rates
    lines += ["## Share of replies flagged for a memory issue", "", "| judge | session 1 | sessions 2-3 |", "|---|---|---|"]
    for j, jl in JUDGES.items():
        s1 = [memory_flag(v) for vs in V[j].values() for v in vs.values() if v["episode"] == "ep000"]
        s23 = [memory_flag(v) for vs in V[j].values() for v in vs.values() if v["episode"] != "ep000"]
        if s1:
            lines.append(f"| {jl} | {100 * mean(s1):.1f}% | {100 * mean(s23):.1f}% |")
            rows += [{"analysis": "rate", "judge": jl, "measure": "session 1", "value": mean(s1)},
                     {"analysis": "rate", "judge": jl, "measure": "sessions 2-3", "value": mean(s23)}]
    lines.append("")

    # what Sol's memory flags cite
    lines += ["## What GPT-6.1 Sol's memory flags cite (its justification, keyword rules)", "",
              "| sessions | flags | an earlier session or the notes | earlier in the same session | unclear |",
              "|---|---|---|---|---|"]
    for lab, pick in (("1", lambda e: e == "ep000"), ("2-3", lambda e: e != "ep000")):
        cnt = Counter(cites(v.get("justification", "")) for vs in V["gpt61_sol"].values() for v in vs.values()
                      if pick(v["episode"]) and memory_flag(v))
        n = sum(cnt.values())
        lines.append(f"| {lab} | {n} | " + " | ".join(f"{100 * cnt[c] / n:.0f}%" for c in
                                                       ("an earlier session or the notes", "earlier in the same session",
                                                        "unclear")) + " |")
        for c, k in cnt.items():
            rows.append({"analysis": "cites", "judge": "GPT-6.1 Sol", "measure": f"sessions {lab}: {c}", "value": k / n})
    lines.append("")

    # against the labelers (sessions 2-3)
    fail_turns: dict = {}
    for (name, m, e), p in side.items():
        fail_turns.setdefault((name, m, e), {"repeat": set(), "missed": set()})["repeat"] |= {
            x["turn"] for x in p.get("repeats") or [] if isinstance(x.get("turn"), int)}
    for (name, m, e), p in cont.items():
        fail_turns.setdefault((name, m, e), {"repeat": set(), "missed": set()})["missed"] |= {
            o["turn"] for o in p.get("opportunities") or [] if o.get("status") == "missed" and isinstance(o.get("turn"),
                                                                                                         int)}
    lines += ["## Against the labelers' memory failures (sessions 2-3, labelled sessions)", "",
              "| judge | flagged replies at a labelled failure | unflagged replies at one | lift | repeat-forcing "
              "replies it flags for memory |", "|---|---|---|---|---|"]
    for j, jl in JUDGES.items():
        hit = {True: [], False: []}
        rec = []
        for key, vs in V[j].items():
            for v in vs.values():
                if v["episode"] == "ep000" or (key, v["member"], v["episode"]) not in fail_turns:
                    continue
                ft = fail_turns[key, v["member"], v["episode"]]
                at = v["turn"] in ft["repeat"] or v["turn"] in ft["missed"]
                hit[memory_flag(v)].append(at)
                if v["turn"] in ft["repeat"]:
                    rec.append(memory_flag(v))
        if hit[True] and hit[False]:
            a, b = mean(hit[True]), mean(hit[False])
            lines.append(f"| {jl} | {100 * a:.0f}% ({len(hit[True])}) | {100 * b:.0f}% | {a / b:.1f}x | "
                         f"{100 * mean(rec):.0f}% ({len(rec)}) |")
            rows += [{"analysis": "labelers", "judge": jl, "measure": "precision", "value": a},
                     {"analysis": "labelers", "judge": jl, "measure": "base rate", "value": b},
                     {"analysis": "labelers", "judge": jl, "measure": "recall of repeat-forcing", "value": mean(rec)}]
        elif hit[False]:
            lines.append(f"| {jl} | - (no flags) | {100 * mean(hit[False]):.0f}% | - | {100 * mean(rec):.0f}% |")
    lines.append("")

    # how the others rate Sol's memory flags
    lines += ["## The replies GPT-6.1 Sol flags for memory (sessions 2-3), as the other judges rate them", "",
              "| judge | replies | Major or worse | flags memory | issues it flags instead (top 3) | Major or worse on "
              "the other replies |", "|---|---|---|---|---|---|"]
    for j in ("own", "deepseek_v4", "hy4"):
        n = maj = mem = 0
        iss, base = Counter(), []
        for key, vs in V["gpt61_sol"].items():
            other = V[j].get(key, {})
            for k, v in vs.items():
                if v["episode"] == "ep000" or k not in other:
                    continue
                o = other[k]
                if memory_flag(v):
                    n += 1
                    maj += o["severity"] in ("Major", "Critical")
                    mem += memory_flag(o)
                    iss.update(i for i in ISSUES if o.get(i) == 1 and i not in MEMORY)
                else:
                    base.append(o["severity"] in ("Major", "Critical"))
        if n:
            top = ", ".join(f"{i.replace('_', ' ')} {100 * c / n:.0f}%" for i, c in iss.most_common(3))
            lines.append(f"| {JUDGES[j]} | {n} | {100 * maj / n:.0f}% | {100 * mem / n:.0f}% | {top} | "
                         f"{100 * mean(base):.0f}% |")
            rows += [{"analysis": "others", "judge": JUDGES[j], "measure": "major share", "value": maj / n},
                     {"analysis": "others", "judge": JUDGES[j], "measure": "memory flag share", "value": mem / n},
                     {"analysis": "others", "judge": JUDGES[j], "measure": "major share, other replies",
                      "value": mean(base)}]
    lines.append("")

    # examples
    cands = []
    for key, vs in V["gpt61_sol"].items():
        mj = V["own"].get(key, {})
        for k, v in vs.items():
            e = v["episode"]
            if e == "ep000" or v.get("missed_shared_context") != 1 or v["severity"] not in ("Major", "Critical"):
                continue
            ft = fail_turns.get((key, v["member"], e))
            if not ft or v["turn"] not in ft["repeat"] or k not in mj or memory_flag(mj[k]):
                continue
            if cites(v.get("justification", "")) != "an earlier session or the notes":
                continue
            cands.append((key, k, v, mj[k]))
    random.Random(args.seed).shuffle(cands)
    ex = [f"# GPT-6.1 Sol flags memory, MindJudge-27B does not ({len(cands)} confirmed cases; {args.examples} drawn "
          f"at random, seed {args.seed})", "",
          "Each: a reply in session 2 or 3 that Sol flags for missed shared context at Major or worse, citing an "
          "earlier session or the notes; the member then repeats a recorded fact (labeler, validated); MindJudge-27B "
          "flags no memory issue on it.", ""]
    run_dir = {r["key"]: Path(r["dir"]) for r in runs}
    label = {r["key"]: r["label"] for r in runs}
    for key, k, v, o in cands[: args.examples]:
        m, e, t = v["member"], v["episode"], v["turn"]
        arc = run_dir[key] / "members" / m / "arc"
        turns = session_turns(arc / e / "trace.jsonl")
        notes = parse_log((arc / f"ep{int(e[2:]) - 1:03d}" / "log.md").read_text())
        rep = [x for x in side[key, m, e].get("repeats") or [] if x.get("turn") == t]
        note_ids = [x.get("note") for x in rep]
        prev = turns.get(t - 1, {}).get("utterance") or ""
        ex += [f"## {label[key]}, {m}, session {int(e[2:]) + 1}, reply {t}", "",
               "**The note** (from the memory after the session before): "
               + "; ".join(f"{nid}. {notes[int(nid[1:]) - 1]}" for nid in note_ids
                           if nid and nid[1:].isdigit() and int(nid[1:]) <= len(notes)), "",
               f"**Member, turn before:** {prev}", "",
               f"**Counselor reply:** {turns.get(t, {}).get('counselor_utterance', '')}", "",
               f"**Member answers:** {turns.get(t, {}).get('utterance', '')}", "",
               f"**GPT-6.1 Sol ({v['severity']}; " + ", ".join(i.replace("_", " ") for i in ISSUES if v.get(i) == 1)
               + f"):** {v.get('justification', '')}", "",
               f"**MindJudge-27B ({o['severity']}; " + (", ".join(i.replace("_", " ") for i in ISSUES if o.get(i) == 1)
                                                        or "no issue") + f"):** {o.get('justification', '')}", ""]
    with (args.out / "memory_flags.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["analysis", "judge", "measure", "value"])
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "examples.md").write_text("\n".join(ex) + "\n")
    print("\n".join(lines))
    print(f"\n{len(cands)} confirmed example candidates; examples.md written")


if __name__ == "__main__":
    main()
