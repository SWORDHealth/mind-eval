"""Counselor replies that run on into a turn of their own: the reply writes the member's next message (a line opening
with a chat role, "user ...", sometimes followed by "assistant ..."), seen in Claude Opus 5 on Vertex.

    python analysis/artifacts.py --runs_root <dir> [--ablation <memory_ablation dir>] --out <dir>

Writes leaks.jsonl (one line per leaked reply: run, member, episode, turn and the verdict key "m/ep/tN"), and a
report of leak rates per run and session; when GPT-6.1 Sol has judged a run, it also compares the text check with
Sol's own notes (a verdict whose justification says the reply fabricated or invented a member reply). The other
analyses drop these replies from session scores with --exclude <leaks.jsonl>.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import judge_dir, registry, runs_root

#: A line that opens with a lowercase chat role, as a model starts the next turn: "user ...", "useryeah",
#: "assistantThat's" (the marker is often glued to the text), or "Human:" / "User:" with a colon.
ROLE_LINE = re.compile(r"(?:^|\n)\s*(?:(?:user|assistant|assisted)(?=[\sA-Za-z:])|(?:Human|User|Assistant)\s*:)")
TEMPLATE = re.compile(r"<\|(?:user|assistant|im_start|im_end)\|?>", re.I)
SOL_NOTE = re.compile(r"fabricat|invent(?:s|ed|ing)? (?:a )?member|member reply|speaks? for the member|speaking for the member",
                      re.I)


def leaked(text: str) -> bool:
    """Whether a counselor reply runs on into another turn. The first line is left out (a reply may begin "Member,")."""
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return bool(ROLE_LINE.search("\n" + body) or TEMPLATE.search(text))


def replies(run: Path):
    for trace in sorted(run.glob("members/*/arc/ep*/trace.jsonl")):
        member, episode = trace.parents[2].name, trace.parent.name
        rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
        for r in rows:
            u = r.get("counselor_utterance")
            if isinstance(u, str) and r.get("turn") is not None:
                yield member, episode, r["turn"], u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--ablation", type=Path, help="also scan the ablation's runs (<dir>/main6/<counselor>/<mode>/)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    targets = [(r["key"], r["dir"], r) for r in runs]
    if args.ablation:
        targets += [(f"ablation/{p.parent.name}/{p.name}", p, None)
                    for p in sorted((args.ablation / "main6").glob("*/*")) if (p / "run.json").is_file()]
    args.out.mkdir(parents=True, exist_ok=True)
    out, lines = [], ["# Replies that run on into another turn", "",
                      "| run | replies | leaked, s1 | s2 | s3 | later sessions | Sol notes a fabricated turn | both |",
                      "|---|---|---|---|---|---|---|---|"]
    for name, run, reg in targets:
        n, k = Counter(), Counter()
        found = set()
        for m, e, t, u in replies(run):
            n[e] += 1
            if leaked(u):
                k[e] += 1
                found.add((m, e, t))
                out.append({"run": name, "member": m, "episode": e, "turn": t, "key": f"{m}/{e}/t{t}"})
        sol = set()
        if reg is not None:
            p = judge_dir(root, reg, "gpt61_sol", judges) / "verdicts.jsonl"
            if p.is_file():
                for line in p.read_text().splitlines():
                    v = json.loads(line)
                    if SOL_NOTE.search(v.get("justification") or ""):
                        sol.add((v["member"], v["episode"], v["turn"]))
        if not sum(k.values()) and not sol:
            continue
        later = sum(k[e] for e in n if e != "ep000")
        rate = lambda e: f"{k[e]} ({100 * k[e] / n[e]:.1f}%)" if n[e] else "-"
        lines.append(f"| {name} | {sum(n.values())} | {rate('ep000')} | {rate('ep001')} | {rate('ep002')} | {later} | "
                     f"{len(sol) if reg is not None else '-'} | {len(found & sol) if reg is not None else '-'} |")
    (args.out / "leaks.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
