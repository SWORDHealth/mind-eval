"""The memory ablation: each memory mode against the compacted rerun, paired within member.

    python analysis/ablation.py --root <memory_ablation dir> --batch main6 --out <dir>

Runs: <root>/<batch>/<counselor>/<mode>/ (made by fork_run.py; session 1 is the source run's, copied). Verdicts:
<root>/judging/<judge>/<batch>/<counselor>/<mode>/<view>/judge/mindeval2/verdicts.jsonl, for the views of
judge_views.py (as_sent: what the counselor saw; reference: the benchmark's memory in every mode). Labels, when
given: c3_label.py output directories (continuity_v1 and member_side_v1) over the same runs, named
<counselor>/<mode>.

Every comparison is a condition minus compacted, per member, over the sessions both finished (2 onward), then
averaged over members; 95% intervals resample members. The compacted rerun against the source run (sessions 2-3,
which the source run also has) is the noise floor: the same condition, resampled. n (members, sessions) is printed
beside every cell.

Reported per counselor, judge and view: MQM (session severity) and the memory-related issue rates
(missed_shared_context, false_memory) and Major+ share; per counselor: the counselor's prompt and reply tokens per
session (from the traces), continuity use per opportunity and re-asks, and the member's corrections and repeats.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import BOOT, WEIGHT, ci, git_describe, load_verdicts, mean, member_draws
from common import ISSUES as ALL_ISSUES

MODES = ("none", "full", "split")
ISSUES = ("missed_shared_context", "false_memory")


def session_scores(verdicts: dict) -> dict[tuple[str, str], dict]:
    acc: dict = defaultdict(lambda: {"w": [], "major": 0, **{k: 0 for k in ISSUES}})
    for v in verdicts.values():
        s = acc[v["member"], v["episode"]]
        s["w"].append(WEIGHT[v["severity"]])
        s["major"] += v["severity"] in ("Major", "Critical")
        for k in ISSUES:
            s[k] += v.get(k) == 1
    return {k: {"mqm": sum(s["w"]) / len(s["w"]), "major_plus": s["major"] / len(s["w"]),
                **{i: s[i] / len(s["w"]) for i in ISSUES}, "n": len(s["w"])} for k, s in acc.items()}


def issue_rates(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(member, episode) -> share of the session's replies carrying each issue flag, sessions 2 onward."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for v in load_verdicts(path).values():
        if v["episode"] != "ep000":
            for i in ALL_ISSUES:
                acc[v["member"], v["episode"]][i].append(v.get(i) == 1)
    return {k: {i: mean(x) for i, x in d.items()} for k, d in acc.items()}


def traces(run: Path) -> dict[tuple[str, str], dict]:
    """(member, episode) -> counselor tokens and turns, finished sessions only."""
    out = {}
    for t in run.glob("members/*/arc/ep*/trace.jsonl"):
        rows = [json.loads(x) for x in t.read_text().split("\n") if x.strip()]
        if not rows or rows[-1].get("type") != "session" or rows[-1].get("termination") == "error":
            continue
        usage = [r["meta"]["counselor"].get("usage") or {} for r in rows[:-1] if (r.get("meta") or {}).get("counselor")]
        out[t.parents[2].name, t.parent.name] = {
            "counselor_prompt_tokens_first": usage[0].get("prompt_tokens") if usage else None,
            "counselor_completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
            "replies": len(usage), "system_prompt_chars": len(rows[-1]["counselor_system_prompt"]),
            "member_prompt_chars": len(rows[-1]["member_system_prompt"]),
            "left": rows[-1]["termination"] == "member_terminated"}
    return out


def labels(path: Path) -> dict[tuple[str, str], dict]:
    """(member, episode) -> parsed labels, from a c3_label.py output directory."""
    if not (path / "labels.jsonl").is_file():
        return {}
    keys = {k["id"]: k for k in map(json.loads, (path / "keys.jsonl").read_text().splitlines())}
    out = {}
    for line in (path / "labels.jsonl").read_text().splitlines():
        x = json.loads(line)
        if "parsed" in x:
            k = keys[x["id"]]
            out[k["member"], k["episode"]] = x["parsed"]
    return out


STOP = set("the a an and or but to of in on at for with is was were be been it its this that i you he she they we my your "
           "her his their me him them not no do did does have has had so if as about from by up out what how why when "
           "who which there then just like get got going know think feel said say says member counsellor dont im "
           "thats".split())
WORD = re.compile(r"[a-z']+")


def _words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) > 3 and w not in STOP}


def beyond_restatement(run: Path, member: str, ep: str, parsed: dict) -> int:
    """How many of the labeler's `used` codes the member's own earlier words this session do not explain: the
    content words the counselor's reply shares with the note are mostly (>= 70%) absent from the member's turns
    before it. The labeler counts building on a restated fact as use; this keeps only use that needed the memory."""
    trace = run / "members" / member / "arc" / ep / "trace.jsonl"
    log_path = run / "members" / member / "arc" / f"ep{int(ep[2:]) - 1:03d}" / "log.md"
    if not trace.is_file() or not log_path.is_file():
        return 0
    turns = {x["turn"]: x for x in map(json.loads, trace.read_text().split("\n")[:-1]) if x.get("type") == "turn"}
    log = log_path.read_text().splitlines()
    n = 0
    for o in parsed["opportunities"]:
        if o["status"] != "used" or not o.get("turn"):
            continue
        k = int(o["note"][1:]) - 1
        shared = _words(log[k]) & _words((turns.get(o["turn"]) or {}).get("counselor_utterance") or "") \
            if k < len(log) else set()
        before = set().union(*([_words(turns[i].get("utterance") or "") for i in turns if i < o["turn"]] or [set()]))
        if shared and len(shared & before) / len(shared) < 0.7:
            n += 1
    return n


def continuity_measures(p: dict) -> dict:
    ops = p["opportunities"]
    st = defaultdict(int)
    for o in ops:
        st[o["status"]] += 1
    n = len(ops)
    return {"opportunities": n, "used_share": st["used"] / n if n else None, "missed_share": st["missed"] / n if n else None,
            "contradicted": st["contradicted"], "reasks": len(p["reasks"])}


def paired(a: dict, b: dict, field: str, members: list[str], eps: list[str] | None = None) -> dict:
    """mean over members of (mean over shared sessions of a - b), with a member bootstrap."""
    per = {}
    for m in members:
        diffs = [a[m, e][field] - b[m, e][field] for e in (eps or sorted({e for (mm, e) in a if mm == m}))
                 if (m, e) in a and (m, e) in b and e != "ep000"
                 and a[m, e].get(field) is not None and b[m, e].get(field) is not None]
        if diffs:
            per[m] = sum(diffs) / len(diffs)
    if not per:
        return {"diff": None, "n_members": 0}
    ms = sorted(per)
    draws = [mean(per[m] for m in d) for d in member_draws(ms, BOOT)]
    lo, hi = ci(draws)
    n_sessions = sum(1 for m in ms for e in {e for (mm, e) in a if mm == m}
                     if (m, e) in b and e != "ep000")
    return {"diff": mean(per.values()), "lo": lo, "hi": hi, "n_members": len(ms), "n_sessions": n_sessions}


def level(a: dict, field: str) -> float | None:
    xs = [v[field] for (m, e), v in a.items() if e != "ep000" and v.get(field) is not None]
    return mean(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--batch", default="main6")
    ap.add_argument("--judges", default="mindjudge,deepseek_v4")
    ap.add_argument("--labels", type=Path, help="directory holding <prompt>/<counselor>/<mode>/ label outputs")
    ap.add_argument("--sources", default="glm53=/mnt/data/jmendonca/MindSim-runs/mindeval2_preserve/glm53,"
                                        "gemma4_12b=/mnt/data/jmendonca/MindSim-runs/mindeval_qwen/gemma4_12b")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    sources = dict(x.split("=", 1) for x in args.sources.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], [f"# Memory ablation ({args.batch})", "",
                       "Each cell: condition minus compacted, mean over members of the per-member mean over sessions "
                       "2 onward; [95% CI over members]; (members, sessions).", ""]
    batch = args.root / args.batch
    for cdir in sorted(p for p in batch.iterdir() if p.is_dir()):
        counselor = cdir.name
        runs = {m.name: m for m in cdir.iterdir() if (m / "run.json").is_file()}
        if "compacted" not in runs:
            continue
        members = json.loads((runs["compacted"] / "run.json").read_text())["member_ids"]
        lines += [f"## {counselor}", ""]
        # Judges and views.
        for judge in args.judges.split(","):
            base_dir = args.root / "judging" / judge / args.batch / counselor
            ref = base_dir / "compacted" / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"
            if not ref.is_file():
                continue
            comp = session_scores(load_verdicts(ref))
            lines += [f"### Judge: {judge}", "", "| condition | view | MQM | " + " | ".join(ISSUES) + " | Major+ |",
                      "|---|---|---|---|---|---|"]
            lines.append(f"| compacted (level) | - | {level(comp, 'mqm'):.3f} | "
                         + " | ".join(f"{100 * level(comp, i):.1f}%" for i in ISSUES)
                         + f" | {100 * level(comp, 'major_plus'):.1f}% |")
            for mode in MODES:
                for view in ("as_sent", "reference"):
                    vp = base_dir / mode / view / "judge" / "mindeval2" / "verdicts.jsonl"
                    if not vp.is_file():
                        continue
                    cond = session_scores(load_verdicts(vp))
                    cells = []
                    for field in ("mqm", *ISSUES, "major_plus"):
                        r = paired(cond, comp, field, members)
                        rows.append({"counselor": counselor, "judge": judge, "mode": mode, "view": view,
                                     "field": field, **r})
                        if r["diff"] is None:
                            cells.append("-")
                        else:
                            scale = 1 if field == "mqm" else 100
                            cells.append(f"{scale * r['diff']:+.2f} [{scale * r['lo']:+.2f}, {scale * r['hi']:+.2f}]"
                                         f" ({r['n_members']}, {r['n_sessions']})")
                    lines.append(f"| {mode} | {view} | " + " | ".join(cells) + " |")
            # Per session index: does the gap grow over the horizon?
            eps = sorted({e for (m, e) in comp if e != "ep000"})
            lines += ["", f"MQM by session ({judge}; condition minus compacted, [95% CI]):", "",
                      "| condition | view | " + " | ".join(f"s{int(e[2:]) + 1}" for e in eps) + " |",
                      "|---|---|" + "---|" * len(eps)]
            for mode in MODES:
                for view in ("as_sent", "reference"):
                    vp = base_dir / mode / view / "judge" / "mindeval2" / "verdicts.jsonl"
                    if not vp.is_file():
                        continue
                    cond = session_scores(load_verdicts(vp))
                    cells = []
                    for e in eps:
                        r = paired(cond, comp, "mqm", members, [e])
                        rows.append({"counselor": counselor, "judge": judge, "mode": mode, "view": view,
                                     "field": f"mqm_{e}", **r})
                        cells.append("-" if r["diff"] is None else f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
                    lines.append(f"| {mode} | {view} | " + " | ".join(cells) + " |")
            # Which issues move (pooled share of replies, condition minus compacted, paired by session).
            lines += ["", f"Issues that move most ({judge}, points of replies flagged):", ""]
            base_rates = issue_rates(ref)
            for mode in MODES:
                for view in ("as_sent", "reference"):
                    vp = base_dir / mode / view / "judge" / "mindeval2" / "verdicts.jsonl"
                    if not vp.is_file():
                        continue
                    cr = issue_rates(vp)
                    keys = set(cr) & set(base_rates)
                    d = {i: 100 * mean(cr[k][i] - base_rates[k][i] for k in keys) for i in ALL_ISSUES} if keys else {}
                    top = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:4]
                    lines.append(f"- {mode}, {view}: " + ", ".join(f"{i} {x:+.1f}" for i, x in top))
            lines.append("")
        # Tokens and lengths from the traces.
        tr = {mode: traces(run) for mode, run in runs.items()}
        lines += ["### Counselor input and output, per session (mean over sessions 2 onward)", "",
                  "| condition | system prompt chars | first-reply prompt tokens | reply tokens / session | "
                  "replies / session | member left (%) | member prompt chars |", "|---|---|---|---|---|---|---|"]
        for mode in ("compacted", *MODES):
            if mode not in tr:
                continue
            t = tr[mode]
            lines.append(f"| {mode} | {level(t, 'system_prompt_chars'):.0f} | {level(t, 'counselor_prompt_tokens_first') or 0:.0f} | "
                         f"{level(t, 'counselor_completion_tokens'):.0f} | {level(t, 'replies'):.1f} | "
                         f"{100 * mean(float(v['left']) for (m, e), v in t.items() if e != 'ep000'):.0f} | "
                         f"{level(t, 'member_prompt_chars'):.0f} |")
        by_ep = defaultdict(list)
        for (m, e), v in tr.get("full", {}).items():
            if v["counselor_prompt_tokens_first"]:
                by_ep[e].append(v["counselor_prompt_tokens_first"])
        if by_ep:
            lines += ["", "Full memory, the counselor's first-reply prompt tokens by session: "
                      + ", ".join(f"s{int(e[2:]) + 1} {mean(x):.0f} (max {max(x)})" for e, x in sorted(by_ep.items()))]
        # Labels.
        if args.labels:
            cont = {}
            for mode in ("compacted", *MODES):
                cont[mode] = {}
                for k, p in labels(args.labels / "continuity_v1" / counselor / mode).items():
                    c = continuity_measures(p)
                    n_ops = c["opportunities"]
                    c["memory_use_share"] = (beyond_restatement(runs[mode], k[0], k[1], p) / n_ops) if n_ops else None
                    cont[mode][k] = c
            # Re-asks and contradictions from the focused labeler (validated on its own), where it has coded.
            for mode in ("compacted", *MODES):
                for k, f in labels(args.labels / "reask_contradict_v1" / counselor / mode).items():
                    if k in cont[mode]:
                        cont[mode][k].update({"reasks": len(f["reasks"]), "contradicted": len(f["contradictions"])})
            side = {mode: {k: {"corrections": len(p["corrections"]), "repeats": len(p["repeats"])}
                           for k, p in labels(args.labels / "member_side_v1" / counselor / mode).items()}
                    for mode in ("compacted", *MODES)}
            lines += ["", "### Continuity (labeler) and the member's side", "",
                      "| condition | used / opportunity | used beyond restatement / opportunity | missed / opportunity "
                      "| contradictions / session | re-asks / session | member corrections / session | member repeats / "
                      "session |", "|---|---|---|---|---|---|---|---|"]
            for mode in ("compacted", *MODES):
                c, s = cont.get(mode, {}), side.get(mode, {})
                if not c and not s:
                    continue
                vals = [level(c, f) if c else None
                        for f in ("used_share", "memory_use_share", "missed_share", "contradicted", "reasks")]
                vals += [level(s, f) if s else None for f in ("corrections", "repeats")]
                lines.append(f"| {mode} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")
                if mode != "compacted":
                    for f, src in (("used_share", c), ("memory_use_share", c), ("missed_share", c),
                                   ("contradicted", c), ("reasks", c)):
                        if src and cont["compacted"]:
                            rows.append({"counselor": counselor, "judge": "labeler", "mode": mode, "view": "-",
                                         "field": f, **paired(src, cont["compacted"], f, members)})
                    for f in ("corrections", "repeats"):
                        if s and side["compacted"]:
                            rows.append({"counselor": counselor, "judge": "labeler", "mode": mode, "view": "-",
                                         "field": f, **paired(s, side["compacted"], f, members)})
            lines += ["", "Paired against compacted (condition minus compacted, [95% CI], (members, sessions)):", ""]
            for r in rows:
                if r["counselor"] == counselor and r["judge"] == "labeler" and r.get("diff") is not None:
                    lines.append(f"- {r['mode']}, {r['field']}: {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] "
                                 f"({r['n_members']}, {r['n_sessions']})")
        # Noise floor: the compacted rerun against its source, sessions 2-3, under each judge's verdicts of the source.
        lines += [""]
        src = Path(sources.get(counselor, ""))
        for judge in args.judges.split(","):
            rerun = args.root / "judging" / judge / args.batch / counselor / "compacted" / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"
            own = {"mindjudge": [src / "judge" / "mindeval2" / "verdicts.jsonl", src / "judge" / "verdicts.jsonl"],
                   "deepseek_v4": [Path(f"/mnt/data/jmendonca/MindSim-runs/judging/deepseek_v4/{counselor}/judge/mindeval2/verdicts.jsonl"),
                                   args.root / "judging" / "deepseek_v4" / "source" / counselor / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"]}
            src_path = next((p for p in own.get(judge, []) if p.is_file()), None)
            if src_path is None or not rerun.is_file():
                continue
            a, b = session_scores(load_verdicts(rerun)), session_scores(load_verdicts(src_path))
            r = paired(a, b, "mqm", members, ["ep001", "ep002"])
            rows.append({"counselor": counselor, "judge": judge, "mode": "compacted_rerun_vs_source", "view": "as_sent",
                         "field": "mqm", **r})
            if r["diff"] is not None:
                lines.append(f"- Noise floor ({judge}): compacted rerun minus source run, MQM sessions 2-3: "
                             f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] ({r['n_members']}, {r['n_sessions']})")
        lines.append("")
    import csv
    if rows:
        with (args.out / "ablation.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({
        "script": "analysis/ablation.py", "analysis_git": git_describe(), "batch": args.batch,
        "runs": {f"{c.name}/{m.name}": json.loads((m / "run.json").read_text()) | json.loads((m / "fork.json").read_text())
                 for c in batch.iterdir() if c.is_dir() for m in c.iterdir() if (m / "fork.json").is_file()},
        "sources": sources}, indent=2, default=str) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
