"""Memory's effect under GPT-6.1 Sol: the ablation's without-memory sessions 2-3 against the paper's sessions 2-3.

    python analysis/sol_memory.py --ablation <memory_ablation dir> [--runs_root <dir>] --out <dir>

Without memory: main6/<c>/none, sessions 2-3, read by Sol through the reference view (the benchmark's memory in the
counselor prompt; judging/gpt61_sol/main6/<c>/none/reference/). With memory: the paper run's sessions 2-3 as Sol judged
them for the paper (runs.yaml). For the four free counselors the paper run is the ablation's source (same session 1
transcript); GLM-5.3, Sonnet 5 and Terra are compared with their paper rows (an earlier harness version, another
session 1 of the same member). Per member, the mean over sessions 2-3 of the condition minus the paper; per counselor
and mean over counselors, MQM and its parts (drivers.py) and its severity shares, 95% intervals resampling members.
The without-memory sessions were regenerated a month after the paper sessions on other servers, so the contrast carries
any regeneration shift too: the same contrast under MindJudge-27B is shown beside the clean one (without minus the
regenerated with-memory sessions, both made together), whose difference is that shift on MindJudge's scale; where Sol
also judged the regenerated with-memory sessions (judging/gpt61_sol/main6/<c>/compacted/as_sent/), the same three on
Sol's own scale.
Writes sol_memory.csv, report.md, manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import pool  # noqa: E402
from common import (
    BOOT,
    ISSUES,
    WEIGHT,
    git_describe,
    judge_dir,
    load_verdicts,
    mean,
    member_draws,
    registry,
    runs_root,  # noqa: E402
)
from dose_response import analyse, load_labels  # noqa: E402
from drivers import EXCLUSIVE, GROUPS, exclusive_parts, parts  # noqa: E402

#: ablation counselor -> its paper row in runs.yaml
PAPER = {"qwen35_27b": "qwen35_27b", "qwen38_27b": "qwen38_27b", "gemma4_12b": "gemma4_12b",
         "muse_glimmer_30b": "muse_glimmer_30b", "glm53_nvfp4": "glm53", "claude_sonnet5": "sonnet5",
         "gpt56terra": "gpt56terra"}
LATER = ("ep001", "ep002")
FIELDS = {"MQM": "_mqm", **{f"{g} part": f"group:{g}" for g in GROUPS}}


_CACHE: dict = {}


def split_of(path: Path, how: str) -> dict:
    if (path, how) not in _CACHE:
        v = load_verdicts(path)
        _CACHE[path, how] = parts(v) if how == "parts" else exclusive_parts(v) if how == "exclusive" else v
    return _CACHE[path, how]


def per_member(path: Path, field: str, how: str = "parts", episodes: tuple[str, ...] = LATER) -> dict[str, float]:
    """member -> mean over the episodes of the field (members with every one of them)."""
    p = split_of(path, how)
    out = {}
    for m in {m for m, _ in p}:
        xs = [p[m, e].get(field, 0.0) for e in episodes if (m, e) in p]
        if len(xs) == len(episodes):
            out[m] = mean(xs)
    return out


def issue_rate(path: Path, issue: str) -> dict[str, float]:
    """member -> share of sessions 2-3 replies the judge flags with the issue."""
    acc: dict = {}
    for v in split_of(path, "raw").values():
        if v["episode"] in LATER:
            acc.setdefault(v["member"], []).append(v.get(issue) == 1)
    return {m: mean(xs) for m, xs in acc.items()}


def spearman(x: list[float], y: list[float]) -> float:
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = float(k)
        return r
    rx, ry = rk(x), rk(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def severity(path: Path, level: str) -> dict[str, float]:
    """member -> share of replies at `level` over sessions 2-3."""
    acc: dict = {}
    for v in load_verdicts(path).values():
        if v["episode"] in LATER:
            acc.setdefault(v["member"], []).append(v["severity"] == level)
    return {m: mean(xs) for m, xs in acc.items()}


def fmt(r: dict | None, nd: int = 3) -> str:
    return "-" if not r or r.get("diff") is None else f"{r['diff']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--runs_root")
    ap.add_argument("--labels", type=Path, help="later_sessions/ablation_labels (continuity_v1/, member_side_v1/)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    J = args.ablation / "judging"
    args.out.mkdir(parents=True, exist_ok=True)
    rows, used = [], []
    paths = {}
    for c, key in PAPER.items():
        run = next(r for r in runs if r["key"] == key)
        paths[c] = {"sol_none": J / "gpt61_sol" / "main6" / c / "none" / "reference" / "judge" / "mindeval2" / "verdicts.jsonl",
                    "sol_regen": J / "gpt61_sol" / "main6" / c / "compacted" / "as_sent" / "judge" / "mindeval2"
                    / "verdicts.jsonl",
                    "sol_paper": judge_dir(root, run, "gpt61_sol", judges) / "verdicts.jsonl",
                    "mj_none": J / "mindjudge" / "main6" / c / "none" / "reference" / "judge" / "mindeval2" / "verdicts.jsonl",
                    "mj_regen": J / "mindjudge" / "main6" / c / "compacted" / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl",
                    "mj_paper": judge_dir(root, run, "own", judges) / "verdicts.jsonl"}
        ds = J / "deepseek_v4" / ("single" if (J / "deepseek_v4" / "single" / c).is_dir() else "main6") / c
        paths[c]["ds_none"] = ds / "none" / "reference" / "judge" / "mindeval2" / "verdicts.jsonl"
        paths[c]["ds_regen"] = ds / "compacted" / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl"
        used += [str(p) for p in paths[c].values() if p.is_file()]
    lines = ["# Memory's effect under GPT-6.1 Sol (sessions 2-3)", "",
             "Without memory (ablation, reference view) minus the paper's sessions 2-3, per member; 95% intervals "
             "resample members (within each counselor for the mean).", ""]

    def diff(c, a, b, field, sev=None, how="parts", episodes=LATER, issue=None):
        if sev:
            x, y = severity(paths[c][a], sev), severity(paths[c][b], sev)
        elif issue:
            x, y = issue_rate(paths[c][a], issue), issue_rate(paths[c][b], issue)
        else:
            x, y = per_member(paths[c][a], field, how, episodes), per_member(paths[c][b], field, how, episodes)
        return {m: x[m] - y[m] for m in set(x) & set(y)}

    # Sol: MQM and parts, per counselor and pooled
    lines += ["## GPT-6.1 Sol: without memory minus the paper", "",
              "| counselor | members | " + " | ".join(FIELDS) + " |", "|---|---|" + "---|" * len(FIELDS)]
    per_c = {name: {} for name in FIELDS}
    for c in PAPER:
        cells = []
        for name, fld in FIELDS.items():
            d = diff(c, "sol_none", "sol_paper", fld)
            per_c[name][c] = d
            r = pool({c: d})
            cells.append(fmt(r, 2))
            rows.append({"judge": "GPT-6.1 Sol", "contrast": "none minus paper", "counselor": c, "measure": name,
                         **{k: r.get(k) for k in ("diff", "lo", "hi")}, "members": len(d)})
        lines.append(f"| {c} | {len(per_c['MQM'][c])} | " + " | ".join(cells) + " |")
    cells = []
    for name in FIELDS:
        r = pool(per_c[name])
        cells.append(fmt(r, 2))
        rows.append({"judge": "GPT-6.1 Sol", "contrast": "none minus paper", "counselor": "mean", "measure": name,
                     **{k: r.get(k) for k in ("diff", "lo", "hi", "above", "below")}})
    lines += [f"| mean of {len(PAPER)} | | " + " | ".join(cells) + " |", ""]
    for sub, cs in (("the four free counselors (same session 1 as the paper)", list(PAPER)[:4]),
                    ("GLM-5.3, Sonnet 5, Terra (paper rows at the earlier harness)", list(PAPER)[4:])):
        r = pool({c: per_c["MQM"][c] for c in cs})
        lines.append(f"- {sub}: MQM {fmt(r)} ({r['above']} of {r['n_counselors']} above zero)")
        rows.append({"judge": "GPT-6.1 Sol", "contrast": "none minus paper", "counselor": sub, "measure": "MQM",
                     **{k: r.get(k) for k in ("diff", "lo", "hi", "above", "below")}})
    lines.append("")
    # severity
    lines += ["## GPT-6.1 Sol: share of replies at each severity, without memory minus the paper", "",
              "| severity | mean of counselors |", "|---|---|"]
    for lvl in [s for s in WEIGHT if s != "No concern"]:
        r = pool({c: diff(c, "sol_none", "sol_paper", None, lvl) for c in PAPER})
        lines.append(f"| {lvl} | {fmt(r)} |")
        rows.append({"judge": "GPT-6.1 Sol", "contrast": "none minus paper", "counselor": "mean",
                     "measure": f"share {lvl}", **{k: r.get(k) for k in ("diff", "lo", "hi")}})
    lines.append("")
    # Sol's own calibration, where Sol also judged the regenerated with-memory sessions 2-3
    cal = [c for c in PAPER if paths[c]["sol_regen"].is_file()]
    if cal:
        lines += ["## GPT-6.1 Sol calibration: the regenerated with-memory sessions 2-3 (made with the without-memory ones)",
                  "", "| counselor | measure | none minus paper | none minus regenerated (clean) | regenerated minus paper "
                  "(the shift) |", "|---|---|---|---|---|"]
        for name, fld in FIELDS.items():
            acc_s = {"np": {}, "nr": {}, "rp": {}}
            for c in cal:
                acc_s["np"][c] = diff(c, "sol_none", "sol_paper", fld)
                acc_s["nr"][c] = diff(c, "sol_none", "sol_regen", fld)
                acc_s["rp"][c] = diff(c, "sol_regen", "sol_paper", fld)
                if fld in ("_mqm", "group:memory", "group:exploration"):
                    lines.append(f"| {c} | {name} | " + " | ".join(fmt(pool({c: acc_s[k][c]}), 2)
                                                                   for k in ("np", "nr", "rp")) + " |")
            lines.append(f"| mean of {len(cal)} | {name} | " + " | ".join(fmt(pool(acc_s[k]), 2)
                                                                          for k in ("np", "nr", "rp")) + " |")
            for k, cname in (("np", "none minus paper"), ("nr", "none minus regenerated"),
                             ("rp", "regenerated minus paper")):
                r = pool(acc_s[k])
                rows.append({"judge": "GPT-6.1 Sol", "contrast": f"calibration: {cname}", "counselor": "mean",
                             "measure": name, **{x: r.get(x) for x in ("diff", "lo", "hi")}})
        lines.append("")
    # MindJudge: the same contrast, the clean one, and their difference (the regeneration shift on its scale)
    lines += ["## The regeneration shift, on MindJudge-27B's scale (MQM, sessions 2-3)", "",
              "| counselor | none minus paper (as Sol's contrast) | none minus regenerated (clean) | "
              "regenerated minus paper (the shift) |", "|---|---|---|---|"]
    acc = {"np": {}, "nr": {}, "rp": {}}
    for c in PAPER:
        acc["np"][c] = diff(c, "mj_none", "mj_paper", "_mqm")
        acc["nr"][c] = diff(c, "mj_none", "mj_regen", "_mqm")
        acc["rp"][c] = diff(c, "mj_regen", "mj_paper", "_mqm")
        lines.append(f"| {c} | " + " | ".join(fmt(pool({c: acc[k][c]}), 2) for k in ("np", "nr", "rp")) + " |")
    lines.append("| mean | " + " | ".join(fmt(pool(acc[k]), 2) for k in ("np", "nr", "rp")) + " |")
    for k, name in (("np", "none minus paper"), ("nr", "none minus regenerated"), ("rp", "regenerated minus paper")):
        r = pool(acc[k])
        rows.append({"judge": "MindJudge-27B", "contrast": name, "counselor": "mean", "measure": "MQM",
                     **{x: r.get(x) for x in ("diff", "lo", "hi")}})
    lines.append("")
    # Each reply counted once (the figure's split), per counselor
    lines += ["## GPT-6.1 Sol, each weighted reply counted once: without memory minus the paper", "",
              "| counselor | " + " | ".join(EXCLUSIVE) + " |", "|---|" + "---|" * len(EXCLUSIVE)]
    ex = {cat: {} for cat in ["_mqm", *EXCLUSIVE]}
    for c in [*PAPER, "mean"]:
        cells = []
        for cat in ["_mqm", *EXCLUSIVE]:
            if c != "mean":
                ex[cat][c] = diff(c, "sol_none", "sol_paper", cat, how="exclusive")
            r = pool(ex[cat]) if c == "mean" else pool({c: ex[cat][c]})
            rows.append({"judge": "GPT-6.1 Sol", "contrast": "none minus paper (each reply once)", "counselor": c,
                         "measure": cat, **{k: r.get(k) for k in ("diff", "lo", "hi")}})
            if cat != "_mqm":
                cells.append(fmt(r, 2))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")
    # Issues: share of replies flagged, without memory minus the paper
    lines += ["## GPT-6.1 Sol: share of replies flagged with each issue, without memory minus the paper", "",
              "| issue | change (mean of counselors) |", "|---|---|"]
    res = []
    for issue in ISSUES:
        r = pool({c: diff(c, "sol_none", "sol_paper", None, issue=issue) for c in PAPER})
        res.append((issue, r))
        rows.append({"judge": "GPT-6.1 Sol", "contrast": "none minus paper", "counselor": "mean",
                     "measure": f"flag rate {issue}", **{k: r.get(k) for k in ("diff", "lo", "hi", "above", "below")}})
    for issue, r in sorted(res, key=lambda x: -abs(x[1]["diff"] or 0)):
        lines.append(f"| {issue} | {fmt(r)} |")
    lines.append("")
    # Session by session
    lines += ["## GPT-6.1 Sol by session: without memory minus the paper", "",
              "| session | MQM | memory part | exploration part |", "|---|---|---|---|"]
    for e, lab in (("ep001", "2"), ("ep002", "3")):
        cells = []
        for name, fld in (("MQM", "_mqm"), ("memory part", "group:memory"), ("exploration part", "group:exploration")):
            r = pool({c: diff(c, "sol_none", "sol_paper", fld, episodes=(e,)) for c in PAPER})
            cells.append(fmt(r))
            rows.append({"judge": "GPT-6.1 Sol", "contrast": "none minus paper", "counselor": "mean",
                         "measure": f"{name}, session {lab}", **{k: r.get(k) for k in ("diff", "lo", "hi")}})
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    # Three judges on the same without-memory sessions, each against its clean or available baseline
    lines += ["## The judges on the same without-memory sessions (MQM, sessions 2-3)", "",
              "| counselor | GPT-6.1 Sol: none minus paper | DeepSeek-V4.1: none minus regenerated | "
              "MindJudge-27B: none minus regenerated |", "|---|---|---|---|"]
    jd = {"sol": {}, "ds": {}, "mj": {}}
    for c in PAPER:
        jd["sol"][c] = per_c["MQM"][c]
        jd["ds"][c] = diff(c, "ds_none", "ds_regen", "_mqm")
        jd["mj"][c] = diff(c, "mj_none", "mj_regen", "_mqm")
        lines.append(f"| {c} | " + " | ".join(fmt(pool({c: jd[k][c]}), 2) for k in ("sol", "ds", "mj")) + " |")
        for k, jl in (("sol", "GPT-6.1 Sol"), ("ds", "DeepSeek-V4.1"), ("mj", "MindJudge-27B")):
            r = pool({c: jd[k][c]})
            rows.append({"judge": jl, "contrast": "memory effect (judge comparison)", "counselor": c, "measure": "MQM",
                         **{x: r.get(x) for x in ("diff", "lo", "hi")}})
    lines.append("| mean | " + " | ".join(fmt(pool(jd[k]), 2) for k in ("sol", "ds", "mj")) + " |")
    for k, jl in (("sol", "GPT-6.1 Sol"), ("ds", "DeepSeek-V4.1"), ("mj", "MindJudge-27B")):
        r = pool(jd[k])
        rows.append({"judge": jl, "contrast": "memory effect (judge comparison)", "counselor": "mean", "measure": "MQM",
                     **{x: r.get(x) for x in ("diff", "lo", "hi")}})
    lines.append("")
    # Overlap: of the without-memory replies Sol flags for memory, the share each other judge flags for memory
    mem_issues = GROUPS["memory"]
    lines += ["## Sol's memory failures in the without-memory sessions, as the other judges read them", "",
              "| judge | replies Sol flags for memory | flagged for memory | rated Major or worse |", "|---|---|---|---|"]
    for jk, jl in (("ds_none", "DeepSeek-V4.1"), ("mj_none", "MindJudge-27B")):
        n = flag = major = 0
        for c in PAPER:
            sol = {k: v for k, v in split_of(paths[c]["sol_none"], "raw").items() if v["episode"] in LATER}
            other = split_of(paths[c][jk], "raw")
            for k, v in sol.items():
                if any(v.get(i) == 1 for i in mem_issues) and k in other:
                    n += 1
                    o = other[k]
                    flag += any(o.get(i) == 1 for i in mem_issues)
                    major += o["severity"] in ("Major", "Critical")
        if n:
            lines.append(f"| {jl} | {n} | {100 * flag / n:.0f}% | {100 * major / n:.0f}% |")
            rows.append({"judge": jl, "contrast": "overlap with Sol memory flags (none, sessions 2-3)",
                         "counselor": "all", "measure": "memory flag share", "diff": flag / n, "members": n})
            rows.append({"judge": jl, "contrast": "overlap with Sol memory flags (none, sessions 2-3)",
                         "counselor": "all", "measure": "major share", "diff": major / n, "members": n})
    lines.append("")
    # Validity and dose-response, from the labelers
    if args.labels:
        def labels(prompt, c, mode):
            d = args.labels / prompt / c / mode
            return load_labels(d) if (d / "keys.jsonl").is_file() else {}
        use, rep_ = {}, {}
        for c in PAPER:
            lab = labels("continuity_v1", c, "compacted")
            sh = [len([o for o in (p.get("opportunities") or []) if o.get("status") == "used"])
                  / len(p["opportunities"]) for (_, m, e), p in lab.items() if e in LATER and p.get("opportunities")]
            use[c] = mean(sh) if sh else None
            labn = labels("member_side_v1", c, "none")
            rs = [len(p.get("repeats") or []) for (_, m, e), p in labn.items() if e in LATER]
            rep_[c] = mean(rs) if rs else None
        cs = [c for c in PAPER if use[c] is not None]
        eff = {c: mean(per_c["MQM"][c].values()) for c in cs}
        effm = {c: mean(per_c["memory part"][c].values()) for c in cs}
        lines += ["## Who loses most without memory: note use with memory against Sol's memory effect", "",
                  "| counselor | relevant notes used with memory (labeler, sessions 2-3) | member repeats a session without"
                  " memory | Sol: MQM effect | Sol: memory part |", "|---|---|---|---|---|"]
        for c in cs:
            lines.append(f"| {c} | {100 * use[c]:.0f}% | {rep_[c] if rep_[c] is None else round(rep_[c], 2)} | "
                         f"{eff[c]:+.2f} | {effm[c]:+.2f} |")
            rows.append({"judge": "labelers", "contrast": "validity", "counselor": c, "measure": "notes used",
                         "diff": use[c]})
            rows.append({"judge": "labelers", "contrast": "validity", "counselor": c,
                         "measure": "member repeats without memory", "diff": rep_[c]})
        lines += [f"- Spearman across {len(cs)} counselors: note use against the MQM effect "
                  f"{spearman([use[c] for c in cs], [eff[c] for c in cs]):+.2f}, against the memory part "
                  f"{spearman([use[c] for c in cs], [effm[c] for c in cs]):+.2f}", ""]
        # Dose-response in the without-memory sessions, every judge on the same sessions
        acont, aside = {}, {}
        for c in PAPER:
            acont.update(labels("continuity_v1", c, "none"))
            aside.update(labels("member_side_v1", c, "none"))
        members = sorted({m for (_, m, _) in aside})
        draws = member_draws(members, BOOT)
        lines += ["# Memory failures against score in the without-memory sessions (sessions 2-3)", ""]
        dose_rows: list = []
        for jk, jl in (("sol_none", "GPT-6.1 Sol"), ("ds_none", "DeepSeek-V4.1"), ("mj_none", "MindJudge-27B")):
            sources = [(f"{c}/none", paths[c][jk], set(), set(LATER)) for c in PAPER if paths[c][jk].is_file()]
            analyse(jl, sources, acont, aside, members, draws, dose_rows, lines, unit_of="counselor")
        for r in dose_rows:
            rows.append({"judge": r["judge"], "contrast": f"dose-response, without memory: {r['level']}",
                         "counselor": r.get("bin", "all"), "measure": r["measure"],
                         "diff": r.get("slope", r.get("mqm")), "lo": r.get("lo"), "hi": r.get("hi"),
                         "members": r.get("sessions")})
    with (args.out / "sol_memory.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    judge_ids = {}
    for p in sorted(set(used)):
        jj = Path(p).parent / "judge.json"
        if jj.is_file():
            j = json.loads(jj.read_text())
            judge_ids[p] = {k: j.get(k) for k in ("model", "api_base", "prompt_sha256", "params", "max_tokens")}
    (args.out / "manifest.json").write_text(json.dumps({"script": "analysis/sol_memory.py", "analysis_git": git_describe(),
                                                        "paper_rows": PAPER, "verdicts": judge_ids}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
