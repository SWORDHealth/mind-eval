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
regenerated with-memory sessions, both made together), whose difference is that shift on MindJudge's scale.
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
from common import WEIGHT, git_describe, judge_dir, load_verdicts, mean, registry, runs_root  # noqa: E402
from drivers import GROUPS, parts  # noqa: E402

#: ablation counselor -> its paper row in runs.yaml
PAPER = {"qwen35_27b": "qwen35_27b", "qwen38_27b": "qwen38_27b", "gemma4_12b": "gemma4_12b",
         "muse_glimmer_30b": "muse_glimmer_30b", "glm53_nvfp4": "glm53", "claude_sonnet5": "sonnet5",
         "gpt56terra": "gpt56terra"}
LATER = ("ep001", "ep002")
FIELDS = {"MQM": "_mqm", **{f"{g} part": f"group:{g}" for g in GROUPS}}


def per_member(path: Path, field: str) -> dict[str, float]:
    """member -> mean over sessions 2-3 of the field (members with both sessions)."""
    p = parts(load_verdicts(path))
    out = {}
    for m in {m for m, _ in p}:
        xs = [p[m, e].get(field, 0.0) for e in LATER if (m, e) in p]
        if len(xs) == len(LATER):
            out[m] = mean(xs)
    return out


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
                    "sol_paper": judge_dir(root, run, "gpt61_sol", judges) / "verdicts.jsonl",
                    "mj_none": J / "mindjudge" / "main6" / c / "none" / "reference" / "judge" / "mindeval2" / "verdicts.jsonl",
                    "mj_regen": J / "mindjudge" / "main6" / c / "compacted" / "as_sent" / "judge" / "mindeval2" / "verdicts.jsonl",
                    "mj_paper": judge_dir(root, run, "own", judges) / "verdicts.jsonl"}
        used += [str(p) for p in paths[c].values()]
    lines = ["# Memory's effect under GPT-6.1 Sol (sessions 2-3)", "",
             "Without memory (ablation, reference view) minus the paper's sessions 2-3, per member; 95% intervals "
             "resample members (within each counselor for the mean).", ""]

    def diff(c, a, b, field, sev=None):
        if sev:
            x, y = severity(paths[c][a], sev), severity(paths[c][b], sev)
        else:
            x, y = per_member(paths[c][a], field), per_member(paths[c][b], field)
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
