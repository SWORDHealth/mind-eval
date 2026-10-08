"""C3: continuity failures in sessions 2 onward, and whether the MQM judge sees them.

    python analysis/c3_continuity.py --runs_root <dir> --labels <c3_label.py output dir> --agreement <json> --out <dir>

Per counselor (sessions with no opportunity are left out of the per-opportunity rates):
  used, missed and contradicted per opportunity (pooled over the counselor's opportunities; intervals resample
  members), re-asks per session, and the share of sessions with a contradiction or a re-ask.

Against each judge, on the same replies:
  turn level     of the counselor replies the labeler marks as contradicting a note or re-asking, the share the
                 judge flags with a memory issue (missed_shared_context or false_memory), and with Major or
                 Critical; beside it, the same shares over every other reply in sessions 2 onward;
  session level  of the sessions with a contradiction or re-ask, the share where the judge flags any memory issue;
                 beside it, the share among sessions with neither.
The labeler's agreement with the reference codes (c3_agreement.py) is printed above the results.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import (
    BOOT,
    ci,
    judge_dir,
    judge_identity,
    load_verdicts,
    member_draws,
    registry,
    runs_root,
    write_manifest,
)

MEMORY_FLAGS = ("missed_shared_context", "false_memory")


def load_labels(d: Path) -> dict[tuple[str, str, str], dict]:
    keys = {k["id"]: k for k in map(json.loads, (d / "keys.jsonl").read_text().splitlines())}
    out = {}
    for line in (d / "labels.jsonl").read_text().splitlines():
        x = json.loads(line)
        if "parsed" in x and x["id"] in keys:
            k = keys[x["id"]]
            out[k["name"], k["member"], k["episode"]] = x["parsed"]
    return out


def ratio_ci(per_member: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
    """sum(num) / sum(den) over members, with a member bootstrap."""
    ms = sorted(per_member)
    if not ms:
        return float("nan"), float("nan"), float("nan")

    def r(sample):
        num, den = sum(per_member[m][0] for m in sample), sum(per_member[m][1] for m in sample)
        return num / den if den else float("nan")
    lo, hi = ci([r(d) for d in member_draws(ms, BOOT)])
    return r(ms), lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--agreement", type=Path)
    ap.add_argument("--judges", default="gpt61_sol,own,deepseek_v4,hy4")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = runs_root(args.runs_root)
    runs, judges = registry(root)
    labels = load_labels(args.labels)
    labels_by_run = defaultdict(dict)
    for (name, m, ep), p in labels.items():
        labels_by_run[name][m, ep] = p
    lines = ["# C3: continuity failures and whether the judge sees them", ""]
    if args.agreement and args.agreement.is_file():
        a = json.loads(args.agreement.read_text())
        v, t = a["labeler_vs_reference"], a.get("labeler_test_retest")
        nl = v["note_level"]
        lines += [f"Labeler agreement with reference codes ({v['sessions']} sessions): opportunities found "
                  f"recall {nl['recall']:.2f}, precision {nl['precision']:.2f}; status on shared notes "
                  f"{nl['status_agreement']:.2f} (kappa {nl['status_kappa']:.2f}); any contradiction or re-ask "
                  f"kappa {v['any_contradiction_or_reask']['kappa']:.2f}; per-session Spearman: "
                  + ", ".join(f"{k} {x['spearman']:.2f}" for k, x in v["session_level"].items())
                  + (f". Test-retest: status kappa {t['note_level']['status_kappa']:.2f}, recall "
                     f"{t['note_level']['recall']:.2f}." if t else "."), ""]
    rows = []
    lines += ["| counselor | sessions | with opportunity | used / opp. | missed / opp. | contradicted / opp. | "
              "re-asks / session | sessions with contradiction or re-ask |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lab = labels_by_run.get(r["key"], {})
        if not lab:
            continue
        per = {s: defaultdict(lambda: [0, 0]) for s in ("used", "missed", "contradicted")}
        reask, flagged, n_op_sessions = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), 0
        for (m, ep), p in lab.items():
            ops = p["opportunities"]
            if ops:
                n_op_sessions += 1
            for s in per:
                per[s][m][0] += sum(1 for o in ops if o["status"] == s)
                per[s][m][1] += len(ops)
            reask[m][0] += len(p["reasks"])
            reask[m][1] += 1
            flagged[m][0] += any(o["status"] == "contradicted" for o in ops) or bool(p["reasks"])
            flagged[m][1] += 1
        cells, row = [], {"counselor": r["key"], "sessions": len(lab), "sessions_with_opportunity": n_op_sessions}
        for s in per:
            v, lo, hi = ratio_ci({m: tuple(x) for m, x in per[s].items() if x[1]})
            row.update({s: v, f"{s}_lo": lo, f"{s}_hi": hi})
            cells.append(f"{v:.2f} [{lo:.2f}, {hi:.2f}]")
        for name, src in (("reasks_per_session", reask), ("share_contradiction_or_reask", flagged)):
            v, lo, hi = ratio_ci({m: tuple(x) for m, x in src.items()})
            row.update({name: v, f"{name}_lo": lo, f"{name}_hi": hi})
            cells.append(f"{v:.2f} [{lo:.2f}, {hi:.2f}]")
        rows.append(row)
        lines.append(f"| {r['label']} | {len(lab)} | {n_op_sessions} | " + " | ".join(cells) + " |")

    # Cross-tab against each judge, pooled over counselors.
    xrows = []
    lines += ["", "## Does the judge flag what the labeler finds?", "",
              "| judge | contradiction/re-ask replies | flagged memory issue | flagged Major+ | other replies: memory "
              "issue | other: Major+ | sessions with contradiction/re-ask: judge memory flag | other sessions |",
              "|---|---|---|---|---|---|---|---|"]
    for judge in args.judges.split(","):
        hit = {"bad": [0, 0, 0], "other": [0, 0, 0]}  # replies, memory flag, Major+
        sess = {"bad": [0, 0], "other": [0, 0]}
        for r in runs:
            lab = labels_by_run.get(r["key"], {})
            path = judge_dir(root, r, judge, judges) / "verdicts.jsonl"
            if not lab or not path.is_file():
                continue
            by_session = defaultdict(dict)
            for v in load_verdicts(path).values():
                by_session[v["member"], v["episode"]][v["turn"]] = v
            for (m, ep), p in lab.items():
                bad_turns = {o["turn"] for o in p["opportunities"] if o["status"] == "contradicted" and o["turn"]}
                bad_turns |= {x["turn"] for x in p["reasks"] if x.get("turn")}
                vs = by_session.get((m, ep), {})
                if not vs:
                    continue
                any_mem = False
                for turn, v in vs.items():
                    mem = any(v.get(f) == 1 for f in MEMORY_FLAGS)
                    any_mem |= mem
                    h = hit["bad" if turn in bad_turns else "other"]
                    h[0] += 1
                    h[1] += mem
                    h[2] += v["severity"] in ("Major", "Critical")
                s = sess["bad" if bad_turns else "other"]
                s[0] += 1
                s[1] += any_mem
        if not hit["bad"][0] and not hit["other"][0]:
            continue
        b, o = hit["bad"], hit["other"]
        sb, so = sess["bad"], sess["other"]
        x = {"judge": judge, "bad_replies": b[0], "bad_memory_flag": b[1] / b[0] if b[0] else None,
             "bad_major": b[2] / b[0] if b[0] else None, "other_replies": o[0],
             "other_memory_flag": o[1] / o[0] if o[0] else None, "other_major": o[2] / o[0] if o[0] else None,
             "bad_sessions": sb[0], "bad_sessions_memory_flag": sb[1] / sb[0] if sb[0] else None,
             "other_sessions": so[0], "other_sessions_memory_flag": so[1] / so[0] if so[0] else None}
        xrows.append(x)

        def pct(v):
            return "-" if v is None else f"{100 * v:.0f}%"
        lines.append(f"| {judge} | {b[0]} | {pct(x['bad_memory_flag'])} | {pct(x['bad_major'])} | "
                     f"{pct(x['other_memory_flag'])} | {pct(x['other_major'])} | "
                     f"{pct(x['bad_sessions_memory_flag'])} of {sb[0]} | {pct(x['other_sessions_memory_flag'])} of {so[0]} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "per_counselor.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "judge_crosstab.json").write_text(json.dumps(xrows, indent=2) + "\n")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    write_manifest(args.out, "analysis/c3_continuity.py", runs,
                   {j: judge_identity(root, runs, j, judges) for j in args.judges.split(",")},
                   {"labels": str(args.labels), "agreement": str(args.agreement)})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
