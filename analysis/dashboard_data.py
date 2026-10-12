"""The findings dashboard's datasets, from the analysis outputs: one JSON file per dataset, rows of plain values.

    python analysis/dashboard_data.py --runs_root <dir> --later <later_sessions dir> --out <dir>

Reads what c1_ranking.py, c2_issues.py, c3_continuity.py, c3_agreement.py, ablation.py, situation_choice.py and
api_cost.py wrote, plus the judged paper runs for the within-member change from session 1 to sessions 2-3 (computed
here with the same member bootstrap). Writes nothing anywhere else.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from common import (
    EPISODES,
    ablation_counselors,
    ci,
    load_judge,
    mean,
    member_draws,
    registry,
    runs_root,
    set_exclusions,
)

JUDGES = {"gpt61_sol": "GPT-6.1 Sol", "own": "MindJudge-27B", "deepseek_v4": "DeepSeek-V4.1", "hy4": "Hy4-preview",
          "mindjudge": "MindJudge-27B"}
COUNSELORS = {k: c["label"] for k, c in ablation_counselors().items()}
#: Not shown per counselor: GLM-5.3 on FP8 weights, the replicate of glm53_nvfp4 (it ran compacted and none only).
REPLICATES = ("glm53",)
#: The labeler measures the pooled dataset carries, and their names on the page.
LABEL_FIELDS = {"memory_use_share": "use beyond restatement", "used_share": "use", "reasks": "re-asks / session",
                "repeats": "member repeats / session", "corrections": "member corrections / session"}


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def f(x, nd=3):
    return None if x in (None, "", "nan") else round(float(x), nd)


def ranking_change(root: Path) -> list[dict]:
    """Within-member MQM change from session 1 to sessions 2-3, per judge and counselor, with a member bootstrap."""
    runs, judges = registry(root)
    members = runs[0]["run"]["member_ids"]
    draws = member_draws(members)
    out = []
    for judge in ("gpt61_sol", "own", "deepseek_v4", "hy4"):
        sess = load_judge(root, runs, judge, judges, check_coverage=False)["sessions"]
        for r in runs:
            s = sess[r["key"]]
            per = {m: (s[m, EPISODES[0]]["score"], (s[m, EPISODES[1]]["score"] + s[m, EPISODES[2]]["score"]) / 2)
                   for m in members if all((m, e) in s for e in EPISODES)}
            ms = [m for m in members if m in per]
            change = mean(per[m][1] - per[m][0] for m in ms)
            lo, hi = ci([mean(per[m][1] - per[m][0] for m in d if m in per) for d in draws])
            out.append({"judge": JUDGES[judge], "counselor": r["label"], "fingerprint": r["fingerprint"][:8],
                        "mqm_s1": round(mean(per[m][0] for m in ms), 3), "mqm_later": round(mean(per[m][1] for m in ms), 3),
                        "change": round(change, 3), "lo": round(lo, 3), "hi": round(hi, 3), "members": len(ms)})
    return out


def ranking_stats(later: Path) -> list[dict]:
    out = []
    for sub, name in (("c1", "All 18 counselors"), ("c1_shared", "11 on one schedule")):
        tau = {(r["judge"], r["contrast"]): r for r in rows(later / sub / "taus.csv")}
        inter = {(r["judge"], r["change"]): r for r in rows(later / sub / "interaction.csv")}
        spread = {(r["judge"], r["window"]): r for r in rows(later / sub / "spread.csv")}
        for j in ("gpt61_sol", "own", "deepseek_v4", "hy4"):
            t, i, s = tau[j, "s1~later"], inter[j, "later-s1"], spread[j, "later"]
            out.append({"set": name, "judge": JUDGES[j], "tau": f(t["tau"], 2), "tau_lo": f(t["lo"], 2),
                        "tau_hi": f(t["hi"], 2), "sd_change": f(i["sd_of_change_over_counselors"], 2),
                        "sd_noise95": f(i["null_95"], 2), "p": f(i["p_perm"], 4),
                        "spread_ratio": f(s["ratio_to_s1"], 2), "spread_lo": f(s["ratio_lo"], 2),
                        "spread_hi": f(s["ratio_hi"], 2)})
    return out


def issues(later: Path) -> list[dict]:
    keep = {"missed_shared_context": "Missed shared context", "gives_bad_advice": "Gives bad advice",
            "uncritical_endorsement": "Uncritical endorsement", "false_memory": "False memory",
            "treating_perspective_as_fact": "Perspective as fact", "critical": "Critical severity",
            "overstatement": "Overstatement", "anthropomorphizing": "Anthropomorphizing"}
    return [{"judge": JUDGES[r["judge"]], "category": keep[r["category"]], "rate_s1": f(r["rate_s1"], 4),
             "rate_later": f(r["rate_later"], 4), "change": f(r["change_later"], 4), "lo": f(r["change_later_lo"], 4),
             "hi": f(r["change_later_hi"], 4), "later_only": r["later_only"] == "True",
             "counselors_up": int(r["counselors_rising"])}
            for r in rows(later / "c2" / "pooled.csv") if r["category"] in keep]


def continuity(later: Path, change: list[dict]) -> list[dict]:
    per = json.loads((later / "c3" / "per_counselor.json").read_text())
    runs_labels = {r["key"]: r["label"] for r in registry(runs_root(None))[0]}
    ch = {r["counselor"]: r for r in change if r["judge"] == "GPT-6.1 Sol"}
    return [{"counselor": runs_labels[r["counselor"]], "use": round(r["used"], 3), "use_lo": round(r["used_lo"], 3),
             "use_hi": round(r["used_hi"], 3), "repeats": round(r["repeats_per_session"], 2),
             "sessions": r["sessions"], "mqm_change_gpt61": ch[runs_labels[r["counselor"]]]["change"]}
            for r in per]


def judge_flags(later: Path) -> list[dict]:
    return [{"judge": JUDGES[r["judge"]], "repeat_replies": r["bad_replies"],
             "memory_flag_repeat": round(r["bad_memory_flag"], 3), "memory_flag_other": round(r["other_memory_flag"], 3),
             "major_repeat": round(r["bad_major"], 3), "major_other": round(r["other_major"], 3)}
            for r in json.loads((later / "c3" / "judge_crosstab.json").read_text())]


def ablation_mqm(later: Path) -> list[dict]:
    out = []
    for r in rows(later / "ablation_main6" / "ablation.csv"):
        if r["judge"] not in ("mindjudge", "deepseek_v4") or not r["diff"]:
            continue
        m = re.fullmatch(r"mqm(?:_ep(\d+))?", r["field"])
        if not m or r["mode"] == "compacted_rerun_vs_source" or r["counselor"] in REPLICATES:
            continue
        session = f"s{int(m.group(1)) + 1}" if m.group(1) else "s2–s6"
        out.append({"counselor": COUNSELORS[r["counselor"]], "mode": r["mode"], "judge": JUDGES[r["judge"]],
                    "view": "as sent" if r["view"] == "as_sent" else "reference memory", "session": session,
                    "diff": f(r["diff"]), "lo": f(r["lo"]), "hi": f(r["hi"]), "members": int(r["n_members"])})
    return out


def ablation_behaviour(later: Path) -> list[dict]:
    """Per counselor and mode: the labeler and trace measures, read from ablation.py's report tables."""
    text = (later / "ablation_main6" / "report.md").read_text()
    out = []
    for block in text.split("\n## ")[1:]:
        key = block.split("\n", 1)[0].strip()
        if key not in COUNSELORS or key in REPLICATES:
            continue
        lab = block.split("### Continuity (labeler)")[1] if "### Continuity (labeler)" in block else ""
        tr = block.split("### Counselor input and output")[1].split("###")[0]
        for mode in ("compacted", "none", "full", "split"):
            lm = re.search(rf"^\| {mode} \| ([^\n]+)\|$", lab, re.M)
            tm = re.search(rf"^\| {mode} \| ([^\n]+)\|$", tr, re.M)
            lv = [float(x) for x in lm.group(1).split("|")] if lm else [None] * 7
            tv = [float(x) for x in tm.group(1).split("|")]
            out.append({"counselor": COUNSELORS[key], "mode": mode, "use": lv[0], "use_beyond_restatement": lv[1],
                        "reasks": lv[4], "member_repeats": lv[6], "replies": tv[3], "member_left": tv[4] / 100,
                        "reply_tokens": tv[2], "first_prompt_tokens": tv[1]})
    return out


def full_tokens(later: Path) -> list[dict]:
    text = (later / "ablation_main6" / "report.md").read_text()
    out = []
    for block in text.split("\n## ")[1:]:
        key = block.split("\n", 1)[0].strip()
        m = re.search(r"first-reply prompt tokens by session: ([^\n]+)", block)
        if key not in COUNSELORS or key in REPLICATES or not m:
            continue
        for s, mean_tok, mx in re.findall(r"s(\d) (\d+) \(max (\d+)\)", m.group(1)):
            out.append({"counselor": COUNSELORS[key], "session": f"s{s}", "prompt_tokens": int(mean_tok),
                        "max_tokens": int(mx)})
    return out


def situations(later: Path) -> list[dict]:
    out = []
    for r in json.loads((later / "situations_main6" / "rows.json").read_text()):
        if r["counselor"] in REPLICATES:
            continue
        pair = r["pair"].replace("-compacted", " vs compacted").replace("compacted-source", "rerun vs source (floor)")
        if "choice_accuracy" in r:
            out.append({"counselor": COUNSELORS[r["counselor"]], "pair": pair, "measure": "forced-choice accuracy",
                        "value": round(r["choice_accuracy"], 3), "lo": round(r["lo"], 3), "hi": round(r["hi"], 3)})
        elif r.get("minus_floor") is not None:
            out.append({"counselor": COUNSELORS[r["counselor"]], "pair": pair, "measure": "similarity minus floor",
                        "value": round(r["minus_floor"], 3), "lo": round(r["lo"], 3), "hi": round(r["hi"], 3)})
    return out


def ablation_pooled(later: Path) -> list[dict]:
    """The mean across counselors (ablation.py's pooled.csv): MQM per judge and view, overall and per session, the
    labeler measures, and the noise floor."""
    out = []
    for r in rows(later / "ablation_main6" / "pooled.csv"):
        if not r["diff"]:
            continue
        m = re.fullmatch(r"mqm(?:_ep(\d+))?", r["field"])
        if r["judge"] in ("mindjudge", "deepseek_v4") and m:
            measure, session = "MQM", (f"s{int(m.group(1)) + 1}" if m.group(1) else "s2–s6")
        elif r["judge"] == "labeler" and r["field"] in LABEL_FIELDS:
            measure, session = LABEL_FIELDS[r["field"]], "s2–s6"
        else:
            continue
        out.append({"measure": measure, "judge": JUDGES.get(r["judge"], "Labelers"),
                    "view": {"as_sent": "as sent", "reference": "reference memory"}.get(r["view"], "-"),
                    "mode": "rerun vs source (floor)" if r["mode"] == "compacted_rerun_vs_source" else r["mode"],
                    "session": session, "diff": f(r["diff"]), "lo": f(r["lo"]), "hi": f(r["hi"]),
                    "counselors": int(r["n_counselors"]), "above": int(r["above"]), "below": int(r["below"])})
    return out


def situations_pooled(later: Path) -> list[dict]:
    path = later / "situations_main6" / "pooled.json"
    if not path.is_file():
        return []
    return [{"measure": x["measure"], "pair": x["pair"].replace("-compacted", " vs compacted")
             .replace("compacted-source", "rerun vs source (floor)"), "value": round(x["diff"], 3),
             "lo": round(x["lo"], 3), "hi": round(x["hi"], 3), "counselors": x["n_counselors"]}
            for x in json.loads(path.read_text()) if x.get("diff") is not None]


def validation(later: Path) -> list[dict]:
    c = json.loads((later / "c3_labels" / "continuity_v1_agreement.json").read_text())
    m = json.loads((later / "c3_labels" / "member_side_v1_agreement.json").read_text())["labeler_vs_reference"]
    rc = json.loads((later / "c3_validation" / "rc_agreement.json").read_text())["labeler_vs_reference"]
    v, t = c["labeler_vs_reference"], c["labeler_test_retest"]
    return [
        {"labeler": "Continuity use", "measure": "Used per session, Spearman vs reference",
         "value": round(v["session_level"]["used"]["spearman"], 2), "used": "yes"},
        {"labeler": "Continuity use", "measure": "Status on shared notes, kappa vs reference",
         "value": round(v["note_level"]["status_kappa"], 2), "used": "yes"},
        {"labeler": "Continuity use", "measure": "Status, kappa test–retest", "value": round(t["note_level"]["status_kappa"], 2),
         "used": "yes"},
        {"labeler": "Continuity use", "measure": "Pooled used share, labeler", "value": round(v["status_shares"]["labeler"]["used"], 2),
         "used": "yes"},
        {"labeler": "Continuity use", "measure": "Pooled used share, reference",
         "value": round(v["status_shares"]["reference"]["used"], 2), "used": "yes"},
        {"labeler": "Member repeats", "measure": "Repeats per session, Spearman vs reference",
         "value": round(m["repeats"]["spearman"], 2), "used": "yes"},
        {"labeler": "Member repeats", "measure": "Any repeat, kappa vs reference", "value": round(m["repeats"]["any_kappa"], 2),
         "used": "yes"},
        {"labeler": "Strict re-asks", "measure": "Any re-ask, kappa vs reference", "value": round(rc["reasks"]["any_kappa"], 2),
         "used": "no"},
        {"labeler": "Strict contradictions", "measure": "Any contradiction, kappa vs reference",
         "value": round(rc["contradictions"]["any_kappa"], 2), "used": "no"},
    ]


def facts(later: Path, stats, iss, cont, flags, pool, beh, tok, sit, extra: dict | None = None) -> list[dict]:
    """The numbers the page's sentences quote, each named by a key, read from the datasets above."""
    def pick(rs, **kw):
        return next(r for r in rs if all(r.get(k) == v for k, v in kw.items()))
    g = pick(stats, set="All 18 counselors", judge="GPT-6.1 Sol")
    taus = [r["tau"] for r in stats if r["set"] == "All 18 counselors"]
    ps = [r["p"] for r in stats]
    shared = pick(stats, set="11 on one schedule", judge="GPT-6.1 Sol")
    msc = pick(iss, judge="GPT-6.1 Sol", category="Missed shared context")
    use = sorted(cont, key=lambda r: -r["use"])
    fl = {r["judge"]: r for r in flags}
    def pooled(measure, mode, judge, view="-"):
        return pick(pool, measure=measure, mode=mode, judge=judge, view=view, session="s2–s6")

    def ci_text(r, nd=2):
        return f"{r['diff']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"
    full_mj = pooled("MQM", "full", "MindJudge-27B", "as sent")
    none_ds = pooled("MQM", "none", "DeepSeek-V4.1", "reference memory")
    none_mj = pooled("MQM", "none", "MindJudge-27B", "reference memory")
    beyond = pooled("use beyond restatement", "none", "Labelers")
    reasks = pooled("re-asks / session", "none", "Labelers")
    repeats = pooled("member repeats / session", "none", "Labelers")
    tmax = max(r["max_tokens"] for r in tok)
    floor = [r["value"] for r in sit if r["measure"] == "forced-choice accuracy" and "floor" in r["pair"]]
    def ranks(rs):
        rho = {}
        xs = [r["use"] for r in rs]
        ys = [r["mqm_change_gpt61"] for r in rs]
        rx = {i: sorted(xs).index(x) for i, x in enumerate(xs)}
        ry = {i: sorted(ys).index(y) for i, y in enumerate(ys)}
        n = len(xs)
        mx, my = mean(rx.values()), mean(ry.values())
        rho = sum((rx[i] - mx) * (ry[i] - my) for i in range(n)) / (
            sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
        return rho

    out = {
        "tau_min": f"{min(taus):.2f}", "tau_max": f"{max(taus):.2f}",
        "interaction_p_max": f"{max(ps):.3f}", "spread_gpt": f"{g['spread_ratio']:.2f}×",
        "spread_shared_gpt": f"{shared['spread_ratio']:.2f}×",
        "msc_s1": f"{100 * msc['rate_s1']:.1f}%", "msc_later": f"{100 * msc['rate_later']:.1f}%",
        "msc_up": f"{msc['counselors_up']}", "n_counselors": f"{len(cont)}",
        "use_top": f"{use[0]['use']:.2f}", "use_top_name": use[0]["counselor"],
        "use_bottom": f"{use[-1]['use']:.2f}", "use_bottom_name": use[-1]["counselor"],
        "use_change_rho": f"{ranks(cont):+.2f}",
        "flag_gpt_repeat": f"{100 * fl['GPT-6.1 Sol']['memory_flag_repeat']:.0f}%",
        "flag_gpt_other": f"{100 * fl['GPT-6.1 Sol']['memory_flag_other']:.0f}%",
        "flag_mj_repeat": f"{100 * fl['MindJudge-27B']['memory_flag_repeat']:.0f}%",
        "repeat_replies": f"{fl['GPT-6.1 Sol']['repeat_replies']:,}",
        "abl_counselors": f"{full_mj['counselors']}",
        "full_mj": ci_text(full_mj), "full_mj_above": f"{full_mj['above']}", "full_mj_d": f"{full_mj['diff']:+.2f}",
        "none_ds_d": f"{none_ds['diff']:+.2f}", "none_mj_d": f"{none_mj['diff']:+.2f}",
        "none_ds": ci_text(none_ds), "none_ds_above": f"{none_ds['above']}", "none_ds_n": f"{none_ds['counselors']}",
        "none_mj": ci_text(none_mj), "none_mj_above": f"{none_mj['above']}",
        "beyond_none": ci_text(beyond, 3), "beyond_none_below": f"{beyond['below']}", "beyond_n": f"{beyond['counselors']}",
        "reasks_none": ci_text(reasks), "repeats_none": ci_text(repeats),
        "full_tokens_max": f"{tmax:,}",
        "floor_min": f"{min(floor):.2f}", "floor_max": f"{max(floor):.2f}",
    }
    out.update(extra or {})
    return [{"key": k, "display": v} for k, v in out.items()]


def strict(later: Path) -> tuple[list[dict], dict]:
    """Strict re-asks and contradictions coded by reference coders: the judges on those replies, and the figures."""
    r = json.loads((later / "c3_strict" / "results" / "results.json").read_text())
    rows = [{"sessions": x["set"], "judge": x["judge"], "view": x["view"], "coded_replies": x["failure_replies"],
             "memory_flag_failure": round(x["memory_flag_failure"], 3),
             "memory_flag_other": round(x["memory_flag_other"], 3) if x["memory_flag_other"] is not None else None,
             "sessions_with_failure": x["sessions_with_failure"],
             "session_flagged": round(x["session_memory_flag_failure"], 3)} for x in r["crosstab"]]
    rates = {x["group"]: x for x in r["rates"]}
    figs = {"strict_kappa_reask": f"{r['agreement']['reasks']['kappa_any']:.2f}",
            "strict_kappa_contra": f"{r['agreement']['contradictions']['kappa_any']:.2f}",
            "strict_double_n": f"{r['agreement']['reasks']['sessions']}",
            "strict_paper_reasks": f"{rates['paper']['reasks'][0]:.2f}",
            "strict_glm_none_reasks": f"{rates['ablation glm53 none']['reasks'][0]:.2f}",
            "strict_glm_comp_reasks": f"{rates['ablation glm53 compacted']['reasks'][0]:.2f}"}
    return rows, figs


def api_plans(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_root")
    ap.add_argument("--later", required=True, type=Path)
    ap.add_argument("--api_plans", required=True, type=Path, help="JSON rows written from api_cost.py's estimates")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", type=Path, help="replies left out of every score (artifacts.py's leaks.jsonl)")
    args = ap.parse_args()
    set_exclusions(args.exclude)
    root = runs_root(args.runs_root)
    args.out.mkdir(parents=True, exist_ok=True)
    change = ranking_change(root)
    stats, iss = ranking_stats(args.later), issues(args.later)
    cont, flags = continuity(args.later, change), judge_flags(args.later)
    abl, beh, tok = ablation_mqm(args.later), ablation_behaviour(args.later), full_tokens(args.later)
    pool, sit, sitp = ablation_pooled(args.later), situations(args.later), situations_pooled(args.later)
    strict_rows, strict_figs = strict(args.later)
    data = {"strict_crosstab": strict_rows, "ranking_change": change, "ranking_stats": stats, "issues": iss, "continuity": cont, "judge_flags": flags,
            "ablation_mqm": abl, "ablation_pooled": pool, "ablation_behaviour": beh, "full_tokens": tok, "situations": sit,
            "situations_pooled": sitp,
            "validation": validation(args.later), "api_plans": api_plans(args.api_plans),
            "facts": facts(args.later, stats, iss, cont, flags, pool, beh, tok, sit, strict_figs)}
    for name, rs in data.items():
        (args.out / f"{name}.json").write_text(json.dumps(rs, ensure_ascii=False, indent=1) + "\n")
        print(f"{name}: {len(rs)} rows")


if __name__ == "__main__":
    main()
