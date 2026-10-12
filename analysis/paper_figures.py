"""The paper's figures, drawn from the dashboard datasets (dashboard_data.py's output). Needs matplotlib, which the
repo does not depend on: run it with any Python that has it.

    python analysis/paper_figures.py --data <dashboard data dir> --out <dir>

Writes PDF and PNG of:
  fig_change      within-member MQM change from session 1 to sessions 2-3, per counselor (GPT-6.1 Sol, MindJudge-27B)
  fig_continuity  continuity use (labeler) against the GPT-6.1 Sol change, per counselor
  fig_ablation    memory ablation: MQM change from compacted per judge and view, mean over counselors and each one
  fig_sessions    the same, mean over counselors, per session, for the contrasts the text discusses
  fig_labelers    memory ablation: what the labelers see change inside the sessions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, MUTED, GRID = "#9a9893", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8.5, "axes.titleweight": "medium", "legend.frameon": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "pdf.fonttype": 42})


def load(data: Path, name: str) -> list[dict]:
    return json.loads((data / f"{name}.json").read_text())


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"{name}: pdf, png")


def zero(ax, axis: str = "x") -> None:
    (ax.axvline if axis == "x" else ax.axhline)(0, color=MUTED, lw=0.8, zorder=1)


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def fig_change(data: Path, out: Path) -> None:
    rows = load(data, "ranking_change")
    judges = [("GPT-6.1 Sol", BLUE), ("MindJudge-27B", ORANGE)]
    order = [r["counselor"] for r in sorted((r for r in rows if r["judge"] == "GPT-6.1 Sol"), key=lambda r: r["mqm_s1"])]
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    for k, (judge, color) in enumerate(judges):
        by = {r["counselor"]: r for r in rows if r["judge"] == judge}
        for i, c in enumerate(order):
            r = by.get(c)
            if not r:
                continue
            y = i + (k - 0.5) * 0.32
            ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot(r["change"], y, "o", ms=3.6, color=color, mec="white", mew=0.6, zorder=3,
                    label=judge if i == 0 else None)
    zero(ax)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("MQM change, sessions 2–3 minus session 1 (higher is worse)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, 1.07), ncol=2, fontsize=7)
    ax.set_title("Change from session 1, best session-1 score first", loc="left", pad=16)
    save(fig, out, "fig_change")


def fig_continuity(data: Path, out: Path) -> None:
    rows = load(data, "continuity")
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for r in rows:
        ax.plot([r["use_lo"], r["use_hi"]], [r["mqm_change_gpt61"]] * 2, color=BLUE, lw=0.9, alpha=0.5, zorder=2)
        ax.plot(r["use"], r["mqm_change_gpt61"], "o", ms=4, color=BLUE, mec="white", mew=0.6, zorder=3)
    top = sorted(rows, key=lambda r: -r["mqm_change_gpt61"])[:2]
    low = sorted(rows, key=lambda r: r["mqm_change_gpt61"])[:2]
    for r, (dx, dy) in zip(top + low, [(8, 2), (8, -9), (6, -3), (6, 3)]):
        ax.annotate(r["counselor"], (r["use"], r["mqm_change_gpt61"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=MUTED)
    rho = spearman([r["use"] for r in rows], [r["mqm_change_gpt61"] for r in rows])
    zero(ax, "y")
    ax.set_xlabel("Share of relevant notes used (labeler), sessions 2–3")
    ax.set_ylabel("MQM change from session 1\n(GPT-6.1 Sol, higher is worse)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_title(f"Counselors that use their memory improve (Spearman ρ {rho:.2f}, {len(rows)} counselors)",
                 loc="left", fontsize=7.5)
    save(fig, out, "fig_continuity")


def fig_ablation(data: Path, out: Path) -> None:
    pooled = [r for r in load(data, "ablation_pooled") if r["measure"] == "MQM" and r["session"] == "s2–s6"]
    each = [r for r in load(data, "ablation_mqm") if r["session"] == "s2–s6"]
    panels = [("MindJudge-27B", "as sent"), ("MindJudge-27B", "reference memory"),
              ("DeepSeek-V4.1", "as sent"), ("DeepSeek-V4.1", "reference memory")]
    modes = ["none", "full", "split"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.6), sharex=True, sharey=True)
    for ax, (judge, view) in zip(axes.flat, panels):
        for i, mode in enumerate(modes):
            ones = [r["diff"] for r in each if (r["judge"], r["view"], r["mode"]) == (judge, view, mode)]
            ax.plot(ones, [i - 0.18] * len(ones), "o", ms=2.6, color=GREY, alpha=0.8, zorder=2)
            r = next((z for z in pooled if (z["judge"], z["view"], z["mode"]) == (judge, view, mode)), None)
            if r:
                ax.plot([r["lo"], r["hi"]], [i + 0.12] * 2, color=BLUE, lw=1.8, solid_capstyle="round", zorder=3)
                ax.plot(r["diff"], i + 0.12, "o", ms=5, color=BLUE, mec="white", mew=0.8, zorder=4)
                ax.annotate(f"{r['diff']:+.2f}", (r["hi"], i + 0.12), xytext=(4, -2.5), textcoords="offset points",
                            fontsize=6.5, color=INK)
        zero(ax)
        ax.set_yticks(range(len(modes)), modes)
        ax.set_ylim(len(modes) - 0.5, -0.5)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(f"{judge} · {view}", loc="left")
    fig.supxlabel("MQM change from compacted, sessions 2–6 (higher is worse)", fontsize=8, y=0.02)
    h1, = axes[0, 0].plot([], [], "o", ms=5, color=BLUE, label="mean of counselors, 95% CI")
    h2, = axes[0, 0].plot([], [], "o", ms=2.6, color=GREY, label="one counselor")
    fig.legend(handles=[h1, h2], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_ablation")


def fig_sessions(data: Path, out: Path) -> None:
    pooled = [r for r in load(data, "ablation_pooled") if r["measure"] == "MQM" and r["session"] != "s2–s6"]
    series = [("full", "MindJudge-27B", "as sent", "Full · MindJudge, as sent", ORANGE),
              ("none", "DeepSeek-V4.1", "reference memory", "None · DeepSeek, reference memory", BLUE),
              ("none", "MindJudge-27B", "reference memory", "None · MindJudge, reference memory", AQUA)]
    sess = ["s2", "s3", "s4", "s5", "s6"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, (mode, judge, view, name, color) in enumerate(series):
        pts = [next((r for r in pooled if (r["mode"], r["judge"], r["view"], r["session"]) == (mode, judge, view, s)), None)
               for s in sess]
        xs = [i + (k - 1) * 0.06 for i, p in enumerate(pts) if p]
        ps = [p for p in pts if p]
        ax.plot(xs, [p["diff"] for p in ps], color=color, lw=1.6, zorder=3, label=name)
        ax.vlines(xs, [p["lo"] for p in ps], [p["hi"] for p in ps], color=color, lw=1.0, zorder=2)
        ax.plot(xs, [p["diff"] for p in ps], "o", ms=3.6, color=color, mec="white", mew=0.6, zorder=4)
    zero(ax, "y")
    ax.set_xticks(range(len(sess)), [f"Session {s[1:]}" for s in sess])
    ax.set_ylabel("MQM change from compacted")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=6.5, loc="center right")
    ax.set_title("Mean of 11 counselors by session, 95% CI", loc="left")
    save(fig, out, "fig_sessions")


def fig_labelers(data: Path, out: Path) -> None:
    pooled = [r for r in load(data, "ablation_pooled") if r["judge"] == "Labelers"]
    measures = [("use beyond restatement", "Use beyond restatement (share)", "{:+.3f}"),
                ("re-asks / session", "Re-asks / session", "{:+.2f}"),
                ("member repeats / session", "Member repeats / session", "{:+.2f}")]
    modes = [("none", BLUE), ("full", ORANGE), ("split", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.9))
    for ax, (m, title, fmt) in zip(axes, measures):
        for i, (mode, color) in enumerate(modes):
            r = next((z for z in pooled if z["measure"] == m and z["mode"] == mode), None)
            if not r:
                continue
            ax.plot([r["lo"], r["hi"]], [i, i], color=color, lw=1.8, solid_capstyle="round", zorder=2)
            ax.plot(r["diff"], i, "o", ms=5, color=color, mec="white", mew=0.8, zorder=3)
            ax.annotate(fmt.format(r["diff"]), (r["hi"], i),
                        xytext=(4, -2.5), textcoords="offset points", fontsize=6.5, color=INK)
        zero(ax)
        ax.set_yticks(range(len(modes)), [m for m, _ in modes])
        ax.set_ylim(len(modes) - 0.5, -0.5)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel("change from compacted")
        ax.margins(x=0.25)
    fig.tight_layout()
    save(fig, out, "fig_labelers")


GROUP_COLORS = {"memory": BLUE, "exploration": ORANGE, "advice and validation": AQUA, "style and scope": YELLOW,
                "other": "#e87ba4"}


def read_csv(path: Path) -> list[dict]:
    import csv
    return list(csv.DictReader(path.open()))


#: drivers.py's mutually exclusive categories. A reply flagged for memory and exploration both is drawn in memory's
#: blue hatched with exploration's orange, so the two groups' parts cannot trade weight within the same replies.
EXCLUSIVE_STYLE = {"memory only": (BLUE, None), "memory and exploration": (BLUE, ORANGE),
                   "exploration only": (ORANGE, None), "advice and validation": (AQUA, None),
                   "style and scope": (YELLOW, None), "other": ("#e87ba4", None)}


def fig_drivers(later: Path, out: Path) -> None:
    rows = [r for r in read_csv(later / "drivers" / "drivers_exclusive.csv") if r["judge"] == "GPT-6.1 Sol"]
    by = {}
    for r in rows:
        by.setdefault(r["counselor"], {})[r["field"]] = float(r["diff"])
    mean_key = "mean of counselors"
    order = sorted((c for c in by if c != mean_key), key=lambda c: by[c]["_mqm"])
    ys = {c: i for i, c in enumerate(order)}
    ys[mean_key] = len(order) + 0.6
    plt.rcParams["hatch.linewidth"] = 1.4
    fig, ax = plt.subplots(figsize=(4.8, 4.9))
    for c, y in ys.items():
        pos = neg = 0.0
        for cat, (color, hatch) in EXCLUSIVE_STYLE.items():
            v = by[c].get(cat, 0.0)
            left = pos if v >= 0 else neg + v
            if hatch:
                ax.barh(y, abs(v), left=left, color=color, height=0.64, hatch="////", edgecolor=hatch, linewidth=0,
                        label=cat if c == order[0] else None)
                ax.barh(y, abs(v), left=left, height=0.64, fill=False, edgecolor="white", linewidth=0.6)
            else:
                ax.barh(y, abs(v), left=left, color=color, height=0.64, edgecolor="white", linewidth=0.6,
                        label=cat if c == order[0] else None)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.plot(by[c]["_mqm"], y, "D", ms=3.8, color=INK, zorder=4, label="total change" if c == order[0] else None)
    zero(ax)
    lo = min(sum(min(by[c].get(k, 0.0), 0) for k in EXCLUSIVE_STYLE) for c in ys)
    hi = max(sum(max(by[c].get(k, 0.0), 0) for k in EXCLUSIVE_STYLE) for c in ys)
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.axhline(len(order) - 0.2, color=GRID, lw=0.8)
    ax.set_yticks(list(ys.values()), [c if c != mean_key else f"Mean of {len(order)}" for c in ys])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("MQM change from session 1 to sessions 2–3 (GPT-6.1 Sol)\neach weighted reply counted once, in one "
                  "category")
    ax.grid(axis="x", color=GRID, lw=0.6)
    handles, labels = ax.get_legend_handles_labels()
    keep = ["total change", *EXCLUSIVE_STYLE]
    pairs = sorted((p for p in zip(handles, labels) if p[1] in keep), key=lambda p: keep.index(p[1]))
    ax.legend(*zip(*pairs), fontsize=6.5, loc="upper center", bbox_to_anchor=(0.4, -0.15), ncol=3)
    ax.set_title("What moves each counselor's score after session 1", loc="left")
    save(fig, out, "fig_drivers")


def fig_dose(later: Path, out: Path) -> None:
    rows = read_csv(later / "dose_response" / "dose_response.csv")
    bins = [r for r in rows if r["judge"] == "GPT-6.1 Sol" and r["level"] == "session_bin"]
    reply = [r for r in rows if r["level"] == "reply" and r["measure"].startswith("repeat-forcing")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.1, 1]})
    xs = range(len(bins))
    a.vlines(xs, [float(r["lo"]) for r in bins], [float(r["hi"]) for r in bins], color=BLUE, lw=1.4)
    a.plot(xs, [float(r["mqm"]) for r in bins], "o-", color=BLUE, ms=5, mec="white", mew=0.8, lw=1.4)
    a.set_xticks(list(xs), [r["bin"] for r in bins])
    a.set_xlabel("times the member repeated a recorded fact")
    a.set_ylabel("session MQM (GPT-6.1 Sol)")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_title("Sessions where the member repeats score worse", loc="left")
    order = ["GPT-6.1 Sol", "DeepSeek-V4.1", "Hy4-preview", "MindJudge-27B"]
    rs = sorted((r for r in reply if r["judge"] in order), key=lambda r: order.index(r["judge"]))
    ys = range(len(rs))
    b.hlines(ys, [float(r["lo"]) for r in rs], [float(r["hi"]) for r in rs], color=ORANGE, lw=1.8)
    b.plot([float(r["slope"]) for r in rs], ys, "o", color=ORANGE, ms=5, mec="white", mew=0.8)
    b.set_yticks(list(ys), [r["judge"] for r in rs])
    b.invert_yaxis()
    zero(b)
    b.set_xlabel("extra MQM weight on the reply that made\nthe member repeat (vs other replies)")
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_title("…and the reply that caused it", loc="left")
    fig.tight_layout()
    save(fig, out, "fig_dose")


def fig_memory_share(later: Path, out: Path) -> None:
    rows = [r for r in read_csv(later / "memory_share" / "memory_share.csv")
            if r["judge"] == "DeepSeek-V4.1" and r["view"] == "reference"]
    by = {}
    for r in rows:
        by.setdefault(r["counselor"], {})[r["measure"]] = float(r["diff"])
    labels = {k: c["label"] for k, c in __import__("common").ablation_counselors().items()}
    cs = sorted((c for c in by if c != "mean of counselors"), key=lambda c: by[c]["without memory"])
    cs.append("mean of counselors")
    labels["mean of counselors"] = f"Mean of {len(cs) - 1} counselors"
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for i, c in enumerate(cs):
        w, wo = by[c]["with memory"], by[c]["without memory"]
        ax.plot([w, wo], [i, i], color=GREY, lw=1.2, zorder=1)
        ax.plot(w, i, "o", color=BLUE, ms=4.5, mec="white", mew=0.6, zorder=3, label="with memory" if i == 0 else None)
        ax.plot(wo, i, "o", color=ORANGE, ms=4.5, mec="white", mew=0.6, zorder=3,
                label="without the counselor's memory" if i == 0 else None)
    zero(ax)
    ax.axhline(len(cs) - 1.5, color=GRID, lw=0.8)
    ax.set_yticks(range(len(cs)), [labels.get(c, c) for c in cs])
    ax.set_xlabel("MQM change from session 1 (DeepSeek-V4.1,\nreference memory; higher is worse)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Memory keeps later sessions at session-1 quality", loc="left")
    save(fig, out, "fig_memory_share")


def fig_overlap(later: Path, out: Path) -> None:
    rows = read_csv(later / "judge_overlap" / "overlap.csv")
    order = ["GPT-6.1 Sol", "Hy4-preview", "DeepSeek-V4.1", "MindJudge-27B"]
    rows = sorted(rows, key=lambda r: order.index(r["judge"]))
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ys = range(len(rows))
    h = 0.36
    ax.barh([y - h / 2 for y in ys], [100 * float(r["memory_flag"]) for r in rows], height=h, color=BLUE,
            label="flags a memory issue")
    ax.barh([y + h / 2 for y in ys], [100 * float(r["major_plus"]) for r in rows], height=h, color=ORANGE,
            label="rates it Major or Critical")
    for y, r in zip(ys, rows):
        ax.annotate(f"{100 * float(r['memory_flag']):.0f}%", (100 * float(r["memory_flag"]), y - h / 2),
                    xytext=(3, -2.5), textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_yticks(list(ys), [r["judge"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of the replies GPT-6.1 Sol flags for memory")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("The other judges on Sol's memory failures", loc="left")
    save(fig, out, "fig_overlap")


def fig_trajectory(later: Path, out: Path) -> None:
    rows = read_csv(later / "trajectory" / "trajectory.csv")

    def pts(**kw):
        rs = [r for r in rows if all(r[k] == v for k, v in kw.items())]
        return sorted(((int(r["session"]), float(r["diff"]), float(r["lo"]), float(r["hi"])) for r in rs))
    modes = [("compacted", "with memory (benchmark log)", BLUE), ("none", "without the counselor's memory", ORANGE),
             ("full", "full transcripts", AQUA), ("split", "split notes", YELLOW)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    panels = [(axes[0], dict(judge="DeepSeek-V4.1", view="reference", measure="MQM"), "MQM (DeepSeek-V4.1, reference memory)"),
              (axes[1], dict(judge="labelers", measure="member repeats"), "member repeats a session (labeler)"),
              (axes[2], None, "GPT-6.1 Sol parts, paper runs")]
    for ax, sel, ylabel in panels[:2]:
        for k, (mode, name, color) in enumerate(modes):
            ps = pts(set="ablation", mode=mode, **sel)
            if not ps:
                continue
            xs = [x + (k - 1.5) * 0.05 for x, *_ in ps]
            emph = mode in ("compacted", "none")
            ax.plot(xs, [y for _, y, _, _ in ps], "-o", color=color, lw=1.8 if emph else 1.0, ms=3.6 if emph else 2.6,
                    alpha=1 if emph else 0.7, mec="white", mew=0.5, label=name)
            ax.vlines(xs, [lo for *_, lo, _ in ps], [hi for *_, hi in ps], color=color, lw=0.9, alpha=1 if emph else 0.6)
        ax.set_xticks(range(1, 7) if sel.get("judge") != "labelers" else range(2, 7))
        ax.set_xlabel("session")
        ax.set_ylabel(ylabel, fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_title("(a) Session score", loc="left", fontsize=7.5)
    axes[1].set_title("(b) Member repeats", loc="left", fontsize=7.5)
    ax = axes[2]
    for measure, color, name in (("memory part", "#4a3aa7", "memory issues"), ("exploration part", "#e34948",
                                                                                   "exploration issues")):
        ps = pts(set="paper", judge="GPT-6.1 Sol", measure=measure)
        base = ps[0][1]
        ax.plot([x for x, *_ in ps], [y - base for _, y, _, _ in ps], "-o", color=color, lw=1.8, ms=3.6, mec="white",
                mew=0.5, label=name)
        ax.vlines([x for x, *_ in ps], [lo - base for *_, lo, _ in ps], [hi - base for *_, hi in ps], color=color, lw=0.9)
    zero(ax, "y")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("session")
    ax.set_ylabel("MQM part, change from session 1", fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=6.3, loc="lower left")
    ax.set_title("(c) Paper runs, GPT-6.1 Sol", loc="left", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, "fig_trajectory")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="dashboard_data.py's output directory")
    ap.add_argument("--later", type=Path, help="the later_sessions dir: drivers/, dose_response/, memory_share/, "
                    "judge_overlap/ for the figures on what drives the change")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_change, fig_continuity, fig_ablation, fig_sessions, fig_labelers):
        f(args.data, args.out)
    if args.later:
        for f in (fig_drivers, fig_dose, fig_memory_share, fig_overlap, fig_trajectory):
            f(args.later, args.out)


if __name__ == "__main__":
    main()
