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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="dashboard_data.py's output directory")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_change, fig_continuity, fig_ablation, fig_sessions, fig_labelers):
        f(args.data, args.out)


if __name__ == "__main__":
    main()
