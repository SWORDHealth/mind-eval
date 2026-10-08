"""C3 labeler validation: agreement of the labeler with reference codes, and with itself.

    python analysis/c3_agreement.py --labels <dir>/labels.jsonl --ref <ref files...> [--retest <dir>/labels.jsonl]
        --out <file.json>

Reference codes are one JSON object per session ({"id", "opportunities", "reasks"}), the labeler's are its
labels.jsonl. Reported over the sessions both coded:
  - session level: Spearman and mean difference (labeler - reference) for opportunities, used, missed,
    contradicted and re-asks per session, and Cohen's kappa for "any contradiction or re-ask";
  - note level: how many of the reference's opportunities the labeler found (recall) and how many of its own the
    reference has (precision), and, on notes both coded, status agreement and Cohen's kappa;
  - the used share of opportunities, pooled, for each coder.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

STATUSES = ("used", "missed", "contradicted")


def load_labels(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        x = json.loads(line)
        if "parsed" in x:
            out[x["id"]] = x["parsed"]
    return out


def load_ref(paths: list[Path]) -> dict[str, dict]:
    out = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                out[x["id"]] = {"opportunities": [o for o in x.get("opportunities", [])
                                                  if str(o.get("status", "")).lower() in STATUSES],
                                "reasks": x.get("reasks", [])}
    return out


def counts(c: dict) -> dict:
    st = Counter(str(o["status"]).lower() for o in c["opportunities"])
    return {"opportunities": len(c["opportunities"]), **{s: st[s] for s in STATUSES}, "reasks": len(c["reasks"])}


def rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def compare(lab: dict[str, dict], ref: dict[str, dict]) -> dict:
    ids = sorted(set(lab) & set(ref))
    cl, cr = {i: counts(lab[i]) for i in ids}, {i: counts(ref[i]) for i in ids}
    session = {}
    for k in ("opportunities", *STATUSES, "reasks"):
        a, b = [cl[i][k] for i in ids], [cr[i][k] for i in ids]
        session[k] = {"spearman": spearman(a, b), "labeler_mean": sum(a) / len(a), "reference_mean": sum(b) / len(b)}
    flag = [(cl[i]["contradicted"] + cl[i]["reasks"] > 0, cr[i]["contradicted"] + cr[i]["reasks"] > 0) for i in ids]
    found = shared = lab_total = ref_total = 0
    status_pairs = []
    for i in ids:
        ln = {o["note"]: str(o["status"]).lower() for o in lab[i]["opportunities"]}
        rn = {o["note"]: str(o["status"]).lower() for o in ref[i]["opportunities"]}
        lab_total += len(ln)
        ref_total += len(rn)
        both = set(ln) & set(rn)
        shared += len(both)
        status_pairs += [(ln[n], rn[n]) for n in both]
    found = shared
    reask_l = {(i, r.get("turn")) for i in ids for r in lab[i]["reasks"]}
    reask_r = {(i, r.get("turn")) for i in ids for r in ref[i]["reasks"]}

    def used_share(c):
        st = Counter(str(o["status"]).lower() for i in ids for o in c[i]["opportunities"])
        tot = sum(st.values())
        return {s: st[s] / tot if tot else float("nan") for s in STATUSES}

    return {
        "sessions": len(ids),
        "session_level": session,
        "any_contradiction_or_reask": {"kappa": kappa(flag), "labeler": sum(a for a, _ in flag),
                                       "reference": sum(b for _, b in flag)},
        "note_level": {"labeler_opportunities": lab_total, "reference_opportunities": ref_total, "shared": shared,
                       "recall": found / ref_total if ref_total else float("nan"),
                       "precision": found / lab_total if lab_total else float("nan"),
                       "status_agreement": sum(a == b for a, b in status_pairs) / len(status_pairs)
                       if status_pairs else float("nan"),
                       "status_kappa": kappa(status_pairs),
                       "status_confusion": {f"{a}->{b}": n for (a, b), n in Counter(status_pairs).items()}},
        "reasks": {"labeler": len(reask_l), "reference": len(reask_r), "same_turn": len(reask_l & reask_r)},
        "status_shares": {"labeler": used_share(lab), "reference": used_share(ref)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--ref", nargs="+", required=True, type=Path)
    ap.add_argument("--retest", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    lab, ref = load_labels(args.labels), load_ref(args.ref)
    result = {"labeler_vs_reference": compare(lab, ref)}
    if args.retest:
        result["labeler_test_retest"] = compare(lab, load_labels(args.retest))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
