"""Fork existing arcs after session 1 and run sessions 2 onward under a counselor memory mode.

    python analysis/fork_run.py --src <run dir> --members_file <ids.txt> -- <generate_interactions arguments>

The arguments after `--` go to `mindeval/scripts/generate_interactions.py` unchanged (`--output_dir`, `--sessions`,
`--memory_mode`, ...); models and endpoints come from MINDEVAL_* as usual. Before each member starts, its
session 1 is copied from the source run into the new run: the trace, the log written after it and its situation,
plus the checkpoint the harness would have written after session 1. The member then resumes at session 2.

Session 2's situation is not written again: the one the source run wrote (`ep001/situation.yaml`) is used, so the
member walks into session 2 with exactly the inputs it had in the source run (situation, memory, clock, schedule),
whatever the counselor's memory mode. Later situations are written as usual, from the forked session before.

Nothing is written to the source run. The new run's `fork.json` records the source, the members and the code. The
members are given in a file, one id per line, in the order they run; `profiles.jsonl` in the new run holds their
rows from the source run's profile file (checked against its sha256), since the harness runs the first N rows of a
profile file. Rerun the same command to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jsonargparse import CLI  # noqa: E402

from mindeval import arc as arc_module  # noqa: E402
from mindeval.config import load_config  # noqa: E402
from mindeval.models import MemberRow, Situation  # noqa: E402
from mindeval.scripts import generate_interactions  # noqa: E402

HERE = Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def situation_from_file(path: Path) -> Situation:
    """A situation.yaml as the harness writes it: one comment line, then the JSON."""
    return Situation.model_validate(json.loads(path.read_text(encoding="utf-8").split("\n", 1)[1]))


class Fork:
    def __init__(self, src: Path, members: list[str], seed: int):
        self.src, self.members, self.seed = src, members, seed
        self.run = json.loads((src / "run.json").read_text())
        self.cfg = load_config()
        if self.run["fingerprint"] != self.cfg.fingerprint:
            raise SystemExit(f"{src} was made at fingerprint {self.run['fingerprint']}, this code is "
                             f"{self.cfg.fingerprint}: its schedules would not carry over")
        missing = [m for m in members if m not in self.run["member_ids"]]
        if missing:
            raise SystemExit(f"not in {src}: {missing}")
        self.pinned = {m: situation_from_file(self._src_arc(m) / "ep001" / "situation.yaml") for m in members}
        path = Path(self.run["profiles"]["path"])
        if _sha(path) != self.run["profiles"]["sha256"]:
            raise SystemExit(f"{path} is not the profile file {src} was run on")
        self.lines = {MemberRow.model_validate_json(line).member_id: line for line in path.read_text().splitlines()
                      if line.strip()}

    def _src_arc(self, member: str) -> Path:
        return self.src / "members" / member / "arc"

    def write_profiles(self, dest: Path) -> None:
        text = "".join(self.lines[m] + "\n" for m in self.members)
        if dest.is_file() and dest.read_text() != text:
            raise SystemExit(f"{dest} holds other members; use a fresh output directory")
        dest.write_text(text)

    def install(self, arc_dir: Path) -> None:
        """Session 1 and the checkpoint after it, unless this member has already started here."""
        if (arc_dir / "_state.json").exists() or (arc_dir / "arc.json").exists() or (arc_dir / "ep000").exists():
            return
        member = arc_dir.parent.name
        src = self._src_arc(member)
        record = json.loads((src / "arc.json").read_text())
        row = MemberRow.model_validate_json(self.lines[member])
        arc_seed = arc_module.member_seed(member, self.seed)
        arc_id = arc_module.arc_id_of(row, arc_seed, self.cfg.fingerprint)
        if (arc_id, arc_seed) != (record["arc_id"], record["arc_seed"]):
            raise RuntimeError(f"{member}: the source arc is {record['arc_id']}, this code makes {arc_id}")
        first = record["episodes"][0]
        (arc_dir / "ep000").mkdir(parents=True)
        for name in ("trace.jsonl", "log.md", "situation.yaml"):
            shutil.copyfile(src / "ep000" / name, arc_dir / "ep000" / name)
        state = {"version": arc_module.STATE_VERSION, "arc_id": arc_id, "fingerprint": self.cfg.fingerprint,
                 "episodes_done": 1, "clock": first["closed_ts"] + record["shape"]["gaps_hours"][0],
                 "log": arc_module.parse_log((src / "ep000" / "log.md").read_text()), "episodes": [first]}
        (arc_dir / arc_module.STATE_FILE).write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")

    def provenance(self, out: Path, argv: list[str]) -> None:
        git = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=HERE,
                             capture_output=True, text=True).stdout.strip()
        record = {"source": str(self.src), "source_git": self.run.get("git"),
                  "source_counselor": self.run["counselor"]["model"], "fingerprint": self.cfg.fingerprint,
                  "members": self.members, "seed": self.seed, "code": git, "argv": argv,
                  "pinned_session2_situation_sha256": {
                      m: _sha(self._src_arc(m) / "ep001" / "situation.yaml") for m in self.members}}
        path = out / "fork.json"
        if path.is_file():
            old = json.loads(path.read_text())
            if {k: old[k] for k in ("source", "members", "seed")} != {k: record[k] for k in ("source", "members", "seed")}:
                raise SystemExit(f"{path} is a fork of other members or another run; use a fresh output directory")
            record = {**old, "resumed_with": old.get("resumed_with", []) + [{"code": git, "argv": argv}]}
        path.write_text(json.dumps(record, indent=2) + "\n")


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    split = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--members_file", required=True, type=Path)
    args = ap.parse_args(sys.argv[1:split])
    rest = sys.argv[split + 1:]
    out = Path(rest[rest.index("--output_dir") + 1])
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    members = [x.strip() for x in args.members_file.read_text().splitlines() if x.strip()]
    fork = Fork(args.src.resolve(), members, seed)
    if seed != fork.run["seed"]:
        raise SystemExit(f"--seed {seed}, but {args.src} was run with seed {fork.run['seed']}")
    out.mkdir(parents=True, exist_ok=True)
    profiles = out / "profiles.jsonl"
    fork.write_profiles(profiles)
    fork.provenance(out, rest)

    original_aside = generate_interactions.set_aside_partial
    original_evolve = arc_module.evolve_situation

    def set_aside_then_install(arc_dir: Path):
        moved = original_aside(arc_dir)
        fork.install(arc_dir)
        return moved

    def evolve_or_pinned(cfg, row, prev, last, gap_hours, patient, *, episode_id, **kw):
        if episode_id == 1:  # session 2: the source run's situation, so the member's inputs are the source's
            return fork.pinned[row.member_id]
        return original_evolve(cfg, row, prev, last, gap_hours, patient, episode_id=episode_id, **kw)

    generate_interactions.set_aside_partial = set_aside_then_install
    arc_module.evolve_situation = evolve_or_pinned
    argv = rest + ["--profiles_path", str(profiles), "--members", str(len(members))]
    try:
        return CLI([generate_interactions.main], as_positional=False, args=argv)
    except SystemExit as e:
        return e.code
    finally:
        generate_interactions.set_aside_partial = original_aside
        arc_module.evolve_situation = original_evolve


if __name__ == "__main__":
    raise SystemExit(main())
