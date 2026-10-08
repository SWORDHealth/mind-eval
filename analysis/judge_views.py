"""Build the views a judge reads an ablation run through, without touching the run.

The MQM judge sees earlier sessions only through the counselor's system prompt (its `# Memory` section). Under
`none`, `full` or `split` that prompt differs from the benchmark's, so a difference in scores could come from what
the judge was shown rather than from what the counselor did. Two views separate the two:

  as_sent     the run as it is: run.json and members/ symlinked; the judge sees what the counselor saw.
  reference   every session's counselor system prompt replaced by the benchmark's (compacted mode), rendered from
              the run's template and the shared log written after the session before, which every mode keeps.
              The judge then has the same yardstick in every condition. Only the trace's last line changes; the
              rest of each trace is copied byte for byte.

    python analysis/judge_views.py <run dir> <view root>      -> <view root>/{as_sent,reference}/

For a compacted run the two views are the same text, which is checked, and only as_sent is written.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindeval.arc import counselor_memory_for, parse_log  # noqa: E402
from mindeval.session import counselor_template  # noqa: E402


def _template(run: dict):
    want = run["counselor"]["system_prompt_sha256"]
    for path in (Path(run["counselor"]["system_prompt"]),
                 Path(__file__).resolve().parents[1] / "examples" / "counselor_system.j2"):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == want:
            return counselor_template(path.read_text())
    raise SystemExit(f"no counselor template with sha256 {want[:12]}")


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.resolve() != target.resolve():
            raise SystemExit(f"{link} points elsewhere")
        return
    link.symlink_to(target)


def build(run_dir: Path, root: Path) -> list[Path]:
    run = json.loads((run_dir / "run.json").read_text())
    mode = run.get("memory_mode", "compacted")
    as_sent = root / "as_sent"
    as_sent.mkdir(parents=True, exist_ok=True)
    _link(run_dir / "run.json", as_sent / "run.json")
    _link(run_dir / "members", as_sent / "members")
    views = [as_sent]
    template = _template(run)
    ref = root / "reference"
    changed = 0
    for trace in sorted(run_dir.glob("members/*/arc/ep*/trace.jsonl")):
        lines = trace.read_text(encoding="utf-8").split("\n")
        while lines and not lines[-1]:
            lines.pop()
        last = json.loads(lines[-1])
        if last.get("type") != "session":
            continue  # unfinished: the judge skips it anyway
        ep = int(trace.parent.name[2:])
        arc = trace.parents[1]
        log = parse_log((arc / f"ep{ep - 1:03d}" / "log.md").read_text()) if ep > 0 else []
        prompt = template.render(memory=counselor_memory_for("compacted", log, [], ep))
        if mode == "compacted":
            if prompt != last["counselor_system_prompt"]:
                raise SystemExit(f"{trace}: a compacted run whose prompt is not the benchmark's")
            continue
        changed += prompt != last["counselor_system_prompt"]
        last["counselor_system_prompt"] = prompt
        out = ref / trace.relative_to(run_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(lines[:-1] + [json.dumps(last, ensure_ascii=False, sort_keys=True)
                                       .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
                                       .replace("\x85", "\\u0085")]) + "\n"
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            tmp = out.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(out)
    if mode != "compacted":
        _link(run_dir / "run.json", ref / "run.json")
        views.append(ref)
    print(f"{run_dir}: {mode}; views {[v.name for v in views]}; {changed} session prompts replaced")
    return views


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
