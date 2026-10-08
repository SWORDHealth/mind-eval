"""Check against real runs that `memory_mode=compacted` gives the counselor exactly what it was given.

For every finished session of every run named, the counselor system prompt is rebuilt from the run's counselor
template and the log written after the session before, through `arc.counselor_memory_for("compacted", ...)`, and
compared with the prompt recorded in the trace. Session 1 is rebuilt too, and `none` is checked to equal it.

    python analysis/check_replay.py <run dir> [<run dir> ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindeval.arc import counselor_memory_for, parse_log  # noqa: E402
from mindeval.session import counselor_template  # noqa: E402


def check(run_dir: Path) -> tuple[int, list[str]]:
    run = json.loads((run_dir / "run.json").read_text())
    # The recorded path, or this checkout's default template, whichever has the sha256 the run recorded.
    want = run["counselor"]["system_prompt_sha256"]
    for tpl_path in (Path(run["counselor"]["system_prompt"]), Path(__file__).resolve().parents[1] / "examples" /
                     "counselor_system.j2"):
        text = tpl_path.read_bytes() if tpl_path.is_file() else b""
        if hashlib.sha256(text).hexdigest() == want:
            break
    else:
        return 0, [f"no template with sha256 {want[:12]} (recorded: {run['counselor']['system_prompt']})"]
    template = counselor_template(text.decode())
    n, bad = 0, []
    for arc in sorted(run_dir.glob("members/*/arc")):
        eps = sorted(arc.glob("ep*/trace.jsonl"))
        for trace in eps:
            ep = int(trace.parent.name[2:])
            last = json.loads(trace.read_text().splitlines()[-1])
            if last.get("type") != "session":
                continue
            log = parse_log((arc / f"ep{ep - 1:03d}" / "log.md").read_text()) if ep > 0 else []
            rebuilt = template.render(memory=counselor_memory_for("compacted", log, [], ep))
            n += 1
            if rebuilt != last["counselor_system_prompt"]:
                bad.append(f"{trace}")
            if ep > 0 and template.render(memory=counselor_memory_for("none", log, [], ep)) != \
                    template.render(memory=counselor_memory_for("compacted", [], [], 0)):
                bad.append(f"{trace}: none differs from the session-1 prompt")
    return n, bad


def main() -> None:
    failed = False
    for d in sys.argv[1:]:
        n, bad = check(Path(d))
        failed |= bool(bad) or n == 0
        print(f"{d}: {n} sessions, {len(bad)} mismatches" + (f"; first: {bad[0]}" if bad else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
