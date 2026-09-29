"""`python mindeval/scripts/generate_judgments.py`: annotates a run's counselor replies with the MQM
judge (see mindeval.judge).
"""

from __future__ import annotations

import sys
from pathlib import Path

from jsonargparse import CLI

from mindeval import env
from mindeval.config import CallSpec
from mindeval.judge import JudgeError, judge_run
from mindeval.utils import InputError, hold_lock, release_locks


def _missing(names: list[str]) -> InputError:
    return InputError("set " + ", ".join(names) + " in the environment or in a .env file (see .env.example)")


def main(output_dir: Path, judge_temperature: float | None = 1.0, judge_max_tokens: int = 16384,
         judge_timeout: float = 1800, judge_retries: int = 3, max_workers: int = 16, limit: int = 0) -> int:
    """Annotates a run's counselor replies with the MQM judge. Verdicts and the summary are written
    under output_dir/judge/; rerun the same command to judge whatever is still pending.

    Models, endpoints and keys come from the environment (or a .env file in the working directory):
    MINDEVAL_JUDGE_MODEL, with optional _API_BASE and _API_KEY, and MINDEVAL_JUDGE_PROMPT, the path to
    the judge's prompt template. See .env.example.

    Args:
        output_dir: the run directory; verdicts go to <output_dir>/judge/
        judge_temperature: null to leave it unsent
        judge_max_tokens: room for the judge's reasoning and verdict
        judge_timeout: seconds per request
        judge_retries: attempts per reply
        max_workers: replies judged in parallel
        limit: judge at most N more replies; 0 judges them all

    Returns:
        0 if every finished session is judged, 1 if replies are still pending.
    """
    env.load_env_file(Path(".env"))
    try:
        if max_workers < 1 or judge_retries < 1 or judge_timeout <= 0:
            raise InputError("--max_workers and --judge_retries must be at least 1, and --judge_timeout positive")
        missing: list[str] = []
        je = env.endpoint("JUDGE", missing)
        prompt = env.required("JUDGE_PROMPT", missing)
        if missing:
            raise _missing(missing)
        spec = CallSpec(model=je.model, api_base=je.api_base, api_key=je.api_key, temperature=judge_temperature,
                        max_tokens=judge_max_tokens, timeout=judge_timeout, max_retries=judge_retries)
        if not (output_dir / "run.json").is_file():
            raise InputError(f"{output_dir} has no run.json; point --output_dir at a mindeval run directory")
        hold_lock(output_dir / "judge")  # one judge per run at a time
        try:
            _, errors, pending = judge_run(output_dir, spec, Path(prompt), workers=max_workers, limit=limit)
        except JudgeError as e:
            raise InputError(str(e)) from None
        if pending:
            print(f"{pending} replies are not judged yet ({errors} errored this pass); rerun the same command to "
                  f"judge them")
        return 1 if pending else 0
    except InputError as e:
        print(f"mindeval: {e}", file=sys.stderr)
        return 2
    finally:
        release_locks()


if __name__ == "__main__":
    raise SystemExit(CLI([main], as_positional=False))
