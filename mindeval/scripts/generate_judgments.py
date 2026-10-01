"""`python mindeval/scripts/generate_judgments.py`: annotates a run with either judge (see
mindeval.judge).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from jsonargparse import CLI

from mindeval import env
from mindeval.config import CallSpec
from mindeval.judge import JUDGE_VERSION_DICT, judge_run
from mindeval.utils import InputError, check_params, hold_lock

#: --judge_version's choices, read off JUDGE_VERSION_DICT so a new judge is one entry there, not two.
JudgeVersion = Literal[tuple(JUDGE_VERSION_DICT)]

#: --judge_params when none are given: the settings of the benchmark's judge, GPT-6.1 Sol.
DEFAULT_JUDGE_PARAMS = {"reasoning_effort": "high"}


def _missing(names: list[str]) -> InputError:
    return InputError("set " + ", ".join(names) + " in the environment or in a .env file (see .env.example)")


def main(output_dir: Path, judge_version: JudgeVersion = "mindeval2",
         judge_temperature: float | None = None, judge_max_tokens: int = 32768, judge_params: dict | None = None,
         judge_timeout: float = 1800, judge_retries: int = 3, max_workers: int = 16, limit: int = 0) -> int:
    """Annotates a run's finished sessions with the judge that judge_version picks.

    Verdicts and the summary are written under output_dir/judge/<judge_version>/; rerun the same
    command to judge whatever is still pending.

    Models, endpoints and keys come from the environment (or a .env file in the working directory):
    MINDEVAL_JUDGE_MODEL, with optional _API_BASE and _API_KEY; .env.example sets the benchmark's judge,
    GPT-6.1 Sol, and the defaults below are its settings. mindeval2 uses the prompt in
    mindeval/judge_prompts.py unless MINDEVAL_JUDGE_PROMPT names another file. See .env.example.

    Args:
        output_dir: the run directory; verdicts go to <output_dir>/judge/<judge_version>/
        judge_version: mindeval2 is an MQM judge, one verdict per counselor reply, and needs the
            counselor system prompt from the trace; mindeval1 is MindEval v1's five-criterion rubric,
            one verdict per session, and does not
        judge_temperature: null (the default) to leave it unsent, as reasoning models such as GPT-6.1 Sol expect
        judge_max_tokens: room for the judge's reasoning and verdict
        judge_params: JSON merged into every request; unset, it is '{"reasoning_effort": "high"}' (GPT-6.1 Sol's
            setting), and '{}' sends nothing extra. Part of the judge's identity, so a later pass with different
            params is refused like a different model
        judge_timeout: seconds per request
        judge_retries: attempts per unit
        max_workers: units judged in parallel
        limit: judge at most N more units; 0 judges them all

    Returns:
        0 if every finished session is judged, 1 if units are still pending.
    """
    env.load_env_file(Path(".env"))
    try:
        if max_workers < 1 or judge_retries < 1 or judge_timeout <= 0:
            raise InputError("--max_workers and --judge_retries must be at least 1, and --judge_timeout positive")
        judge_params = dict(DEFAULT_JUDGE_PARAMS) if judge_params is None else judge_params
        check_params("--judge_params", judge_params)
        missing: list[str] = []
        je = env.endpoint("JUDGE", missing)
        prompt = (env.optional("JUDGE_PROMPT", missing) if JUDGE_VERSION_DICT[judge_version].takes_prompt_file
                  else None)
        if missing:
            raise _missing(missing)
        spec = CallSpec(model=je.model, api_base=je.api_base, api_key=je.api_key, temperature=judge_temperature,
                        max_tokens=judge_max_tokens, timeout=judge_timeout, max_retries=judge_retries,
                        params=judge_params)
        if not (output_dir / "run.json").is_file():
            raise InputError(f"{output_dir} has no run.json; point --output_dir at a mindeval run directory")
        with hold_lock(output_dir / "judge" / judge_version):  # each judge version locks its own directory
            _, errors, pending = judge_run(output_dir, judge_version, spec, workers=max_workers, limit=limit,
                                           prompt_path=Path(prompt) if prompt else None)
            if pending:
                print(f"{pending} units are not judged yet ({errors} errored this pass); rerun the same command "
                      f"to judge them")
            return 1 if pending else 0
    except InputError as e:
        print(f"mindeval: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(CLI([main], as_positional=False))
