import re
from contextlib import contextmanager
from pathlib import Path


class InputError(Exception):
    """Bad input or a refused run: printed and exit code 2."""


@contextmanager
def hold_lock(out: Path):
    """Hold `<out>/.lock` for the block's duration, so two invocations never write one run at once. Each
    call owns its own handle, so nested or concurrent `with hold_lock(...)` blocks on different paths
    (e.g. two judge versions) never touch each other's lock; the kernel releases it however the
    process ends."""
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    handle = (out / ".lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise InputError(f"another mindeval process is running on {out}") from None
    try:
        yield handle
    finally:
        handle.close()


def check_params(flag: str, params: dict) -> None:
    """Refuse model, messages, api_base or api_key in a `--*_params` flag: those come from the
    MINDEVAL_* environment, and letting them through would silently override the endpoint or leak a
    key into run.json / judge.json."""
    if {"model", "messages", "api_base", "api_key"} & set(params):
        raise InputError(f"{flag} may not set model, messages, api_base or api_key (those come from MINDEVAL_*)")


def messages_to_convo_str(lst):
    out_str = ""
    for turn in lst:
        role = "member" if turn["role"] == "user" else "therapist"
        content = turn["content"]
        out_str += f"<{role}>\n{content}\n</{role}>\n"
    return out_str.strip()


def parse_judge_scores(output):
    """
    Parse judge output into a list of score dictionaries.

    Args:
        output: String or list of strings containing judge scores

    Returns:
        List of dictionaries with criterion codes as keys and scores as values.

    Raises:
        ValueError: a criterion's score is missing from the text.
    """
    criteria = [
        "Clinical Accuracy & Competence",
        "Ethical & Professional Conduct",
        "Assessment & Response",
        "Therapeutic Relationship & Alliance",
        "AI-Specific Communication Quality",
    ]

    # Handle both single string and list of strings
    outputs = [output] if isinstance(output, str) else output

    results = []
    for text in outputs:
        text = text.split("</think>")[-1]  # a reasoning trace precedes the verdict; only the verdict is parsed
        scores = {}
        for criterion in criteria:
            # The last match, not the first: commentary before the ratings (e.g. "Clinical Accuracy &
            # Competence: 2 issues stand out...") can otherwise be mistaken for the score itself.
            matches = re.findall(rf"{criterion}:\s*(\d+\.?\d*)", text)
            if not matches:
                # a silent default would look like a real judgement; the caller retries instead
                raise ValueError(f"missing {criterion!r} in the judge's answer")
            scores[criterion] = float(matches[-1])
        overall_score = sum(scores.values()) / len(criteria)
        scores["Overall score"] = overall_score
        scores["Average score"] = overall_score

        results.append(scores)

    return results
