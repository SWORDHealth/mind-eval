import re
from pathlib import Path


class InputError(Exception):
    """Bad input or a refused run: printed and exit code 2."""


_HELD: list = []


def hold_lock(out: Path):
    """Hold `<out>/.lock` until the command returns, so two invocations never write one run at once.
    `release_locks()` releases it; the kernel releases it however the process ends."""
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    handle = (out / ".lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise InputError(f"another mindeval process is running on {out}") from None
    _HELD.append(handle)
    return handle


def release_locks() -> None:
    """Release every lock `hold_lock` is still holding. Call from a `finally`, whatever the outcome."""
    while _HELD:
        _HELD.pop().close()


def print_messages(messages):
    role_to_name = {"user": "Member", "assistant": "Clinician", "system": "system"}
    for m in messages:
        print(f"{role_to_name[m['role']]}: {m['content']}")
        print("###########################################")


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
            # Match pattern like "CAC1: 4" with optional whitespace
            match = re.search(rf"{criterion}:\s*(\d+\.?\d*)", text)
            if match is None:
                # a silent default would look like a real judgement; the caller retries instead
                raise ValueError(f"missing {criterion!r} in the judge's answer")
            scores[criterion] = float(match.group(1))
        overall_score = sum(scores.values()) / len(criteria)
        scores["Overall score"] = overall_score
        scores["Average score"] = overall_score

        results.append(scores)

    return results


def separate_thinking_from_response(
    response: str,
    beginning_thinking_tag: str = "<think>",
    end_thinking_tag: str = "</think>",
) -> dict[str, str]:
    # regex for getting the content between the tags into thinking and the content after the end tag into response. Using only regex (no split) to avoid issues if the tags appear multiple times
    thinking_match = re.search(
        f"{re.escape(beginning_thinking_tag)}(.*?){re.escape(end_thinking_tag)}",
        response,
        re.DOTALL,
    )
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        actual_response = (
            response[thinking_match.end() :].strip()
            if response[thinking_match.end() :].strip()
            else ""
        )
    else:
        reponse_parts = response.split(end_thinking_tag)
        if len(reponse_parts) > 1:
            thinking = reponse_parts[0].replace(beginning_thinking_tag, "").strip()
            actual_response = reponse_parts[1].strip()
        else:
            thinking = ""
            actual_response = response.strip()
    return {"response": actual_response.strip(), "thinking": thinking}
