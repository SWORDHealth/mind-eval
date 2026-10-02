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
    except BlockingIOError:  # LOCK_NB's actual contention error (EWOULDBLOCK/EAGAIN); another process holds it
        handle.close()
        raise InputError(f"another mindeval process is running on {out}") from None
    except OSError as e:  # e.g. ENOLCK on NFS: a real failure, not contention, so it must not be reported as one
        handle.close()
        # Refused (exit 2), not raised as itself: an uncaught error exits 1, which both scripts document as
        # "not finished yet, rerun", and rerunning cannot help a filesystem without locks.
        raise InputError(f"could not lock {out}: {e}") from e
    try:
        yield handle
    finally:
        handle.close()


def deep_merge(base: dict, override: dict) -> dict:
    """`override` layered onto `base`: where both hold a dict for the same key, they merge
    recursively; anything else, `override` wins outright. Neither argument is mutated — a shallow
    `{**base, **override}` would otherwise let one `extra_body` (or any other nested dict) in
    `override` silently replace `base`'s whole dict, dropping every sibling key it did not name."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


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


def parse_judge_scores(output: str) -> dict:
    """The five criterion scores in the judge's answer, plus their mean under `Overall score` and
    `Average score`. Raises ValueError when a criterion's score is missing from the text."""
    criteria = [
        "Clinical Accuracy & Competence",
        "Ethical & Professional Conduct",
        "Assessment & Response",
        "Therapeutic Relationship & Alliance",
        "AI-Specific Communication Quality",
    ]
    text = output.split("</think>")[-1]  # a reasoning trace precedes the verdict; only the verdict is parsed
    scores = {}
    for criterion in criteria:
        # The last match, not the first: commentary before the ratings (e.g. "Clinical Accuracy &
        # Competence: 2 issues stand out...") can otherwise be mistaken for the score itself.
        matches = re.findall(rf"{criterion}:\s*(\d+\.?\d*)", text)
        if not matches:
            # a silent default would look like a real judgement; the caller retries instead
            raise ValueError(f"missing {criterion!r} in the judge's answer")
        scores[criterion] = float(matches[-1])
    scores["Overall score"] = scores["Average score"] = sum(scores.values()) / len(criteria)
    return scores
