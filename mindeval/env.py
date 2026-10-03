"""Where the models are and how to reach them. This comes from environment variables, never from code.

For each role (PATIENT, COUNSELOR, JUDGE):

    MINDEVAL_<ROLE>_MODEL      litellm model id, e.g. hosted_vllm/<served-name>, openai/<model>
    MINDEVAL_<ROLE>_API_BASE   the endpoint; unset means the provider's own
    MINDEVAL_<ROLE>_API_KEY    the key; unset means litellm's provider variables (OPENAI_API_KEY, ...)

and MINDEVAL_JUDGE_PROMPT, the path to the judge's prompt template. `.env.example` lists them all.
A `.env` file in the working directory is read first; a variable already set in the environment wins.
Keys are never written to any output.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "MINDEVAL_"
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _value(raw: str) -> str:
    """A quoted value up to its closing quote; an unquoted one up to an inline ` #` comment."""
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        return raw[1:end] if end != -1 else raw[1:]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_env_file(path: Path) -> list[str]:
    """`KEY=VALUE` lines (optionally `export KEY=VALUE`, optionally quoted, optionally with a trailing
    `# comment`) into `os.environ`, without overriding anything already set. Returns the names set."""
    loaded = []
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = _LINE.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        key, value = m.group(1), _value(m.group(2))
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


@dataclass(frozen=True)
class Endpoint:
    model: str
    api_base: str | None
    api_key: str | None = field(default=None, repr=False)


def _get(name: str, missing: list[str]) -> str | None:
    value = os.environ.get(PREFIX + name, "").strip()
    if "<" in value and ">" in value:
        missing.append(f"{PREFIX}{name} (still the .env.example placeholder)")
        return None
    return value or None


def endpoint(role: str, missing: list[str]) -> Endpoint | None:
    """The role's model, endpoint and key; a missing or placeholder model or endpoint goes to `missing`."""
    model = _get(f"{role}_MODEL", missing)
    api_base = _get(f"{role}_API_BASE", missing)
    api_key = os.environ.get(f"{PREFIX}{role}_API_KEY", "").strip() or None
    if model is None:
        if not any(m.startswith(f"{PREFIX}{role}_MODEL") for m in missing):
            missing.append(f"{PREFIX}{role}_MODEL")
        return None
    return Endpoint(model=model, api_base=api_base, api_key=api_key)


def optional(name: str, missing: list[str]) -> str | None:
    """A variable that may be unset; one still holding the .env.example placeholder goes to `missing`."""
    return _get(name, missing)
