"""Test helpers shared across the test modules."""

import os
from types import SimpleNamespace

from jsonargparse import CLI

from mindeval.scripts import generate_interactions, generate_judgments


def _clear_env(monkeypatch):
    """Removes every MINDEVAL_* variable, so a test starts clean regardless of what the shell or an
    earlier test exported."""
    for name in [k for k in os.environ if k.startswith("MINDEVAL_")]:
        monkeypatch.delenv(name)


def _response(content=None, tool_call=None, finish="stop", call_id="call_1", **extra):
    """A litellm completion as mindeval reads one: `content`, or a `commit_move` call carrying
    `tool_call` as its arguments; `extra` lands on the message (e.g. reasoning_items)."""
    calls = None
    if tool_call is not None:
        calls = [SimpleNamespace(id=call_id, function=SimpleNamespace(name="commit_move", arguments=tool_call))]
    message = SimpleNamespace(content=content, tool_calls=calls, **extra)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], model="fake", id="r")


def scripted(completion):
    """A litellm whose one answer is `completion(**request)`, sync (the judge) and awaited (the interactions)."""
    async def acompletion(**kw):
        return completion(**kw)
    return SimpleNamespace(completion=completion, acompletion=acompletion)


async def no_pause(*args):
    """A backoff that does not wait."""


def interactions(*argv):
    try:
        return CLI([generate_interactions.main], as_positional=False, args=list(argv))
    except SystemExit as e:  # a malformed flag: jsonargparse's own exit code, same shape as InputError's
        return e.code


def judgments(*argv):
    try:
        return CLI([generate_judgments.main], as_positional=False, args=list(argv))
    except SystemExit as e:  # a malformed flag: jsonargparse's own exit code, same shape as InputError's
        return e.code
