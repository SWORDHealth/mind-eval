"""Test helpers shared by test_smoke.py and test_judge.py."""

import os

from jsonargparse import CLI

from mindeval.scripts import generate_interactions, generate_judgments


def _clear_env(monkeypatch):
    """Removes every MINDEVAL_* variable, so a test starts clean regardless of what the shell or an
    earlier test exported."""
    for name in [k for k in os.environ if k.startswith("MINDEVAL_")]:
        monkeypatch.delenv(name)


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
