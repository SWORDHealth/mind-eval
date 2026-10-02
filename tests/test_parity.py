"""The interaction runtime against its goldens (tests/parity.py): the same scripted scenarios must leave the same
files and send the same requests, in the same order. A deliberate behaviour change regenerates them."""

import pytest
from parity import SCENARIOS, golden, run


def _first_difference(want, got, where=""):
    if type(want) is not type(got):
        return f"{where}: {type(want).__name__} -> {type(got).__name__}"
    if isinstance(want, dict):
        for k in sorted(set(want) | set(got)):
            if k not in got or k not in want:
                return f"{where}/{k}: {'missing' if k not in got else 'unexpected'}"
            diff = _first_difference(want[k], got[k], f"{where}/{k}")
            if diff:
                return diff
        return None
    if isinstance(want, list):
        for i, (a, b) in enumerate(zip(want, got)):
            diff = _first_difference(a, b, f"{where}[{i}]")
            if diff:
                return diff
        return f"{where}: {len(want)} items -> {len(got)}" if len(want) != len(got) else None
    return None if want == got else f"{where}: {want!r} -> {got!r}"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_matches_its_golden(name):
    want, got = golden(name), run(name)
    assert want == got, _first_difference(want, got)
