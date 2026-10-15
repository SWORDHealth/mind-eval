"""The native path (`--usersim_models`) against the same goldens as the litellm path (tests/parity.py).

Every scenario runs through UserSim's own model layer, with the scripted model answering at Data Designer's HTTP
layer. The conversations must be the ones the goldens hold: the same exit codes, the same requests in the same
order with the same messages, and the same files. What may differ is only what the path itself changes:

- a counselor's OpenAI Responses reasoning items, which UserSim's model layer does not carry (a counselor that needs
  them stays on the litellm path);
- each call's response metadata (`model`, `response_id`), which comes from another client;
- what the run records about the two sides' settings (endpoint, temperature, params, max_tokens), which on this
  path are the models file's, not mindeval's flags.
"""

import pytest
from parity import COUNSELOR, SCENARIOS, golden, run
from test_parity import _first_difference

#: Keys whose values the native path legitimately changes in each call's recorded response metadata (`meta`).
PATH_KEYS = {"reasoning_items", "model", "response_id", "usage"}
#: The run's record of the two sides' settings (run.json, and each trace's setup line).
SETTINGS = {"api_base", "temperature", "params", "max_tokens"}


def _without(obj, keys: set[str], under: str | None = None):
    """`obj` without `keys`; with `under`, only below a key of that name."""
    if isinstance(obj, dict):
        return {k: _without(v, keys, under if under and k != under else None)
                for k, v in obj.items() if not (k in keys and under is None)}
    if isinstance(obj, list):
        return [_without(v, keys, under) for v in obj]
    return obj


def _comparable(result: dict) -> dict:
    files = _without(result["files"], PATH_KEYS, under="meta")
    files = _without(files, {"completion_tokens"}, under="hygiene_attempts")  # from usage, as above
    for side in ("counselor", "patient", "member"):
        files = _without(files, SETTINGS, under=side)
    files.get("run.json", {}).pop("usersim_models", None)
    requests = [{"n": r["messages"]["n"],
                 "tail": [{k: v for k, v in m.items() if k != "reasoning_items"} for m in r["messages"]["tail"]]}
                for r in result["requests"]]
    return {"codes": result["codes"], "requests": requests, "files": files}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_native_path_runs_the_same_conversations(name):
    want, got = _comparable(golden(name)), _comparable(run(name, native=True))
    assert want == got, _first_difference(want, got)


def test_the_native_path_sends_what_the_models_file_and_mindeval_ask_for():
    """Through UserSim's model layer: the patient's move is forced and its models-file settings sent; the counselor
    gets its own earlier reasoning back (replay_reasoning), and the patient never sees it. The run records the
    settings each side was sent, not mindeval's defaults."""
    result = run("two_members", native=True)
    sent, record = result["sent"], result["files"]["run.json"]
    counselor = [r for r in sent if r["model"] == COUNSELOR]
    patient = [r for r in sent if r["model"] != COUNSELOR]
    commits = [r for r in patient if "commit_move" in (r.get("tools") or {}).get("names", [])]

    assert commits and all(r["tool_choice"] == {"type": "function", "function": {"name": "commit_move"}}
                           for r in commits)
    assert all(r["chat_template_kwargs"] == {"enable_thinking": True} for r in patient)
    assert any("reasoning_content" in m for r in counselor for m in r["messages"]["tail"] if m["role"] == "assistant")
    assert not any("reasoning_content" in m for r in patient for m in r["messages"]["tail"])
    for side, requests in (("patient", patient), ("counselor", counselor)):
        assert {(r["temperature"], r["top_p"]) for r in requests} == {(record[side]["temperature"],
                                                                       record[side]["params"]["top_p"])}
    assert record["patient"]["params"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
