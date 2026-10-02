"""utils.py: deep_merge's recursive merge, and hold_lock's contention-vs-real-error distinction."""

import errno
import json

import pytest
from conftest import _clear_env, interactions, judgments

from mindeval.config import load_config
from mindeval.scripts import generate_interactions
from mindeval.utils import InputError, deep_merge, hold_lock


def test_deep_merge_merges_nested_dicts_recursively():
    base = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "other": 1}, "a": 1}
    override = {"extra_body": {"chat_template_kwargs": {"foo": "bar"}}, "b": 2}
    merged = deep_merge(base, override)
    assert merged == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "foo": "bar"}, "other": 1},
                      "a": 1, "b": 2}
    # neither argument is mutated
    assert base == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "other": 1}, "a": 1}
    assert override == {"extra_body": {"chat_template_kwargs": {"foo": "bar"}}, "b": 2}


def test_deep_merge_a_non_dict_override_wins_outright():
    assert deep_merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}


def test_patient_params_deep_merge_keeps_the_vllm_providers_own_extra_body(tmp_path, monkeypatch):
    """--patient_params is layered onto config/patient.yaml's provider params with deep_merge, not a
    shallow **-merge: an unrelated extra_body override must not drop the provider's own
    chat_template_kwargs.enable_thinking. Checked on the CallSpec generate_interactions.py builds."""
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    monkeypatch.setenv("MINDEVAL_PATIENT_MODEL", "hosted_vllm/some-model")
    monkeypatch.setenv("MINDEVAL_COUNSELOR_MODEL", "openai/counselor")
    seen = {}

    async def preflight(cfg, row, counselor, patient, models):  # the first to use the patient spec; no call
        seen["params"] = patient.params
        raise InputError("stop here")

    monkeypatch.setattr(generate_interactions, "preflight", preflight)
    assert interactions("--output_dir", str(tmp_path / "run"), "--members", "1",
                        "--patient_params", '{"extra_body": {"something_else": true}}') == 2
    provider = load_config().patient.provider("hosted_vllm/some-model")
    assert provider.params["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert seen["params"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}, "something_else": True}


def test_hold_lock_refuses_contention_as_an_input_error(tmp_path):
    with hold_lock(tmp_path):
        with pytest.raises(InputError, match="another mindeval process is running"):
            with hold_lock(tmp_path):
                pass  # never reached; LOCK_NB refuses immediately, no second process needed


def test_hold_lock_does_not_disguise_a_real_os_error_as_contention(tmp_path, monkeypatch):
    """Only LOCK_NB's own contention error (EWOULDBLOCK/EAGAIN, raised as BlockingIOError) means
    "another process is running". Anything else — ENOLCK on NFS, say — is reported as itself, and
    refused: exit 2, not the traceback and exit 1 that both scripts document as "rerun"."""
    import fcntl

    def raises_enolck(handle, flags):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", raises_enolck)
    with pytest.raises(InputError, match="could not lock .*No locks available") as exc_info:
        with hold_lock(tmp_path):
            pass
    assert "another mindeval process" not in str(exc_info.value)
    assert exc_info.value.__cause__.errno == errno.ENOLCK


def test_both_scripts_refuse_a_filesystem_without_locks(tmp_path, monkeypatch, capsys):
    import fcntl

    def raises_enolck(handle, flags):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    for name, value in {"PATIENT_MODEL": "hosted_vllm/patient", "COUNSELOR_MODEL": "openai/counselor",
                        "JUDGE_MODEL": "openai/judge"}.items():
        monkeypatch.setenv(f"MINDEVAL_{name}", value)
    monkeypatch.setattr(fcntl, "flock", raises_enolck)
    out = tmp_path / "run"
    assert interactions("--output_dir", str(out), "--members", "1") == 2
    (out / "run.json").write_text(json.dumps({"members": 1, "sessions": 1}))
    assert judgments("--output_dir", str(out), "--judge_version", "mindeval1") == 2
    assert capsys.readouterr().err.count("could not lock") == 2
