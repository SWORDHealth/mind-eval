"""Seeds are drawn from the pinned SEED_FINGERPRINT namespace, not cfg.fingerprint: a byte edit to
config/*.yaml or prompts/*.j2 (a comment, a rename) must not re-draw the benchmark. See config.py."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from conftest import _response

from mindeval import llm
from mindeval.arc import member_seed, run_arc
from mindeval.config import CallSpec, load_config
from mindeval.models import MemberRow
from mindeval.session import Counselor, counselor_template

ROOT = Path(__file__).resolve().parents[1]
COUNSELOR_MODEL = "openai/counselor"
ROW = next(MemberRow.model_validate_json(line) for line in
          (ROOT / "data/profiles.jsonl").read_text().splitlines() if '"m000119"' in line)


def _fake(**kw):
    first = kw["messages"][0]["content"]
    if isinstance(kw.get("tool_choice"), dict):
        return _response(tool_call=json.dumps({"reasoning": "ok", "move": "do_answer_question"}))
    if kw["model"] == COUNSELOR_MODEL:
        return _response("How has your week been?")
    if "running log" in first:
        return _response("- Member talked about the week.")
    if "Answer with JSON only" in first:
        next_situation = {"event": "The week went on.", "automatic_thoughts": ["same as before"],
                          "behaviors": ["stayed in"], "physical": [], "goal": "talk it through",
                          "checkpoints": ["the money"], "opening_message": "hey, again"}
        return _response(json.dumps(next_situation))
    return _response("not really")


def _run(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(llm, "_LIB", SimpleNamespace(completion=_fake))
    counselor = Counselor(spec=CallSpec(model=COUNSELOR_MODEL, api_base=None, temperature=1.0, max_tokens=None,
                                        timeout=60, max_retries=1),
                          template=counselor_template((ROOT / "examples/counselor_system.j2").read_text()))
    patient = CallSpec(model="hosted_vllm/patient", api_base=None, temperature=1.0, max_tokens=None,
                       timeout=60, max_retries=1,
                       params={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}})
    return run_arc(cfg, ROW, counselor, patient=patient, sessions=2, max_turns=1,
                   arc_seed=member_seed(ROW.member_id, 0), out=tmp_path)


def test_arc_id_and_run_id_ignore_cfg_fingerprint(tmp_path, monkeypatch):
    """The regression this pins: 978ae09 made cfg.fingerprint (a hash of config/prompt bytes) seed
    material, so a comment edit re-drew every arc under the same --seed. arc_id/run_id must come from
    SEED_FINGERPRINT alone; cfg.fingerprint may still differ (it is still an identity field)."""
    cfg = load_config()
    edited = replace(cfg, fingerprint="0" * 16)  # simulates a byte edit that leaves the data alone
    assert edited.fingerprint != cfg.fingerprint

    real = _run(tmp_path / "real", monkeypatch, cfg)
    after_edit = _run(tmp_path / "after_edit", monkeypatch, edited)

    assert real.arc_id == after_edit.arc_id
    assert [e.run_id for e in real.episodes] == [e.run_id for e in after_edit.episodes]
    assert [e.session_id for e in real.episodes] == [e.session_id for e in after_edit.episodes]
    # the identity field on arc.json still tracks the real, unedited cfg.fingerprint
    assert real.fingerprint == cfg.fingerprint and after_edit.fingerprint == edited.fingerprint


def test_m000119_matches_mindeval2_a18de8be(tmp_path, monkeypatch):
    """Hardcoded from a run of MindEval2 @ a18de8be (config/prompts unedited) over the same profile,
    sessions and seed; the parity harness (see the workflow's tmp/parity/) checks all 50 members, this
    pins one so a future regression fails a plain `pytest`."""
    arc = _run(tmp_path, monkeypatch, load_config())
    assert arc.arc_id == "ee02be63feb3"
    assert arc.shape.anchor_weekday == 4 and arc.shape.anchor_hour == 20.254629527198947
    assert arc.shape.gaps_hours == [8.748854344668784]  # the one inter-session gap at 2 sessions
    assert arc.episodes[0].run_id == "75cb0707986d"

    session_1 = json.loads((tmp_path / "ep000" / "trace.jsonl").read_text().splitlines()[-1])
    lines = session_1["member_system_prompt"].splitlines()
    assert "It is Friday evening." in lines  # not "Sunday evening": the 978ae09 regression's own example

