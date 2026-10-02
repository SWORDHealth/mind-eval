"""The Guard (guard.py), ported from MindSim with its tests: every rule fires when it should, abstains when it
should and mutates nothing; ceilings beat floors; and, switched on, a vetoed move is answered and committed again,
then clamped. No model is involved in the rules themselves."""

import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import _response, scripted

from mindeval import engine, llm
from mindeval.config import CallSpec, load_config
from mindeval.guard import RULES, Guard, Ledger, RuleContext
from mindeval.member import Todos, build_member, member_turn
from mindeval.models import Knobs, MemberRow, Move, TurnRecord

ROOT = Path(__file__).resolve().parents[1]
#: rung -> the move committed to reach it (moves.yaml's canonical table).
BY_LEVEL = {"withhold": "do_disengage", "deflect": "do_deflect", "minimize": "do_minimize",
            "partial": "do_share", "full": "do_go_deeper"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def knobs(concealment="high"):
    return Knobs(concealment_propensity=concealment, frustration_reactivity="medium", session_affect="sadness",
                 verbosity="medium", writing_style="plain")


def mv(level=None, *, move=None, secondary=None, ending=False):
    return Move(reasoning="because", move=move or (BY_LEVEL[level] if level else "do_agree"),
                secondary_move=secondary, end_conversation=ending)


def ledger(levels, *, withholds=0, max_turns=12):
    st = Ledger(max_turns=max_turns)
    st.level_history = list(levels)
    if levels:
        st.max_level = max(levels, key=list(BY_LEVEL).index)
    st.consecutive_withholds = withholds
    return st


def guard(cfg, concealment="high", max_turns=12):
    return Guard(cfg.guard.params(), knobs(concealment), max_turns, cfg.moves)


def fired(verdict):
    return [h.rule for h in verdict.hits]


def test_the_guard_is_off_by_default_and_configures_every_rule(cfg):
    assert cfg.guard.enabled is False
    assert set(cfg.guard.params()) == {r.name for r in RULES}


def test_a_move_with_no_disclosure_signal_bypasses_every_rule(cfg):
    verdict = guard(cfg).check(mv(move="do_express_sadness"), ledger(["withhold", "withhold"], withholds=2), turn=11)
    assert (verdict.vetoed, verdict.hits, verdict.applicable) == (False, [], [])


def test_a_signal_free_move_takes_its_rung_from_the_secondary(cfg):
    g = guard(cfg)
    assert g.level_of(mv(move="do_answer_question")) is None
    assert g.level_of(mv(move="do_answer_question", secondary="do_go_deeper")) == "full"
    assert g.level_of(mv(move="do_minimize", secondary="do_go_deeper")) == "minimize"  # the primary wins
    assert g.check(mv(move="do_answer_question", secondary="do_go_deeper"), Ledger(12), turn=1).vetoed


@pytest.mark.parametrize("concealment, level, history, turn, rule, clamp", [
    ("high", "full", [], 1, "C1_early_disclosure_ceiling", "minimize"),
    ("high", "full", ["minimize"], 6, "C2_ratchet", "partial"),
    ("high", "full", ["full", "partial", "full"], 9, "C3_full_disclosure_budget", "partial"),
    ("high", "minimize", ["minimize", "minimize"], 7, "F1_anti_stonewall", "partial"),
    ("high", "minimize", ["minimize", "deflect"], 8, "F3_session_arc", "partial"),
    ("high", "withhold", ["full"], 6, "F4_non_regression", "partial"),
])
def test_each_rule_fires_and_clamps(cfg, concealment, level, history, turn, rule, clamp):
    verdict = guard(cfg, concealment).check(mv(level), ledger(history), turn=turn)
    assert rule in fired(verdict) and verdict.clamp_to == clamp and verdict.vetoed


def test_f2_refuses_a_second_consecutive_withhold(cfg):
    verdict = guard(cfg).check(mv("withhold"), ledger(["withhold"], withholds=1), turn=7)
    assert "F2_responsiveness" in fired(verdict) and verdict.clamp_to == "minimize"


@pytest.mark.parametrize("concealment, level, history, turn, rule", [
    ("high", "partial", [], 4, "C1_early_disclosure_ceiling"),           # past the staircase
    ("low", "full", [], 1, "C1_early_disclosure_ceiling"),               # not restricted at all
    ("high", "partial", [], 6, "C2_ratchet"),                            # nothing leveled yet
    ("low", "full", ["full", "full", "full"], 9, "C3_full_disclosure_budget"),
    ("high", "partial", ["full", "full"], 7, "F1_anti_stonewall"),       # the window got somewhere
    ("high", "minimize", ["minimize"], 7, "F1_anti_stonewall"),          # window not full yet
    ("high", "minimize", ["partial"], 8, "F3_session_arc"),              # already reached partial
    ("high", "minimize", ["minimize", "deflect"], 5, "F3_session_arc"),  # before its threshold
])
def test_each_rule_abstains_where_it_is_not_in_force(cfg, concealment, level, history, turn, rule):
    assert rule not in guard(cfg, concealment).check(mv(level), ledger(history), turn=turn).applicable


def test_f4_lets_a_visibly_hard_turn_motivate_a_retreat(cfg):
    assert "F4_non_regression" not in guard(cfg).check(
        mv("withhold", secondary="do_express_frustration"), ledger(["full"]), turn=6).applicable
    st = ledger(["full"])
    st.turns.append(TurnRecord(session_id="s", run_id="r", turn=5, seed=1, turn_seed=1, fingerprint="f",
                               knobs=knobs(), final_move=Move(reasoning="r", move="do_express_shame")))
    assert "F4_non_regression" not in guard(cfg).check(mv("withhold"), st, turn=6).applicable
    assert guard(cfg).check(mv("partial"), ledger(["full"]), turn=6).hits == []  # one step down is fine


def test_a_member_leaving_is_spared_the_floors_but_not_the_ceilings(cfg):
    assert not guard(cfg).check(mv("minimize", ending=True), ledger(["minimize", "minimize"]), turn=8).vetoed
    assert guard(cfg).check(mv("full", ending=True), Ledger(12), turn=1).vetoed


def test_a_floor_asking_above_the_binding_ceiling_stands_down(cfg):
    # F1's target is partial, but C1 caps turn 3 at minimize: "give more" is unsatisfiable, so F1 abstains.
    verdict = guard(cfg).check(mv("deflect"), ledger(["minimize", "minimize"]), turn=3)
    assert "F1_anti_stonewall" not in verdict.applicable and not verdict.vetoed
    assert "F1_anti_stonewall" in fired(guard(cfg).check(mv("minimize"), ledger(["minimize", "minimize"]), turn=7))


def test_a_ceiling_beats_a_floor_and_only_it_reaches_the_model(cfg):
    params = cfg.guard.params()
    params["F3_session_arc"] = params["F3_session_arc"].model_copy(update={"required_level": "full"})
    verdict = Guard(params, knobs("medium"), 5, cfg.moves).check(mv("partial"), ledger(["deflect", "deflect"]),
                                                                  turn=3)
    assert verdict.conflict and verdict.clamp_to == "minimize"
    assert "do_minimize" in verdict.feedback and "do_go_deeper" not in verdict.feedback
    assert "C1" not in verdict.feedback and "ceiling" not in verdict.feedback.lower()


def test_the_clamp_keeps_the_reasoning_and_drops_the_secondary(cfg):
    g, move = guard(cfg), mv("full", secondary="do_express_shame")
    clamped = g.clamp(move, g.check(move, Ledger(12), turn=1))
    assert (clamped.move, clamped.reasoning, clamped.secondary_move) == ("do_minimize", "because", None)


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name)
def test_evaluating_a_rule_leaves_the_ledger_untouched(cfg, rule):
    st = ledger(["minimize", "partial"], withholds=1)
    snapshot = copy.deepcopy(st)
    ctx = RuleContext(move=mv("full"), level="full", state=st, knobs=knobs(), turn=6, max_turns=12,
                      params=cfg.guard.params()[rule.name], canonical=cfg.moves.canonical_move_by_level,
                      negative_expressive=frozenset(cfg.moves.negative_expressive))
    rule.fn(ctx)
    rule.fn(ctx)
    assert st == snapshot


def test_a_signal_free_turn_does_not_reset_the_withhold_count():
    st = Ledger(12)
    for move, level in (("do_disengage", "withhold"), ("do_express_sadness", None), ("do_disengage", "withhold")):
        st.record(TurnRecord(session_id="s", run_id="r", turn=0, seed=1, turn_seed=1, fingerprint="f",
                             knobs=knobs(), final_move=Move(reasoning="r", move=move), inferred_level=level))
    assert st.consecutive_withholds == 2 and st.level_history == ["withhold", "withhold"]


def test_no_feedback_assumes_how_many_sessions_there_have_been(cfg):
    for name, params in cfg.guard.params().items():
        for claim in ("just met", "single conversation", "first time", "never talked", "new to"):
            assert claim not in params.feedback.lower(), name


# -- switched on, in a member turn ----------------------------------------------------------------

def test_a_vetoed_move_is_answered_committed_again_and_then_clamped(cfg, monkeypatch):
    """A high-concealment member going all the way in turn 1: C1 vetoes it, the feedback comes back as the tool
    result, the member commits again (with its own seed) and is vetoed again, and after max_retries the move is
    clamped to do_minimize. Every verdict is on the turn, and the words are still the member's."""
    requests = []

    def completion(**kw):
        requests.append(kw)
        if any(t["function"]["name"] == "commit_move" for t in kw.get("tools") or []):
            return _response(tool_call=json.dumps({"reasoning": "all of it", "move": "do_go_deeper"}),
                             call_id=f"c{len(requests)}")
        return _response("ok, a bit of it")

    monkeypatch.setattr(llm, "_LIB", scripted(completion))
    railed = replace(cfg, guard=cfg.guard.model_copy(update={"enabled": True}))
    row = MemberRow.model_validate_json((ROOT / "data/profiles.jsonl").read_text().splitlines()[0])
    k = row.knobs.model_copy(update={"concealment_propensity": "high"})
    spec = CallSpec(model="hosted_vllm/p", api_base=None, temperature=1.0, max_tokens=None, timeout=5,
                    max_retries=2)
    member = build_member(railed, row, row.situation, k, spec, models=engine.facades(spec, spec))
    member.said(row.situation.opening_message)
    member.hears("What brings you in?")
    g = Guard.from_config(railed, k, 10)

    def blank(turn, kind):
        return TurnRecord(session_id="s", run_id="r", turn=turn, seed=3, turn_seed=11, fingerprint="f", knobs=k)

    rec = asyncio.run(member_turn(railed, member, "What brings you in?", {}, 1, "s", 3, blank, Todos(row.situation),
                                  guard=g, ledger=Ledger(10)))
    assert rec.final_move.move == "do_minimize" and rec.inferred_level == "minimize"
    assert {"guard_veto", "guard_fallback"} <= set(rec.flags)
    report = rec.meta["guard"]
    assert report["raw_move"]["move"] == "do_go_deeper" and report["raw_inferred_level"] == "full"
    assert report["retries"] == 2 and report["fallback"] is True and len(report["verdicts"]) == 3
    assert [c.get("guard_retry") for c in rec.calls if c["name"] == "commit_move"] == [None, 1, 2]
    feedback = report["verdicts"][0]["feedback"]
    tool_results = [m["content"] for m in member.convo if m["role"] == "tool"]
    assert tool_results[:2] == [feedback, feedback] and tool_results[2].startswith("accepted")
    commits = [r for r in requests if any(t["function"]["name"] == "commit_move" for t in r.get("tools") or [])]
    assert len(commits) == 3 and len({r["seed"] for r in commits}) == 3  # each recommit with its own seed
    assert rec.utterance == "ok, a bit of it"
