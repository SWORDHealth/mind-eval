"""The Guard: pure rules over each committed move, off unless `config/guard.yaml` turns it on. Ported from MindSim's
session harness (mindsim/harness/guard), whose design notes it keeps.

It infers the disclosure rung a move carries from `moves.yaml`'s `level_by_move` (a move outside that table carries
none, and no rule applies to the turn), then asks seven rules whether that rung fits this member at this point of the
session. Ceilings cap how much a member says (C1 early in the session by concealment, C2 one step deeper per turn,
C3 a budget of fully open turns); floors keep a session from going nowhere (F1 no stonewalling across three turns,
F2 no two outright refusals in a row, F3 something real by 60% of the session, F4 no unmotivated backsliding).

**No model calls, no text inspection, no exceptions.** A rule sees the committed `Move`, the session's `Ledger`, the
knobs, the turn and its own parameters, and may change none of them; anything that needs the spoken text is a
retrospective evaluation, not a rule.

A vetoed move is answered with the winning rule's feedback as the tool result, and the member commits again
(`member.member_turn`); once `max_retries` are spent the move is clamped to the canonical move of the resolved rung.

**Ceilings beat floors, before anything fires.** A floor whose target sits above the cap a ceiling enforces this turn
is unsatisfiable and stands down (pre-empted, counted as abstaining); asking for both in one breath produced a
whipsaw of "give more" and "give less" on consecutive resamples. A ceiling and a floor that still collide resolve to
the ceiling, flagged `guard_conflict`: a breached ceiling mislabels a transcript, a breached floor only makes it dull.
Only the winning constraint reaches the model, and its feedback never names a rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Literal, get_args

from pydantic import Field, model_validator

from mindeval.models import DisclosureLevel, KnobLevel, Knobs, Move, TurnRecord, _Strict

LEVELS: tuple[str, ...] = get_args(DisclosureLevel)
_RANK = {level: i for i, level in enumerate(LEVELS)}
RuleKind = Literal["ceiling", "floor"]


def rank(level: DisclosureLevel) -> int:
    return _RANK[level]


def step(level: DisclosureLevel, n: int) -> DisclosureLevel:
    """`n` steps along the ladder, saturating at both ends: one step up from `full` is `full`."""
    return LEVELS[max(0, min(len(LEVELS) - 1, rank(level) + n))]


class RuleHit(_Strict):
    """One rule firing. `reason` is for the trace; `feedback` is what the model is shown, and names no rule."""

    rule: str
    kind: RuleKind
    reason: str
    feedback: str
    clamp_to: DisclosureLevel


class Verdict(_Strict):
    vetoed: bool
    hits: list[RuleHit] = Field(default_factory=list)
    #: Rules in force this turn, whether or not they fired: the honest denominator of a firing rate.
    applicable: list[str] = Field(default_factory=list)
    #: The resolved clamp target after ceilings beat floors.
    clamp_to: DisclosureLevel | None = None
    conflict: bool = False
    #: The winning constraint's feedback, the tool result of a veto.
    feedback: str = ""


# -- the session's ledger --------------------------------------------------------------------------

@dataclass
class Ledger:
    """What the Guard reasons over: the rungs the session has produced. `record` is the only write, so every rule
    is a pure function of it."""

    max_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    #: Every inferred rung, in order; turns whose move carries none do not appear.
    level_history: list[DisclosureLevel] = field(default_factory=list)
    max_level: DisclosureLevel | None = None
    consecutive_withholds: int = 0

    def ever_reached(self, level: DisclosureLevel) -> bool:
        return self.max_level is not None and rank(self.max_level) >= rank(level)

    def full_count(self) -> int:
        return sum(1 for level in self.level_history if level == "full")

    def record(self, rec: TurnRecord) -> None:
        self.turns.append(rec)
        level = rec.inferred_level
        if level is None:
            # Neither advances the history nor touches the withhold count: resetting it here would let a member
            # dodge F2 by alternating a withhold with an expressive move.
            return
        self.level_history.append(level)
        if self.max_level is None or rank(level) > rank(self.max_level):
            self.max_level = level
        self.consecutive_withholds = self.consecutive_withholds + 1 if level == "withhold" else 0


# -- the rules -------------------------------------------------------------------------------------

class RuleParams(_Strict):
    kind: RuleKind
    enabled: bool = True
    #: Shown to the model on a retry; `{ceiling}` / `{floor}` are filled with the canonical move of the clamp rung.
    feedback: str


@dataclass(frozen=True)
class RuleContext:
    move: Move
    #: The rung inferred from the move; a rule never sees a turn whose move carries none.
    level: DisclosureLevel
    state: Ledger
    knobs: Knobs
    #: The turn: 0 is the member's opening, k the member's answer to the counselor's k-th reply.
    turn: int
    #: The session's length in member turns, the opening included (MindSim's measure): the cap on counselor
    #: replies, plus one.
    max_turns: int
    params: Any
    canonical: dict[str, str]
    negative_expressive: frozenset[str]


@dataclass(frozen=True)
class RuleResult:
    applicable: bool
    hit: RuleHit | None = None


ABSTAIN = RuleResult(applicable=False)
SATISFIED = RuleResult(applicable=True)


def _fire(ctx: RuleContext, name: str, clamp_to: DisclosureLevel, reason: str) -> RuleResult:
    target = ctx.canonical[clamp_to]  # the member speaks in moves, so the feedback names one
    return RuleResult(applicable=True, hit=RuleHit(
        rule=name, kind=ctx.params.kind, reason=reason, clamp_to=clamp_to,
        feedback=ctx.params.feedback.format(ceiling=target, floor=target)))


class C1Row(_Strict):
    before_turn: int = Field(ge=1)
    max_level: DisclosureLevel


class C1Params(RuleParams):
    table: dict[KnobLevel, list[C1Row]]


def c1_cap(state: Ledger, knobs: Knobs, turn: int, max_turns: int, params: C1Params) -> DisclosureLevel | None:
    row = next((r for r in params.table.get(knobs.concealment_propensity, []) if turn < r.before_turn), None)
    return row.max_level if row else None


def c1_early_disclosure_ceiling(ctx: RuleContext) -> RuleResult:
    """How much this person says at all, this early: a concealing person does not empty out in turn one."""
    cap = c1_cap(ctx.state, ctx.knobs, ctx.turn, ctx.max_turns, ctx.params)
    if cap is None:
        return ABSTAIN  # past every step of the staircase, or a concealment level it does not restrict
    if rank(ctx.level) <= rank(cap):
        return SATISFIED
    return _fire(ctx, "C1_early_disclosure_ceiling", cap,
                 f"turn {ctx.turn} at concealment={ctx.knobs.concealment_propensity}: {ctx.move.move} infers "
                 f"{ctx.level}, over {cap}")


class C2Params(RuleParams):
    max_steps_per_turn: int = Field(ge=1)


def c2_ratchet(ctx: RuleContext) -> RuleResult:
    """How far the session deepens in one turn, from the deepest rung it has reached. Abstains before anything
    leveled has happened: treating "nothing yet" as `withhold` would cap every first answer at `deflect`."""
    p: C2Params = ctx.params
    prev = ctx.state.max_level
    if prev is None:
        return ABSTAIN
    if rank(ctx.level) <= rank(prev) + p.max_steps_per_turn:
        return SATISFIED
    return _fire(ctx, "C2_ratchet", step(prev, p.max_steps_per_turn),
                 f"session max is {prev}; {ctx.move.move} infers {ctx.level}, more than "
                 f"{p.max_steps_per_turn} step(s) up in one turn")


class C3Params(RuleParams):
    #: null = unlimited.
    budget_by_concealment: dict[KnobLevel, int | None]
    clamp_to: DisclosureLevel


def c3_cap(state: Ledger, knobs: Knobs, turn: int, max_turns: int, params: C3Params) -> DisclosureLevel | None:
    budget = params.budget_by_concealment.get(knobs.concealment_propensity)
    return None if budget is None or state.full_count() < budget else params.clamp_to


def c3_full_disclosure_budget(ctx: RuleContext) -> RuleResult:
    """How many turns go all the way down in one session."""
    p: C3Params = ctx.params
    budget = p.budget_by_concealment.get(ctx.knobs.concealment_propensity)
    if budget is None:
        return ABSTAIN
    if ctx.level != "full" or ctx.state.full_count() < budget:
        return SATISFIED
    return _fire(ctx, "C3_full_disclosure_budget", p.clamp_to,
                 f"{ctx.state.full_count()} full turn(s) already, budget {budget} at "
                 f"concealment={ctx.knobs.concealment_propensity}")


class F1Params(RuleParams):
    window_probes: int = Field(ge=2)
    clamp_step_up: int = Field(ge=1)
    #: Stands down when any turn in the window is at or above this: stonewalling is flatness at the bottom.
    stand_down_at: DisclosureLevel = "partial"


def f1_anti_stonewall(ctx: RuleContext) -> RuleResult:
    """Across N leveled turns stuck at the bottom of the ladder, the rung must rise at least once."""
    p: F1Params = ctx.params
    window = (ctx.state.level_history + [ctx.level])[-p.window_probes:]
    if len(window) < p.window_probes or any(rank(lvl) >= rank(p.stand_down_at) for lvl in window):
        return ABSTAIN
    if any(rank(b) > rank(a) for a, b in zip(window, window[1:])):
        return SATISFIED
    return _fire(ctx, "F1_anti_stonewall", step(max(window, key=rank), p.clamp_step_up),
                 f"flat across {p.window_probes} leveled turns: {window}")


class F2Params(RuleParams):
    max_consecutive_withholds: int = Field(ge=1)
    clamp_to: DisclosureLevel


def f2_responsiveness(ctx: RuleContext) -> RuleResult:
    """No two consecutive outright refusals."""
    p: F2Params = ctx.params
    if ctx.level != "withhold" or ctx.state.consecutive_withholds < p.max_consecutive_withholds:
        return SATISFIED
    return _fire(ctx, "F2_responsiveness", p.clamp_to,
                 f"withhold after {ctx.state.consecutive_withholds} consecutive withhold(s)")


class F3Params(RuleParams):
    fraction_of_max_turns: float = Field(gt=0, le=1)
    required_level: DisclosureLevel


def f3_session_arc(ctx: RuleContext) -> RuleResult:
    """Nothing at `required_level` by this fraction of the session: the next leveled turn is lifted to it."""
    p: F3Params = ctx.params
    threshold = ceil(p.fraction_of_max_turns * ctx.max_turns)
    if ctx.turn < threshold or ctx.state.ever_reached(p.required_level):
        return ABSTAIN
    if rank(ctx.level) >= rank(p.required_level):
        return SATISFIED
    return _fire(ctx, "F3_session_arc", p.required_level,
                 f"turn {ctx.turn} >= {threshold} and nothing has reached {p.required_level}")


class F4Params(RuleParams):
    max_steps_below_max: int = Field(ge=0)
    #: Backsliding is exempt right after visible negative affect: this turn's secondary move, or the last turn's.
    expressive_exemption: bool = True


def f4_non_regression(ctx: RuleContext) -> RuleResult:
    """Taking back what was said stays possible, but has to be motivated by something visible."""
    p: F4Params = ctx.params
    if p.expressive_exemption:
        recent = {ctx.move.secondary_move}
        for turn in ctx.state.turns[-1:]:
            if turn.final_move is not None:
                recent |= {turn.final_move.move, turn.final_move.secondary_move}
        if recent & ctx.negative_expressive:
            return ABSTAIN
    prev = ctx.state.max_level
    if prev is None:
        return ABSTAIN
    floor_ = step(prev, -p.max_steps_below_max)
    if rank(ctx.level) >= rank(floor_):
        return SATISFIED
    return _fire(ctx, "F4_non_regression", floor_,
                 f"session reached {prev}; {ctx.move.move} infers {ctx.level}, more than "
                 f"{p.max_steps_below_max} step(s) below, with no expressive move to motivate it")


@dataclass(frozen=True)
class Rule:
    name: str
    kind: RuleKind
    fn: Callable[[RuleContext], RuleResult]
    params: type[RuleParams]


#: Evaluation order: ceilings, then floors.
RULES: tuple[Rule, ...] = (
    Rule("C1_early_disclosure_ceiling", "ceiling", c1_early_disclosure_ceiling, C1Params),
    Rule("C2_ratchet", "ceiling", c2_ratchet, C2Params),
    Rule("C3_full_disclosure_budget", "ceiling", c3_full_disclosure_budget, C3Params),
    Rule("F1_anti_stonewall", "floor", f1_anti_stonewall, F1Params),
    Rule("F2_responsiveness", "floor", f2_responsiveness, F2Params),
    Rule("F3_session_arc", "floor", f3_session_arc, F3Params),
    Rule("F4_non_regression", "floor", f4_non_regression, F4Params),
)
#: The caps the identity ceilings enforce this turn, computable before any move exists; floors above them stand
#: down. C2 is absent on purpose: floor clamps may breach the ratchet.
CEILING_CAPS = {"C1_early_disclosure_ceiling": c1_cap, "C3_full_disclosure_budget": c3_cap}


class GuardFile(_Strict):
    """config/guard.yaml."""

    enabled: bool = False
    #: Recommits after a veto before the move is clamped.
    max_retries: int = Field(default=2, ge=0)
    #: A member leaving is not clamped into disclosing on the way out; ceilings still apply.
    skip_floors_on_terminate: bool = True
    rules: dict[str, dict]

    @model_validator(mode="after")
    def _every_rule_once(self) -> GuardFile:
        named = {r.name for r in RULES}
        if set(self.rules) != named:
            raise ValueError(f"guard.yaml must configure exactly the rules {sorted(named)}; "
                             f"missing {sorted(named - set(self.rules))}, unknown {sorted(set(self.rules) - named)}")
        for rule in RULES:
            rule.params.model_validate(self.rules[rule.name])
        return self

    def params(self) -> dict[str, RuleParams]:
        return {r.name: r.params.model_validate(self.rules[r.name]) for r in RULES}

    def setup(self) -> dict:
        """As a session's setup records it."""
        return {"enabled": self.enabled, "max_retries": self.max_retries,
                "skip_floors_on_terminate": self.skip_floors_on_terminate, "rules": self.rules}


# -- the Guard -------------------------------------------------------------------------------------

class Guard:
    def __init__(self, params: dict[str, RuleParams], knobs: Knobs, max_turns: int, moves, *,
                 skip_floors_on_terminate: bool = True) -> None:
        """`moves` is the loaded `config.MovesFile`; `params` the rules in force (none: every move passes)."""
        self.params, self.knobs, self.max_turns, self.moves = params, knobs, max_turns, moves
        self.skip_floors_on_terminate = skip_floors_on_terminate
        self._negative = frozenset(moves.negative_expressive)

    @classmethod
    def from_config(cls, cfg, knobs: Knobs, max_turns: int) -> Guard:
        return cls(cfg.guard.params(), knobs, max_turns, cfg.moves,
                   skip_floors_on_terminate=cfg.guard.skip_floors_on_terminate)

    def level_of(self, move: Move) -> DisclosureLevel | None:
        """The rung a move carries; when the move itself carries none, its secondary decides, so a signal-free
        move ("answering") cannot launder depth ("answering by going deeper")."""
        return self.moves.level_of(move.move, move.secondary_move)

    def check(self, move: Move, state: Ledger, turn: int) -> Verdict:
        level = self.level_of(move)
        if level is None:
            return Verdict(vetoed=False)
        skip_floors = self.skip_floors_on_terminate and move.end_conversation
        binding_cap = None
        for name, cap_fn in CEILING_CAPS.items():
            params = self.params.get(name)
            if params is None or not params.enabled:
                continue
            cap = cap_fn(state, self.knobs, turn, self.max_turns, params)
            if cap is not None and (binding_cap is None or rank(cap) < rank(binding_cap)):
                binding_cap = cap
        hits: list[RuleHit] = []
        applicable: list[str] = []
        for rule in RULES:
            params = self.params.get(rule.name)
            if params is None or not params.enabled or (skip_floors and rule.kind == "floor"):
                continue
            result = rule.fn(RuleContext(move=move, level=level, state=state, knobs=self.knobs, turn=turn,
                                         max_turns=self.max_turns, params=params,
                                         canonical=self.moves.canonical_move_by_level,
                                         negative_expressive=self._negative))
            if not result.applicable:
                continue
            if (rule.kind == "floor" and result.hit is not None and binding_cap is not None
                    and rank(result.hit.clamp_to) > rank(binding_cap)):
                continue  # pre-empted: asks for more than a ceiling allows this turn; counted as abstaining
            applicable.append(rule.name)
            if result.hit is not None:
                hits.append(result.hit)
        return self._resolve(hits, applicable)

    @staticmethod
    def _resolve(hits: list[RuleHit], applicable: list[str]) -> Verdict:
        # The tightest of each family; ties go to the earlier rule in RULES.
        ceiling = min((h for h in hits if h.kind == "ceiling"), key=lambda h: rank(h.clamp_to), default=None)
        floor = max((h for h in hits if h.kind == "floor"), key=lambda h: rank(h.clamp_to), default=None)
        conflict = ceiling is not None and floor is not None and rank(floor.clamp_to) > rank(ceiling.clamp_to)
        winner = ceiling or floor
        return Verdict(vetoed=bool(hits), hits=hits, applicable=applicable, conflict=conflict,
                       clamp_to=winner.clamp_to if winner else None, feedback=winner.feedback if winner else "")

    def clamp(self, move: Move, verdict: Verdict) -> Move:
        """Once the retries are spent: the canonical move of the resolved rung, keeping the model's reasoning so the
        trace still shows what it wanted. The secondary goes with the vetoed move. The canonical move may lie outside
        the member's archetype; the clamp is the harness's, not the member's, and the turn is flagged either way."""
        if verdict.clamp_to is None:
            return move
        return move.model_copy(update={"move": self.moves.canonical_move_by_level[verdict.clamp_to],
                                       "secondary_move": None})
