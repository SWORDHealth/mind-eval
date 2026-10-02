"""MindEval's sessions on NeMo UserSim: the `mindeval` probe, and the runner that takes a session through UserSim's
engine.

A session is one UserSim row. Its input (`SessionInput`, in the `mindeval_session` column) is everything the
session reads: the member, this session's situation and knobs, the memories both sides carry in, the clock, the
seed, the counselor's template and both sides' call settings (never a key: keys stay with the facades, see
`engine`). UserSim builds the probe from the row and awaits its `run_dispatch`, which runs `session.run_session`
whole: the member opens, then counselor and member alternate. The probe owns that loop, so none of UserSim's own
turn machinery (generated user turns, their gate judge, the in-loop assistant judge, context compression) touches
a mindeval conversation; every model call still goes through UserSim's `acall_llm`.

The row UserSim returns carries the conversation in UserSim's terms (the counselor is the assistant under test, the
member the user), the outcome and per-call accounting, and the session's record. The trace written beside it
(`trace.jsonl`) stays the record the judge and every other mindeval tool read.

Rows are self-contained, so a host can run them through UserSim's hosted runtime as well; with no trace path, a
session writes nothing but its row.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.llm import get_current_outcome_builder, set_conversation_id, set_current_outcome_builder
from usersim.engine.core.outcomes import FailureAttribution, FailureClass, OutcomeBuilder, OutcomeStatus
from usersim.engine.core.probes import BaseProbe, assistant_message, register_probe
from usersim.engine.core.simulation import ConversationState, make_result
from usersim.engine.core.storage import run_subroot, write_partitioned_dataset
from usersim.engine.generator import ConversationSimulatorGenerator

from mindeval import engine
from mindeval.config import CallSpec, load_config
from mindeval.member import MemberError, build_member
from mindeval.models import Knobs, MemberRow, Situation, TurnRecord
from mindeval.session import Counselor, run_session

LABEL = "mindeval"
SESSION_COLUMN = "mindeval_session"
LOCALE = "en_US"

#: Set by `Runner.session` around one session: the probe leaves the exception a session died with here, so the
#: caller re-raises it as it was, rather than reading a failed row's text.
_FAILURE: ContextVar[dict | None] = ContextVar("mindeval_failure", default=None)


def _spec_fields(spec: CallSpec) -> dict:
    return {k: v for k, v in dataclasses.asdict(spec).items() if k != "api_key"}


class SessionInput(BaseModel):
    """Everything one session reads, as it travels in a UserSim row."""

    model_config = ConfigDict(extra="forbid")

    member: MemberRow
    situation: Situation
    knobs: Knobs
    seed: int
    max_turns: int
    counselor_memory: str
    memory_text: str | None = None
    time_context: str | None = None
    #: Where the session writes its trace; None writes none (a hosted run).
    trace_path: str | None = None
    counselor_template: str
    #: CallSpec fields, less the key.
    counselor: dict
    patient: dict

    @classmethod
    def of(cls, *, counselor: Counselor, patient: CallSpec, **fields: Any) -> SessionInput:
        return cls(counselor_template=counselor.source, counselor=_spec_fields(counselor.spec),
                   patient=_spec_fields(patient), **fields)

    def counselor_spec(self) -> CallSpec:
        return CallSpec(**self.counselor)

    def patient_spec(self) -> CallSpec:
        return CallSpec(**self.patient)


@register_probe(family=LABEL, prompt_version=load_config().fingerprint, variants=("default",))
class MindEvalProbe(BaseProbe):
    """One mindeval session. The member is UserSim's simulated user and the counselor its assistant under test."""

    label = LABEL

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._input = SessionInput.model_validate_json(self._data[SESSION_COLUMN])
        self._counselor = Counselor.from_text(self._input.counselor_spec(), self._input.counselor_template)

    def get_user_system_prompt(self) -> str:
        s = self._input
        return build_member(load_config(), s.member, s.situation, s.knobs, s.patient_spec(),
                            memory_text=s.memory_text, time_context=s.time_context).system_prompt

    def get_assistant_system_prompt(self) -> str:
        return self._counselor.prompt_with(self._input.counselor_memory)

    def _mirror(self, state: ConversationState, rec: TurnRecord) -> None:
        """A written turn, into UserSim's conversation: the counselor's reply (its reasoning on it), then the
        member's words."""
        if rec.counselor_utterance is not None:
            state.messages.append(assistant_message(rec.meta.get("counselor") or {}, rec.counselor_utterance))
        if rec.utterance:
            state.messages.append({"role": "user", "content": rec.utterance})
        state.metadata.setdefault("moves", []).append(rec.final_move.model_dump() if rec.final_move else None)
        state.metadata.setdefault("flags", []).append(rec.flags)

    async def run_dispatch(self, *, models: dict[str, Any], data: dict[str, Any], cfg: Any,
                           state: ConversationState | None = None, seed_state: bool = True) -> dict:
        builder = self._outcome_builder or OutcomeBuilder(provenance=self._provenance)
        previous = get_current_outcome_builder()
        set_current_outcome_builder(builder)
        try:
            state = state or ConversationState(outcome=builder)
            if seed_state:
                self.seed_state_metadata(state)
            if not state.messages:
                state.messages.append({"role": "system", "content": self.get_assistant_system_prompt()})
            s = self._input
            t0 = time.monotonic()
            try:
                record = await run_session(
                    load_config(), s.member, s.situation, s.knobs, self._counselor, models=models,
                    patient=s.patient_spec(), max_turns=s.max_turns, seed=s.seed,
                    trace_path=Path(s.trace_path) if s.trace_path else None, counselor_memory=s.counselor_memory,
                    memory_text=s.memory_text, time_context=s.time_context,
                    on_turn=lambda rec: self._mirror(state, rec))
            except Exception as e:  # noqa: BLE001 — a failed row; the runner re-raises it as it was
                failure = _FAILURE.get()
                if failure is not None:
                    failure["error"] = e
                builder.set_wall_clock_s(time.monotonic() - t0)
                outcome = builder.finalize(
                    status=OutcomeStatus.FAILED, failure_class=FailureClass.INFRASTRUCTURE_ERROR,
                    failure_attribution=(FailureAttribution.USER_MODEL if isinstance(e, MemberError)
                                         else FailureAttribution.ASSISTANT_MODEL),
                    failure_detail=f"{type(e).__name__}: {e}")
                result = make_result(state.messages, state.metadata, False, outcome=outcome, traces=builder.traces())
                result.update(self.build_result_extras(state))
                return result
            state.metadata.update(session_id=record.session_id, run_id=record.run_id,
                                  termination=record.termination, agenda=record.agenda)
            builder.set_n_turns(sum(1 for m in state.messages if m["role"] == "assistant"))
            builder.set_wall_clock_s(time.monotonic() - t0)
            outcome = builder.finalize(status=OutcomeStatus.OK)
            result = make_result(state.messages, state.metadata, True, outcome=outcome, traces=builder.traces())
            result.update(self.build_result_extras(state))
            return result
        finally:
            set_current_outcome_builder(previous)


def trajectory_id(member_id: str, arc_id: str, episode: int, attempt: int) -> str:
    """A session's UserSim id: the member, its arc, the session and how many times it has been started."""
    return hashlib.sha256(f"{member_id}\x00{arc_id}\x00{episode}\x00{attempt}".encode()).hexdigest()[:32]


class Runner:
    """Takes sessions through UserSim's engine: its generator over mindeval's facades (`engine.facades`), and the
    rows written, one parquet file per session, under `rows_dir` as `usersim evaluate` reads them
    (`run=<run_id>/locale=en_US/probe_family=mindeval/`)."""

    def __init__(self, patient: CallSpec, counselor: CallSpec, *, max_turns: int, rows_dir: Path | None = None,
                 run_id: str = "0", provenance: dict | None = None) -> None:
        self.models = engine.facades(patient, counselor)
        self.generator = ConversationSimulatorGenerator.with_models(
            ConversationSimulatorConfig(name="conversation", max_turns=max_turns, context_compression=False),
            self.models)
        self.rows_dir, self.run_id, self.provenance = rows_dir, run_id, provenance or {}

    def row(self, inp: SessionInput, *, trajectory: str, episode: int) -> dict:
        """The UserSim row for one session. The persona is the formulation, which is what makes the member who
        they are; UserSim's own persona-derived settings are given, so none is drawn for a mindeval member."""
        cfg = load_config()
        provenance = {"nemotron_personas_version": None, "scenario_prompt_version": cfg.fingerprint,
                      "code_sha": self.provenance.get("git"), "bank_version": self.provenance.get("bank_version", {})}
        return {"persona": json.dumps({"first_name": inp.member.member_id, "persona": inp.member.formulation}),
                "probe_type": LABEL, "trajectory_id": trajectory, "member_id": inp.member.member_id,
                "episode": episode, "behavioral_profile": "{}", "disclosure_style": "upfront",
                "user_interaction_style": LABEL, "persona_grounding": False,
                "usersim_provenance": json.dumps(provenance), SESSION_COLUMN: inp.model_dump_json()}

    async def session(self, inp: SessionInput, *, trajectory: str, episode: int) -> dict:
        """Run one session and write its row. A session that died re-raises what it died with, after its row is
        written."""
        failure: dict = {}
        token = _FAILURE.set(failure)
        try:
            result = await self.generator.agenerate(self.row(inp, trajectory=trajectory, episode=episode))
        finally:
            _FAILURE.reset(token)
        set_conversation_id(None)
        self._write(result, f"{inp.member.member_id}-ep{episode:03d}-{trajectory[:8]}")
        if "error" in failure:
            raise failure["error"]
        if not result.get("conversation_status"):
            outcome = json.loads(result.get("simulation_outcome") or "{}")
            raise RuntimeError(f"session {episode + 1} failed in UserSim: {outcome.get('failure_detail')}")
        return result

    def _write(self, result: dict, name: str) -> None:
        if self.rows_dir is None:
            return
        import pandas as pd

        row = {k: (json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v)
               for k, v in result.items()}
        row.update(locale=LOCALE, probe_family=LABEL)
        write_partitioned_dataset(pd.DataFrame([row]), run_subroot(self.rows_dir, self.run_id),
                                  basename_template=f"{name}-{{i}}.parquet")
