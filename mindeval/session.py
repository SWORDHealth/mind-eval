"""One session: the member opens, then counselor and member alternate until the member leaves or the
counselor has replied `max_turns` times. Every turn is written to the trace as it happens.

The transcript is kept member-POV (the counselor is `user`, the member `assistant`) and flipped
once, when the counselor is called, so the counselor sees itself as the assistant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from mindeval import engine
from mindeval.config import SEED_FINGERPRINT, CallSpec, Config, turn_seed
from mindeval.guard import Guard, Ledger
from mindeval.member import Todos, build_member, member_turn
from mindeval.models import Knobs, MemberRow, SessionRecord, Situation, TurnRecord

ERROR_FILENAME = "error.jsonl"


#: Where a counselor prompt receives the running log of earlier sessions. Every prompt must carry it
#: exactly once, as written, so every counselor gets the same memory.
MEMORY_FIELD = "{{ memory }}"


def counselor_template(text: str) -> str:
    """A counselor system prompt: any text around one `{{ memory }}`, which is filled with the running
    log of earlier sessions (in session 1 it says there are none). A single trailing newline is dropped."""
    if (n := text.count(MEMORY_FIELD)) != 1:
        raise ValueError(f"must contain {MEMORY_FIELD} exactly once (found {n}): it is where the "
                         "counselor receives its log of earlier sessions")
    return text.removesuffix("\n")


#: What a counselor's own earlier turns carry back besides their words: its reasoning, exactly as the model returned
#: it. Several chat templates render earlier turns' reasoning into the prompt (Kimi K3 is trained on it and requires
#: it; GLM-5.3 and Qwen3.8 keep it by default, and without it see an empty trace on every earlier turn; GPT-5.6
#: renders earlier reasoning items by default, which only its Responses API returns). Templates that clear earlier
#: reasoning ignore it, and a provider that returns none has none carried.
CARRIED = ("reasoning_content", "thinking_blocks", "reasoning_items")


def carried(meta: dict) -> dict:
    return {k: meta[k] for k in CARRIED if meta.get(k)}


@dataclass
class Counselor:
    """The counselor under test: a system prompt template and a model, reached as UserSim's assistant
    (`engine.COUNSELOR`). It is offered no tools and nothing about it is inspected; what it says is the whole
    of what it contributes. Its earlier turns come back to it as it returned them, reasoning included
    (`CARRIED`). `source` is the template's text, which a session's input carries (`probe.SessionInput`)."""

    spec: CallSpec
    template: str
    source: str = ""

    @classmethod
    def from_text(cls, spec: CallSpec, text: str) -> Counselor:
        return cls(spec=spec, template=counselor_template(text), source=text)

    def prompt_with(self, memory: str) -> str:
        """The system prompt for one session."""
        return self.template.replace(MEMORY_FIELD, memory)

    async def respond(self, models: dict, system_prompt: str, transcript: list[dict]) -> tuple[str, dict]:
        flipped = {"user": "assistant", "assistant": "user"}
        messages = [{"role": "system", "content": system_prompt}]
        for m in transcript:
            msg = {"role": flipped[m["role"]], "content": m["content"]}
            if msg["role"] == "assistant":  # the counselor's own earlier turn
                msg.update({k: m[k] for k in CARRIED if m.get(k)})
                # litellm returns an interleaved Claude turn (thinking, text, thinking, text) as one text and a list of
                # thinking blocks, and sends it back thinking-first: a reordering Anthropic rejects as a modified turn
                # (400). A turn with more than one thinking block therefore goes back without its thinking.
                if len(msg.get("thinking_blocks") or []) > 1:
                    del msg["thinking_blocks"]
            messages.append(msg)
        text, meta = await engine.retry_text(models, engine.COUNSELOR, messages, spec=self.spec,
                                             what=f"counselor ({self.spec.model})")
        return text.strip(), {"requested_model": self.spec.model, **meta}


class TraceWriter:
    """Appends one JSON object per line and fsyncs it, so a session that dies at turn 9 leaves nine
    turns on disk. `read_trace` tolerates one truncated trailing line and nothing else. With no path (a
    session hosted elsewhere, whose record is UserSim's row) nothing is written."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = path.open("a", encoding="utf-8")

    @staticmethod
    def _emit(handle, obj: dict) -> None:
        if handle is None:
            return
        line = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        # JSON leaves these three raw, and `str.splitlines` (here and in other readers) breaks on them.
        line = line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029").replace("\x85", "\\u0085")
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def write_turn(self, rec: TurnRecord) -> None:
        self._emit(self._f, rec.model_dump(mode="json"))

    def write_error(self, rec: TurnRecord, error: str) -> None:
        """The turn a session died on, with every call the member side made for it, beside the trace."""
        if self.path is None:
            return
        with self.path.with_name(ERROR_FILENAME).open("a", encoding="utf-8") as f:
            self._emit(f, {"type": "turn_error", "session_id": rec.session_id, "turn": rec.turn,
                           "error": error, "record": rec.model_dump(mode="json")})

    def close(self, rec: SessionRecord | None = None) -> None:
        if rec is not None:
            self._emit(self._f, rec.model_dump(mode="json"))
        if self._f is not None:
            self._f.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc) -> None:
        if self._f is not None and not self._f.closed:
            self.close()


def read_trace(path: Path) -> tuple[list[TurnRecord], SessionRecord | None]:
    lines = path.read_text(encoding="utf-8").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    turns: list[TurnRecord] = []
    session: SessionRecord | None = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "session":
                session = SessionRecord.model_validate(obj)
            else:
                turns.append(TurnRecord.model_validate(obj))
        except (json.JSONDecodeError, ValidationError):
            if i == len(lines) - 1:
                warnings.warn(f"{path}: the last line is incomplete; {len(turns)} turn(s) recovered", stacklevel=2)
                break
            raise
    return turns, session


def input_hash(*inputs: BaseModel) -> str:
    """sha256 over the inputs a session reads, by content: an input edited in place keeps its id but
    not its hash."""
    h = hashlib.sha256()
    for model in inputs:
        h.update(model.model_dump_json().encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _run_id(row: MemberRow, situation: Situation, seed: int, fingerprint: str) -> str:
    material = "\x00".join([row.member_id, situation.situation_id, input_hash(row, situation), str(seed),
                            fingerprint])
    return hashlib.sha256(material.encode()).hexdigest()[:12]


async def run_session(cfg: Config, row: MemberRow, situation: Situation, knobs: Knobs, counselor: Counselor, *,
                      models: dict, patient: CallSpec, max_turns: int, seed: int, trace_path: Path | None,
                      counselor_memory: str, memory_text: str | None = None, time_context: str | None = None,
                      on_turn: Callable[[TurnRecord], None] | None = None) -> SessionRecord:
    """Run one session and return its record. A failure is written to the trace (and the turn it died
    on beside it) and re-raised. `models` are UserSim's, by role (`engine`); `on_turn` sees each turn as
    it is written."""
    member = build_member(cfg, row, situation, knobs, patient, models=models, memory_text=memory_text,
                          time_context=time_context)
    todos = Todos(situation)
    # The Guard measures a session as MindSim does, in member turns with the opening: one more than the replies.
    guard = Guard.from_config(cfg, knobs, max_turns + 1) if cfg.guard.enabled else None
    ledger = Ledger(max_turns)
    # Seeded from the pinned SEED_FINGERPRINT, not cfg.fingerprint: see config.py.
    run_id = _run_id(row, situation, seed, SEED_FINGERPRINT)
    session_id = f"{run_id}-s0"
    counselor_prompt = counselor.prompt_with(counselor_memory)
    turns: list[TurnRecord] = []
    transcript: list[dict] = []
    termination = "max_turns"

    def blank_record(turn: int, kind: str) -> TurnRecord:
        return TurnRecord(session_id=session_id, run_id=run_id, turn=turn, seed=seed,
                          turn_seed=turn_seed(seed, session_id, turn, kind), fingerprint=cfg.fingerprint,
                          knobs=knobs)

    def record(term: str, error: str | None = None) -> SessionRecord:
        return _session_record(cfg, row, situation, knobs, session_id, run_id, seed, turns, term, member,
                               counselor, counselor_prompt, todos, patient, max_turns, error)

    with TraceWriter(trace_path) as trace:
        try:
            # Turn 0: the member opens with the situation's opening message, spoken verbatim.
            rec = blank_record(0, "opening_message")
            rec.utterance = situation.opening_message
            rec.flags = ["opening_message"]
            ledger.record(rec)
            turns.append(rec)
            await asyncio.to_thread(trace.write_turn, rec)  # it fsyncs: off the loop the other members run on
            if on_turn is not None:
                on_turn(rec)
            member.said(rec.utterance)
            transcript.append({"role": "assistant", "content": rec.utterance})

            for turn in range(1, max_turns + 1):
                c0 = time.monotonic()
                text, cmeta = await counselor.respond(models, counselor_prompt, transcript)
                counselor_ms = round((time.monotonic() - c0) * 1000, 1)
                transcript.append({"role": "user", "content": text, **carried(cmeta)})
                member.hears(text)
                rec = await member_turn(cfg, member, text, cmeta, turn, session_id, seed, blank_record, todos,
                                        guard=guard, ledger=ledger)
                rec.timings["counselor_ms"] = counselor_ms
                ledger.record(rec)
                turns.append(rec)
                await asyncio.to_thread(trace.write_turn, rec)
                if on_turn is not None:
                    on_turn(rec)
                transcript.append({"role": "assistant", "content": rec.utterance})
                if rec.final_move is not None and rec.final_move.end_conversation:
                    termination = "member_terminated"
                    break
        except Exception as e:
            partial = getattr(e, "turn_record", None)
            if partial is not None:
                trace.write_error(partial, f"{type(e).__name__}: {e}")
            trace.close(record("error", f"{type(e).__name__}: {e}"))
            raise
        session = record(termination)
        trace.close(session)
    return session


def _session_record(cfg: Config, row: MemberRow, situation: Situation, knobs: Knobs, session_id: str,
                    run_id: str, seed: int, turns: list[TurnRecord], termination: str, member, counselor,
                    counselor_prompt: str, todos: Todos, patient: CallSpec, max_turns: int,
                    error: str | None) -> SessionRecord:
    flags = sorted({f for t in turns for f in t.flags})
    messages = []
    for t in turns:
        if t.counselor_utterance:
            messages.append({"role": "user", "content": t.counselor_utterance})
        if t.utterance:
            messages.append({"role": "assistant", "content": t.utterance})
    return SessionRecord(
        session_id=session_id, run_id=run_id, seed=seed, fingerprint=cfg.fingerprint,
        profile_id=row.member_id, situation_id=situation.situation_id, knobs=knobs, archetype=row.archetype,
        agenda=todos.snapshot(), termination=termination, error=error, flags=flags, member_messages=messages,
        member_system_prompt=member.system_prompt, counselor_system_prompt=counselor_prompt,
        formulation=row.formulation, situation=situation.model_dump(),
        setup={
            "max_turns": max_turns,
            "opening_speaker": "member",
            "member": {"model": patient.model, "api_base": patient.api_base, "temperature": patient.temperature,
                       "max_tokens": patient.max_tokens, "timeout": patient.timeout, "params": patient.params},
            "counselor": {"adapter": "litellm", "model": counselor.spec.model,
                          "temperature": counselor.spec.temperature, "max_tokens": counselor.spec.max_tokens,
                          "params": counselor.spec.params},
            "knob_instructions": cfg.knob_lines(knobs),
            "injections": member.injections,
            **({"guard": cfg.guard.setup()} if cfg.guard.enabled else {}),
        },
    )
