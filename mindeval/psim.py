"""The patient-sim protocol (`--patient_protocol patient_sim`): the member played in the layout and turn protocol a
patient-simulator model was trained on, with this benchmark's member content.

    counselor speaks (stamped with the member's local time) -> expose_intent -> [end_chat] -> the words

The protocol comes from the model's own harness, `patient_sim.py` in the directory `--patient_protocol_dir` names,
loaded at run time: its role text, its two tools and their fixed answers, its stamps and its section texts, with the
training sampling the model card gives. The turn loop below is its `MemberSession`'s, call by call. Nothing of the
module is copied here.

What the member knows is this benchmark's, rendered into the model's sections:

- `# Who you are`: the name (the formulation's first word), the language, how the member writes (the writing style,
  its orthography, the no-dashes rule, the end-of-message rule and the verbosity sentence of config/style2.yaml) and
  how they engage (the archetype, what it does not do, and the concealment, frustration and affect knobs, with the
  move ids they name written as words, since this vocabulary is not the model's);
- `# What is true about your life`: the formulation, verbatim, and the self-harm paragraph of member_system.j2;
- `# What has happened, up to now`: the dates of the earlier sessions and the running log as the member carries it
  (arc.member_memory), then this session's situation (event, thoughts, behaviours, body);
- `# Your plan for this conversation`: the agenda (goal and checkpoints, in the default member's order), or the
  model's no-plan section when the situation has no goal.

What the default member has and this one does not: the `commit_move` and agenda tools (the agenda is shown, not
tracked), the writing cue after the move, the example turns (quoted turns are copied by this model), the
`# Being a person` and `# How conversations end` guidance (the role text covers both) and the time line (the stamps
and the dates carry it). What both share: the opening message spoken verbatim (here after a real `expose_intent`
for it), utterance hygiene with its fallback, the counselor side, the schedule and the session writer.

The member's reasoning stays in its history, as in training. When the history outgrows the server's context the
reasoning of the oldest half of the member's turns that still carry it is dropped and the turn retried
(`reasoning_trimmed`); the model card records the overflow and does not test the trim.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

from mindeval.config import KNOB_NAMES, CallSpec, Config, provider_seed
from mindeval.member import BACKOFF, FALLBACK_UTTERANCE, MemberError, _hygiene_problems, _style
from mindeval.models import Knobs, MemberRow, Move, Situation, TurnRecord

PROTOCOL = "patient_sim"
#: The files of the protocol directory that decide what the member receives; their hash is the run's identity.
PROTOCOL_FILES = ("patient_sim.py", "prompts/role.txt", "tools/expose_intent.json", "tools/end_chat.json")
#: Day 0 of the arc's calendar for the stamps: a Monday (`Anchor.weekday` 0), in UTC.
CALENDAR_START = dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc)
#: As member_system.j2's `# On self-harm`.
SELF_HARM = ("There is no self-harm or suicidal thinking in your picture, and there is nothing here you are hiding on "
             "that subject. If you are asked about it, and you may be, directly, you say clearly and without drama "
             "that it is not something you have been having. You do not hint at it, you do not leave it ambiguous to "
             "seem interesting, and you do not invent it. Answer the question you were asked and move on.")
#: As member_system.j2's `# What you remember` and `# What has happened recently`.
MEMORY_INTRO = ("You have talked with this AI counsellor before, on {dates}. What follows is your own recollection of "
                "those earlier conversations. It is memory, not the conversation you are in now: today starts fresh, "
                "and you are not obliged to bring any of it up. You do not announce that you remember things; it is "
                "simply what you carry in with you, the way anyone carries what was said last time.")
NOT_A_SECRET = ("The event itself is not a secret, it is the reason you are here, and it comes up naturally when you "
                "are asked. What you do not volunteer is everything underneath it. That part follows the rules you "
                "already live by, above.")
AGENDA_CLOSE = ("This is your own list, not homework. Some of it may not happen; something not on it may turn out to "
                "matter more.")
_MOVE_ID = re.compile(r"`do_([a-z_]+)`")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OVERFLOW = re.compile(r"context length|maximum model length|max_model_len|too long", re.I)


@lru_cache(maxsize=4)
def load_protocol(directory: str) -> ModuleType:
    """The model's harness module, `patient_sim.py`, from its directory."""
    path = Path(directory) / "patient_sim.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: --patient_protocol_dir names the model's harness directory")
    name = f"patient_sim_protocol_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


def protocol_sha256(directory: str) -> str:
    h = hashlib.sha256()
    for name in PROTOCOL_FILES:
        h.update(name.encode())
        h.update((Path(directory) / name).read_bytes())
    return h.hexdigest()


def local_time(weekday: int, hour: float, ts_hours: float) -> dt.datetime:
    """The arc's virtual time `ts_hours` as a date on the member's clock."""
    return CALENDAR_START + dt.timedelta(days=weekday, hours=hour + ts_hours)


def _dedash(text: str) -> str:
    return text.replace(" —", ",")


def _sentences(*parts: str | None) -> str:
    return " ".join(" ".join(p.split()) for p in parts if p)


def render_system_prompt(cfg: Config, row: MemberRow, situation: Situation, knobs: Knobs, ps: ModuleType, *,
                         memory_text: str | None, earlier: list[dt.datetime]) -> tuple[str, dict]:
    """The member's system prompt in the model's layout, and what went into it, by part."""
    knob_lines = cfg.knob_lines(knobs, exclude=("verbosity",))
    styled, verbosity_line, style_rule, _, _ = _style(cfg, knobs, knob_lines)
    at = [n for n in KNOB_NAMES if n != "verbosity"].index("writing_style")
    own_style = styled[at]  # with its orthography appended by _style
    archetype = cfg.archetypes.archetypes[row.archetype] if row.archetype else None
    writes = _sentences(own_style, style_rule, cfg.style2.write_rule.get(knobs.writing_style), verbosity_line)
    engages = _MOVE_ID.sub(lambda m: m.group(1).replace("_", " "), _sentences(
        archetype.description if archetype else None, *(archetype.avoids_prose if archetype else []),
        *(line for i, line in enumerate(styled) if i != at)))
    labels = ps.PROFILE_LABELS
    engage_label = next(v for k, v in labels.items() if k.startswith("how_you_engage"))
    profile = "\n".join([f"{labels['name']}: {row.formulation.split(' is ', 1)[0].strip()}",
                         f"{labels['language']}: English", f"{labels['how_you_write']}: {_dedash(writes)}",
                         f"{engage_label}: {_dedash(engages)}"])

    life = f"{row.formulation.strip()}\n\n{SELF_HARM}"
    history = []
    if memory_text:
        dates = ", ".join(f"{t:%a %Y-%m-%d}" for t in earlier)
        history += [MEMORY_INTRO.format(dates=dates), memory_text]
    recent = [f"What has happened recently: {situation.event}", "What went through your head about it:\n"
              + "\n".join(f"- {t}" for t in situation.automatic_thoughts),
              "What you did:\n" + "\n".join(f"- {b}" for b in situation.behaviors)]
    if situation.physical:
        recent.append("What your body did:\n" + "\n".join(f"- {p}" for p in situation.physical))
    history += ["\n".join(recent), NOT_A_SECRET]

    if situation.goal:
        checkpoints = list(situation.checkpoints)
        random.Random(f"{row.member_id}\x00agenda").shuffle(checkpoints)  # the default member's order
        items = [f"- What you want from this conversation: {situation.goal}"]
        if checkpoints:
            items.append("- Things you came in meaning to say, ask, or get to: " + "; ".join(checkpoints) + ".")
        plan = "\n\n".join([ps.PLAN_HEADING, ps.PLAN_FRAMING, "\n".join(items), AGENDA_CLOSE])
    else:
        plan = ps.render_no_plan()

    blocks = [ps.render_role(), ps.OPENING_PARAGRAPH, f"# Who you are\n\n{profile}",
              f"# What is true about your life\n\n{_dedash(life)}",
              "# What has happened, up to now\n\n" + _dedash("\n\n".join(history)), _dedash(plan)]
    injections = {"protocol": PROTOCOL, "profile": profile, "life": life, "history": history, "plan": plan,
                  "agenda_goal": situation.goal, "memory_text": memory_text,
                  "earlier_sessions": [ps.stamp(t) for t in earlier]}
    return "\n\n".join(blocks) + "\n", injections


class _Recorder:
    """An OpenAI client that keeps what each completion cost: `MemberSession` calls `chat.completions.create`."""

    def __init__(self, client) -> None:
        self._client = client
        self.log: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        t0 = time.monotonic()
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        self.log.append({"ms": round((time.monotonic() - t0) * 1000, 1), "finish_reason": choice.finish_reason,
                         "tool_calls": [c.function.name for c in (choice.message.tool_calls or [])],
                         "usage": usage.model_dump() if hasattr(usage, "model_dump") else None})
        return response


@lru_cache(maxsize=8)
def _client(api_base: str, api_key: str, timeout: float):
    from openai import OpenAI
    return OpenAI(base_url=api_base, api_key=api_key, timeout=timeout, max_retries=0)


def _served_name(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("hosted_vllm/") else model


@dataclass
class PsimMember:
    """One session's member. The turn loop is the model's protocol, call by call (its `MemberSession`, with this
    benchmark's retries, hygiene and records); the clock is the member's own: its message goes out `time_elapsed`
    after the counselor's arrived, and the counselor's reply arrives a minute after that, as in training."""

    system_prompt: str
    spec: CallSpec
    ps: ModuleType
    client: _Recorder
    model: str
    start: dt.datetime
    injections: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    now: dt.datetime | None = None
    trimmed: int = 0

    def __post_init__(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.now = self.start

    def _sampling(self, seed: int) -> dict:
        out = {**self.spec.params, "temperature": self.spec.temperature, "max_tokens": self.spec.max_tokens,
               "seed": seed}
        return {k: v for k, v in out.items() if v is not None}

    def _complete(self, seed: int):
        return self.client.chat.completions.create(model=self.model, messages=self.messages, tools=self.ps.TOOLS,
                                                   tool_choice="auto", **self._sampling(seed)).choices[0].message

    def _echo(self, message) -> dict:
        """The assistant message as it goes back into the history: reasoning under `reasoning`, which the chat
        template renders."""
        echoed = message.model_dump(exclude_none=True)
        thought = echoed.pop("reasoning_content", None) or echoed.get("reasoning")
        if thought:
            echoed["reasoning"] = thought
        return echoed

    def _trim(self) -> bool:
        """Drop the reasoning of the oldest half of the member's turns that still carry it."""
        carrying = [m for m in self.messages if m.get("role") == "assistant" and m.get("reasoning")]
        for m in carrying[: max(1, len(carrying) // 2)]:
            del m["reasoning"]
            self.trimmed += 1
        return bool(carrying)

    def _record(self, attempt: int, error: str | None, mark: int) -> dict:
        return {"name": "turn", "kind": "completion", "attempt": attempt, "error": error,
                "calls": list(self.client.log), "messages": self.messages[mark:]}

    def open(self, opening: str, *, seed: int) -> tuple[list[dict], dict, list[str]]:
        """The app's note, the member's `expose_intent` for its opening, then the opening message itself, verbatim.
        Returns (call_records, meta, flags)."""
        ps = self.ps
        self.messages.append({"role": "user", "content": ps.APP_NOTE.replace("{stamp}", ps.stamp(self.start))})
        records, intent, flags = [], None, []
        for attempt in range(1, self.spec.max_retries + 1):
            self.client.log.clear()
            mark = len(self.messages)
            try:
                reply = self._complete(provider_seed(seed + attempt - 1))
            except Exception as e:  # noqa: BLE001 — the opening goes undeclared after the last attempt
                records.append(self._record(attempt, f"{type(e).__name__}: {e}", mark))
                time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])
                continue
            declared = [c for c in (reply.tool_calls or []) if c.function.name == "expose_intent"]
            if declared:
                echoed = self._echo(reply)
                echoed["tool_calls"] = [c for c in echoed["tool_calls"] if c["function"]["name"] == "expose_intent"]
                self.messages += [echoed, {"role": "tool", "tool_call_id": declared[0].id,
                                           "content": ps.ACKS["expose_intent"]}]
                try:
                    intent = json.loads(declared[0].function.arguments or "{}")
                except json.JSONDecodeError:
                    intent = {}
                records.append(self._record(attempt, None, mark))
                break
            records.append(self._record(attempt, "no expose_intent for the opening", mark))
        if intent is None:
            flags.append("opening_undeclared")
        self.messages.append({"role": "assistant", "content": opening})
        self.now = self.start + dt.timedelta(seconds=ps.seconds_of((intent or {}).get("time_elapsed")) or 0)
        return records, {"intent": intent, "opened": ps.stamp(self.start), "sent": ps.stamp(self.now)}, flags

    def _attempt(self, seed: int) -> tuple[str | None, dict | None, dict | None, list[str]]:
        """expose_intent, end_chat only on the last message, then the message with no tool call; every call and
        tool answer stays in the history. At most four calls."""
        intent = end = None
        notes: list[str] = []
        for _ in range(4):
            reply = self._complete(seed)
            self.messages.append(self._echo(reply))
            text = _THINK.sub("", reply.content or "").strip()
            calls = reply.tool_calls or []
            if not calls:
                if intent is None:
                    notes.append("message before expose_intent")
                return text or None, intent, end, notes
            if text:
                notes.append("text beside a tool call")
            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                    notes.append(f"{name}: arguments are not JSON")
                if name == "expose_intent":
                    if intent is not None:
                        notes.append("expose_intent twice")
                    intent = args
                elif name == "end_chat":
                    if intent is None:
                        notes.append("end_chat before expose_intent")
                    end = args
                else:
                    notes.append(f"unknown tool {name}")
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": self.ps.ACKS.get(name, f"unknown tool: '{name}'")})
        notes.append("no message within the call budget")
        return None, intent, end, notes

    def speak(self, counselor_text: str, *, seed: int):
        """The counselor's message, stamped a minute after the member's last one, then one member turn. Returns
        (words, intent, end_chat, notes, arrived, sent, call_records, flags); raises MemberError when no clean turn
        comes back."""
        arrived = self.now + dt.timedelta(minutes=1)
        self.messages.append({"role": "user", "content": f"{self.ps.stamp(arrived)} {counselor_text}"})
        records, flags, error = [], [], "no attempt"
        for attempt in range(1, self.spec.max_retries + 1):
            mark = len(self.messages)
            self.client.log.clear()
            words = intent = end = None
            notes: list[str] = []
            try:
                words, intent, end, notes = self._attempt(provider_seed(seed + attempt - 1))
                problems = _hygiene_problems(words) if words else []
                error = ("utterance failed hygiene: " + "; ".join(problems) if problems
                         else None if words else "no message: " + "; ".join(notes))
            except Exception as e:  # noqa: BLE001
                problems, error = [], f"{type(e).__name__}: {e}"
            if words and (not problems or attempt == self.spec.max_retries):
                record = self._record(attempt, None, mark)
                record.update(notes=notes, hygiene=problems or None, hygiene_dropped=words if problems else None)
                if problems:  # the words never reach the counselor; the member's history says the fallback
                    flags.append("utterance_hygiene")
                    words = self.messages[-1]["content"] = FALLBACK_UTTERANCE
                records.append(record)
                if attempt > 1:
                    flags.append("member_retry")
                gap = self.ps.seconds_of((intent or {}).get("time_elapsed"))
                self.now = arrived + dt.timedelta(seconds=gap if gap is not None else 60)
                return words, intent, end, notes, arrived, self.now, records, flags
            records.append(self._record(attempt, error, mark))
            del self.messages[mark:]
            if _OVERFLOW.search(error) and self._trim():
                flags.append("reasoning_trimmed")
                continue
            if attempt < self.spec.max_retries:
                time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])
        raise MemberError(f"member turn failed after {self.spec.max_retries} attempts: {error}", calls=records)


def build_member(cfg: Config, row: MemberRow, situation: Situation, knobs: Knobs, spec: CallSpec, *,
                 memory_text: str | None, start: dt.datetime, earlier: list[dt.datetime]) -> PsimMember:
    ps = load_protocol(spec.protocol_dir)
    prompt, injections = render_system_prompt(cfg, row, situation, knobs, ps, memory_text=memory_text,
                                              earlier=earlier)
    client = _Recorder(_client(spec.api_base, spec.api_key or "EMPTY", spec.timeout))
    return PsimMember(system_prompt=prompt, spec=spec, ps=ps, client=client, model=_served_name(spec.model),
                      start=start, injections=injections)


def _move(cfg: Config, intent: dict | None, end: dict | None) -> tuple[Move, str | None, str | None]:
    moves = [m for m in ((intent or {}).get("moves") or []) if isinstance(m, str)] or ["other"]
    secondary = next((m for m in moves[1:] if m != moves[0]), None)
    move = Move(reasoning=str((end or {}).get("reason") or ""), move=moves[0], secondary_move=secondary,
                end_conversation=end is not None)
    # Classes and rungs where the model's move has a namesake in this benchmark's vocabulary (do_<name>).
    mine = cfg.moves.moves
    own = f"do_{moves[0]}" if f"do_{moves[0]}" in mine else None
    second = f"do_{secondary}" if secondary and f"do_{secondary}" in mine else None
    return move, (cfg.moves.class_of(own) if own else None), cfg.moves.level_of(own or "", second)


def opening_turn(cfg: Config, member: PsimMember, rec: TurnRecord) -> None:
    rec.calls, meta, flags = member.open(rec.utterance, seed=rec.turn_seed)
    rec.meta = meta
    rec.flags += flags
    if meta["intent"] is not None:
        rec.final_move, rec.move_class, rec.inferred_level = _move(cfg, meta["intent"], None)
        rec.final_move.end_conversation = False


def member_turn(cfg: Config, member: PsimMember, counselor_text: str, counselor_meta: dict, turn: int,
                blank_record) -> TurnRecord:
    """One member turn under the model's protocol: the counselor's message reaches the member stamped, the member
    declares its intent (and, on its last message, `end_chat`), then writes."""
    rec = blank_record(turn, "speak")
    rec.counselor_utterance = counselor_text
    t0 = time.monotonic()
    try:
        words, intent, end, notes, arrived, sent, calls, flags = member.speak(counselor_text, seed=rec.turn_seed)
    except MemberError as e:
        rec.calls = e.calls
        e.turn_record = rec
        raise
    rec.calls = calls
    rec.utterance = words
    rec.final_move, rec.move_class, rec.inferred_level = _move(cfg, intent, end)
    rec.timings = {"member_ms": round((time.monotonic() - t0) * 1000, 1)}
    rec.meta = {"intent": intent, "end_chat": end, "arrived": member.ps.stamp(arrived), "sent": member.ps.stamp(sent),
                "notes": notes, "counselor": counselor_meta}
    rec.flags = sorted(set(flags) | ({"protocol_note"} if notes else set()))
    return rec
