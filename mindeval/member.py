"""The simulated member: one conversation per session, two calls per turn.

    counselor speaks  ->  commit_move(...)  ->  tool result: "accepted" + a writing cue  ->  the words

`commit_move` is a real, forced tool call: the move's schema goes to the API, so its shape is part
of the request. The member then writes the turn, and may first tick off or add to its own agenda
with two optional tools. Nothing reaches the member in anyone's voice but the counselor's and its
own; the harness speaks only through tool results.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

from mindeval import engine, llm
from mindeval.config import CallSpec, Config, provider_seed, render, turn_seed
from mindeval.guard import Guard, Ledger
from mindeval.models import Knobs, MemberRow, Move, Situation, TurnRecord

TOOL_NAME = "commit_move"


def move_tool(moves: dict[str, str]) -> dict:
    """The tool as declared to the API, with this member's move enum baked in. What the moves mean
    is in the member's system prompt."""
    names = sorted(moves)
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": ("Commit the move for this turn before saying anything. Call this "
                            "exactly once, then wait for the answer."),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reasoning": {"type": "string"},
                    "move": {"type": "string", "enum": names},
                    "secondary_move": {"type": ["string", "null"], "enum": names + [None],
                                       "default": None},
                    "end_conversation": {"type": "boolean", "default": False},
                },
                "required": ["reasoning", "move"],
            },
        },
    }


FORCE_MOVE = {"type": "function", "function": {"name": TOOL_NAME}}

#: The agenda tools, offered only between the accepted move and the words, and only when the
#: situation carries an agenda.
TODO_TOOLS = [
    {"type": "function", "function": {
        "name": "complete_todo",
        "description": "Mark one thing from your list as actually done in this conversation — "
                       "said, asked, or gotten to. Only when it genuinely happened.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"todo": {"type": "string"}}, "required": ["todo"]}}},
    {"type": "function", "function": {
        "name": "add_todo",
        "description": "Add something new you now realize you need to say or ask before this "
                       "conversation is over.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"todo": {"type": "string"}}, "required": ["todo"]}}},
]
#: A turn gets at most this many agenda calls before the words are required.
MAX_TODO_CALLS = 3

ACCEPTED = "accepted"

#: The answer to a `commit_move` issued on the speaking call, where the move is already committed
#: and the tool is not offered. See `Member.speak`.
ALREADY_COMMITTED = ("Your move for this turn is already committed and accepted — it stands. "
                     "commit_move is not offered again this turn; your next message is the words "
                     "you actually speak, nothing else.")
STRAY_COMMIT = "stray_commit_move"
MAX_STRAY_COMMITS = 2

#: Backoff between attempts at the same call, in seconds. Patient on purpose: a self-hosted server
#: can refuse requests for a while under load, and losing a session costs far more than waiting.
BACKOFF = (5, 15, 45, 60)


async def _pause(attempt: int) -> None:
    await engine.pause(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])


# Utterance hygiene. A dirty completion is retried like a transport error, and when the retries run
# out the words are replaced by `FALLBACK_UTTERANCE` rather than spoken: once in the member's
# conversation, dirty text is prompt for every later turn.
_CJK = re.compile(r"[　-ヿ㐀-鿿가-힯＀-￯]")
_MD_HEADER = re.compile(r"(?m)^#{1,6}\s|\*\*[A-Z]")
#: Chat-template or tool-call markup in the spoken text: any angle-bracket token. People typing on a
#: phone do not write angle brackets; a model copying tool syntax out of its own history does.
_TEMPLATE_MARKUP = re.compile(r"<[^<>\s][^<>]{0,60}>|<\|")
#: The protocol described instead of performed: tool and move ids, and the agenda's t1/t2 ids.
_PROTOCOL_LEAK = re.compile(
    r"\bcommit_move\b|\bsecondary_move\b|\bcomplete_todo\b|\badd_todo\b"
    r"|\bdo_[a-z_]{3,}\b|\barg_(?:key|value)\b|\bt\d\b")
#: A stage direction about the agenda, in brackets, instead of a spoken turn.
_STAGE_DIRECTION = re.compile(r"\[[^\]]{0,120}(?:\bt\d\b|complete|todo|checkpoint)[^\]]{0,120}\]?",
                              re.I)
MAX_UTTERANCE_CHARS = 2000
#: What the member says when every attempt came back dirty: the most ordinary thing a person says,
#: which reads as having nothing to say rather than as a visible seam.
FALLBACK_UTTERANCE = "i don't know"


def _hygiene_problems(text: str) -> list[str]:
    out = []
    if _CJK.search(text):
        out.append("non-English characters")
    if _MD_HEADER.search(text):
        out.append("markdown formatting")
    if _TEMPLATE_MARKUP.search(text):
        out.append("chat-template markup in the spoken text")
    if _PROTOCOL_LEAK.search(text):
        out.append("the turn protocol described instead of performed")
    if _STAGE_DIRECTION.search(text):
        out.append("a stage direction about the agenda instead of a spoken turn")
    if len(text) > MAX_UTTERANCE_CHARS:
        out.append(f"suspiciously long ({len(text)} chars)")
    norm = " ".join(text.split())
    step = 60
    for i in range(0, max(0, len(norm) - step), step):
        chunk = norm[i:i + step]
        if norm.count(chunk) > 1:
            out.append("self-duplicated text")
            break
    return out


class MemberError(RuntimeError):
    """A member-side call that could not be completed. Ends the session as `error`; `calls` and
    `turn_record` are the evidence, written beside the trace."""

    def __init__(self, message: str, *, calls: list[dict] | None = None) -> None:
        super().__init__(message)
        self.calls = calls or []
        self.turn_record = None


class EmptyCompletion(ValueError):
    """Neither a tool call nor content: a non-answer, retried like a transport failure."""


def _calls_to(meta: dict, name: str) -> list[dict]:
    return [c for c in (meta.get("tool_calls") or []) if c.get("name") == name]


def _attempt_record(attempt: int, problem: str, meta: dict, text: str = "") -> dict:
    usage = meta.get("usage")
    return {"attempt": attempt, "problems": [problem], "text": text.strip()[:400],
            "finish_reason": meta.get("finish_reason"),
            "reasoned": bool(meta.get("reasoning_content")),
            "tool_calls": [c.get("name") for c in (meta.get("tool_calls") or [])],
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "meta": meta}


@dataclass
class Member:
    """The member's own conversation, carried across the whole session."""

    system_prompt: str
    spec: CallSpec
    #: The commit_move tool, enum included.
    tool: dict
    #: The member's move vocabulary, checked at the parse boundary for providers that ignore the enum.
    vocab: frozenset
    #: What the template put into the system prompt, by variable.
    injections: dict = field(default_factory=dict)
    #: The answer an accepted commit gets: "accepted" plus the writing cue.
    accepted_text: str = ACCEPTED
    convo: list[dict] = field(default_factory=list)
    #: The models the session runs with, by UserSim role (`engine.facades`, or a hosted run's); the patient is
    #: `engine.PATIENT`.
    models: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.convo:
            self.convo = [{"role": "system", "content": self.system_prompt}]

    def hears(self, text: str) -> None:
        self.convo.append({"role": "user", "content": text})

    def said(self, text: str) -> None:
        self.convo.append({"role": "assistant", "content": text})

    async def commit(self, *, seed: int) -> tuple[Move | None, dict]:
        """Ask for the move. Returns (move or None, call_record).

        Transport failures and empty completions are retried here, with backoff; the model never saw
        them, so nothing is appended. Once a completion comes back, the assistant turn carrying the
        call is appended whatever it contains, because a tool result has to answer something.
        """
        t0 = time.monotonic()
        content, meta = "", {}
        last: Exception | None = None
        failed: list[dict] = []
        for attempt in range(1, self.spec.max_retries + 1):
            content, meta = "", {}
            try:
                content, meta = await engine.chat(
                    self.models, engine.PATIENT, self.convo, seed=provider_seed(seed + attempt - 1),
                    tools=[self.tool], tool_choice=FORCE_MOVE if self.spec.force_move else "auto")
                content, meta = llm.split_inline_trace(content, meta)
                if not _calls_to(meta, TOOL_NAME) and not content.strip():
                    raise EmptyCompletion(f"no tool call and no content (finish_reason={meta.get('finish_reason')})")
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = e
                failed.append(_attempt_record(attempt, type(e).__name__, meta, content))
                if attempt < self.spec.max_retries:
                    await _pause(attempt)

        def record(arguments, result, via, error) -> dict:
            rec = self._record(TOOL_NAME, seed, arguments, result, via, error, meta, t0)
            if failed:
                rec["failed_attempts"] = failed
            return rec

        if last is not None:
            via = "none" if isinstance(last, EmptyCompletion) else "error"
            return None, record(None, None, via, f"{type(last).__name__} on {len(failed)} attempts: {last}")

        raw = _calls_to(meta, TOOL_NAME)
        if raw:
            via, arguments, call_id = "tool_call", raw[0].get("arguments") or "", raw[0].get("id")
            call_arguments = arguments
        else:
            # A server that does not honour a forced function answers in prose; the move is parsed
            # out of it, and `via` records which mechanism produced it.
            via, arguments, call_id = "content_json", content, None
            # The synthetic commit_move call's arguments must stay valid JSON whatever the model
            # actually said: the prose itself, appended as-is, breaks a later call's history
            # conversion (Anthropic parses a tool_use's input as JSON), ending the session before any
            # HTTP request is made. `arguments` (below, and on the turn's trace) keeps the prose
            # itself; only the convo entry is ever synthesized.
            try:
                parsed = llm.parse_json(content)
                call_arguments = json.dumps(parsed) if isinstance(parsed, dict) else "{}"
            except Exception:  # noqa: BLE001
                call_arguments = "{}"

        self._append_call(call_arguments, call_id)
        try:
            move = Move.model_validate(llm.parse_json(arguments))
            for name in (move.move, move.secondary_move):
                if name is not None and name not in self.vocab:
                    raise ValueError(f"unknown move {name!r}; the moves are listed in your instructions")
        except Exception as e:  # noqa: BLE001
            return None, record(arguments, None, via, f"{type(e).__name__}: {e}")

        return move, record(arguments, move.model_dump(), via, None)

    def answer(self, text: str) -> bool:
        """Answer the last `commit_move` with a tool result. Refuses when there is no call to answer:
        a `tool` message that follows no call is a malformed conversation."""
        last = self.convo[-1] if self.convo else {}
        if not last.get("tool_calls"):
            return False
        self.convo.append({"role": "tool", "tool_call_id": last["tool_calls"][0].get("id") or "call_0",
                           "name": TOOL_NAME, "content": text})
        return True

    async def speak(self, *, seed: int, on_todo=None) -> tuple[str, list[dict]]:
        """The words, and first any agenda bookkeeping the member wants to do.

        With `on_todo` set, `complete_todo`/`add_todo` are offered alongside the words; a call is
        applied, answered, and the member asked again, at most `MAX_TODO_CALLS` times. `commit_move`
        is not offered here (offering it again made models issue a second commit and leave its markup
        in the text). A `commit_move` issued anyway is answered with `ALREADY_COMMITTED` and the words
        asked for again, at most `MAX_STRAY_COMMITS` times; it is recorded on the turn but removed from
        the conversation, so the history does not teach the model to do it again.

        Returns (utterance, call_records).
        """
        records: list[dict] = []
        todo_calls = 0
        strays = 0
        stray_at: list[int] = []  # where in `convo` each answered stray commit sits, to remove
        while True:
            allow_todos = on_todo is not None and todo_calls < MAX_TODO_CALLS
            tools = TODO_TOOLS if allow_todos else None
            choice = "auto" if allow_todos else None
            t0 = time.monotonic()
            last: Exception | None = None
            #: Every attempt's verdict, kept whether or not the turn ends up clean.
            attempts: list[dict] = []
            for attempt in range(1, self.spec.max_retries + 1):
                # Reset per attempt, not per while-iteration: an attempt that raises early (a stray
                # commit already at MAX_STRAY_COMMITS) must not leave `stray` (or `hygiene`) for a
                # later, clean attempt to be mistaken by, as commit() does inside its attempt loop.
                text, meta, calls, hygiene, stray = "", {}, [], [], None
                try:
                    text, meta = await engine.chat(
                        self.models, engine.PATIENT, self.convo,
                        seed=provider_seed(seed + attempt + todo_calls * 7 + strays * 13), tools=tools,
                        tool_choice=choice)
                    text, meta = llm.split_inline_trace(text, meta)
                    calls = [c for c in (meta.get("tool_calls") or [])
                             if c.get("name") in ("complete_todo", "add_todo")] if allow_todos else []
                    if not calls and not text.strip():
                        stray = next(iter(_calls_to(meta, TOOL_NAME)), None)
                        if stray is not None and strays < MAX_STRAY_COMMITS:
                            last = None
                            break  # answered below; not a failed attempt
                        raise ValueError(f"empty model output (finish_reason={meta.get('finish_reason')})")
                    hygiene = [] if calls else _hygiene_problems(text.strip())
                    if hygiene:
                        attempts.append({**_attempt_record(attempt, "", meta, text), "problems": hygiene})
                    if hygiene and attempt < self.spec.max_retries:
                        raise ValueError("utterance failed hygiene: " + "; ".join(hygiene))
                    last = None
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    if not hygiene:
                        attempts.append(_attempt_record(attempt, type(e).__name__, meta, text))
                    if attempt < self.spec.max_retries:
                        await _pause(attempt)
            if last is not None:
                failed = self._record("speak", seed, None, None, "completion", f"{type(last).__name__}: {last}",
                                      meta, t0)
                failed["hygiene_attempts"] = attempts
                raise MemberError(f"speaking failed after {self.spec.max_retries} attempts: "
                                  f"{type(last).__name__}: {last}", calls=[*records, failed]) from last

            if stray is not None:
                cid = stray.get("id")
                stray_at.append(len(self.convo))
                self._append_call(stray.get("arguments") or "", cid)
                self.convo.append({"role": "tool", "tool_call_id": cid or "call_0",
                                   "name": TOOL_NAME, "content": ALREADY_COMMITTED})
                records.append(self._record(STRAY_COMMIT, seed, stray.get("arguments"),
                                            ALREADY_COMMITTED, "tool_call", None, meta, t0))
                strays += 1
                continue

            if calls:
                call = calls[0]
                self._append_call(call.get("arguments") or "", call.get("id"), name=call["name"])
                try:
                    parsed = llm.parse_json(call.get("arguments") or "{}")
                except Exception:  # noqa: BLE001
                    parsed = {}
                todo = parsed.get("todo") if isinstance(parsed, dict) else None
                todo = todo if isinstance(todo, str) else ""
                result = on_todo(call["name"], todo)
                self.convo.append({"role": "tool", "tool_call_id": call.get("id") or "call_0",
                                   "name": call["name"], "content": result})
                records.append(self._record(call["name"], seed, call.get("arguments"),
                                            result, "tool_call", None, meta, t0))
                todo_calls += 1
                continue

            said, dropped = text.strip(), None
            if hygiene:
                said, dropped = FALLBACK_UTTERANCE, said
            self.said(said)
            for at in reversed(stray_at):
                del self.convo[at:at + 2]  # the stray call and its answer; the trace keeps them
            record = self._record("speak", seed, None, said, "completion", None, meta, t0)
            if hygiene:
                record["hygiene"] = hygiene
                record["hygiene_dropped"] = dropped
            if attempts:
                record["hygiene_attempts"] = attempts
            records.append(record)
            return said, records

    def _append_call(self, arguments: str, call_id: str | None, *, name: str = TOOL_NAME) -> None:
        self.convo.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id or "call_0", "type": "function",
             "function": {"name": name, "arguments": arguments}}]})

    def _record(self, name, seed, arguments, result, via, error, meta, t0) -> dict:
        return {"name": name,
                "kind": "tool_call" if name in (TOOL_NAME, STRAY_COMMIT, "complete_todo", "add_todo")
                else "completion",
                "seed": seed, "arguments": arguments, "result": result, "via": via,
                "error": error, "meta": meta or {},
                "ms": round((time.monotonic() - t0) * 1000, 1)}


#: A sentence boundary, captured so the text between sentences survives exactly as written.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(\s+)")
#: A move named in parentheses after the thing it labels: `an "idk" (do_disengage)`.
_PAREN_MOVE = re.compile(r" \((do_[a-z_]+)\)")
#: A run of backticked moves written as one list ("`a`, `b` or `c`"), with the "when ..." that
#: qualifies it, up to the next comma, semicolon, full stop or dash. An Oxford ", and" starts a new
#: run: in this prose it brings in a move with a qualifier of its own ("`do_express_sadness`, and
#: `do_express_hopelessness` when it feels like it will not lift").
_MOVE_RUN = re.compile(r"`do_[a-z_]+`(?:(?:, | or | and )`do_[a-z_]+`)*(?: when [^,;.—]*[^,;.—\s])?")
_RUN_NAME = re.compile(r"`(do_[a-z_]+)`")


def _without_run(s: str, runs: list, i: int) -> str | None:
    """`s` without its i-th move run, every move of which is avoided. A following run takes its place
    along with the words joining the two ("..., and for"); a last run goes with the words joining it to
    the one before. A run alone in its sentence goes by itself, and the sentence closes up around it;
    None when what would be left is no sentence at all (the run was its subject)."""
    run = runs[i]
    if i + 1 < len(runs):
        return s[:run.start()] + s[runs[i + 1].start():]
    if i > 0:
        return s[:runs[i - 1].end()] + s[run.end():]
    before, after = s[:run.start()], s[run.end():]
    if re.match(r" (?:and|or) ", after):  # "stop: `do_disengage` and leaving mid-thread"
        return before + after[after.index(" ", 1) + 1:]
    if re.match(r"[.!?]", after):  # "weight — `do_express_joy`." keeps "weight."
        before = before.rstrip(" —:,")
        return before + after if before else None
    if not before and re.match(r"[;,:] ", after):  # "`do_disagree` ...; you rarely question it"
        return after[2].upper() + after[3:]
    return None


def _drop_moves(sentence: str, avoided: set[str]) -> str | None:
    """One sentence with every avoided move taken out of it, or None if nothing sensible is left."""
    s = _PAREN_MOVE.sub(lambda m: "" if m.group(1) in avoided else m.group(0), sentence)
    while True:
        runs = list(_MOVE_RUN.finditer(s))
        i = next((i for i, r in enumerate(runs) if set(_RUN_NAME.findall(r.group(0))) & avoided), None)
        if i is None:
            break
        run = runs[i]
        names = _RUN_NAME.findall(run.group(0))
        kept = [n for n in names if n not in avoided]
        if not kept:
            s = _without_run(s, runs, i)
            if s is None:
                return None
            continue
        list_end = run.group(0).rindex("`") + 1  # the list alone, without its qualifier
        joined = [f"`{n}`" for n in kept]
        text = joined[0]
        if len(joined) > 1:  # the list keeps its own last joint: "a, b or c" less b is "a or c"
            conj = re.findall(r"`(, | or | and )`", run.group(0)[:list_end])[-1]
            text = ", ".join(joined[:-1]) + conj + joined[-1]
        after = s[run.start() + list_end:]
        if len(kept) == 1 and after.startswith(" are "):  # "`do_blame` is near to hand"
            after = " is " + after[len(" are "):]
        s = s[:run.start()] + text + after
    # Anything still naming an avoided move is written some other way; the sentence goes whole.
    if any(re.search(rf"\b{re.escape(m)}\b", s) for m in avoided):
        return None
    return s


def _without_avoided_moves(text: str, avoided: set[str]) -> str:
    """The text with every avoided move taken out of it. An archetype removes a move from the tool
    enum and the vocab check, but knobs.yaml and style2.yaml still name moves in prose to describe
    when a member reaches for them; left alone, that keeps telling the member to use a move it can no
    longer actually commit.

    Only the move and the words that exist to introduce it go: the rest of its sentence stays, since
    it can also carry the session's affect or a move that is still allowed. A move in a list leaves
    the list; one with its own "when ..." takes it along; a parenthetical `(do_x)` just disappears. A
    sentence goes whole only when nothing sensible would be left of it. A line naming no avoided move
    is returned unchanged."""
    if not avoided or not any(m in text for m in avoided):
        return text
    paragraphs = []
    for para in text.split("\n"):
        parts = _SENTENCE_BREAK.split(para)  # sentence, separator, sentence, ...
        kept = []
        for j in range(0, len(parts), 2):
            sentence = _drop_moves(parts[j], avoided)
            if sentence is not None:
                kept += [sentence, parts[j + 1] if j + 1 < len(parts) else ""]
        paragraphs.append("".join(kept).rstrip())
    return "\n".join(paragraphs)


def _style(cfg: Config, knobs: Knobs, knob_lines: list[str]) -> tuple[list[str], str, str, str, str]:
    """The style layer (config/style2.yaml): the writing-style sentence with its casing appended,
    the verbosity sentence, the orthography rule, the example turns, and the writing cue."""
    s2 = cfg.style2
    style, level = knobs.writing_style, knobs.verbosity
    own = cfg.knob_line(knobs, "writing_style")
    diction = s2.diction.get(style) or cfg.knob_text.writing_style[style]
    styled = f"{diction} {s2.orthography[style]}"
    lines = [styled if line == own else line for line in knob_lines]
    examples = " ".join(f"[ {turn} ]" for turn in s2.examples[level][style])
    cue = f"{s2.write_cue[level]} {s2.write_note} {s2.write_rule[style]}"
    return lines, s2.verbosity[level], s2.rule, f"{s2.examples_intro} {examples}", cue


def session_conduct(cfg: Config, row: MemberRow, knobs: Knobs) -> tuple[str | None, list[str], list[str]]:
    """How the member behaves in one session, as their own prompt puts it, for writing that session's
    opening message: the archetype's description, what it never does, and the session's concealment and
    affect lines (avoided moves taken out, as in `build_member`). All second person."""
    archetype = cfg.archetypes.archetypes[row.archetype] if row.archetype else None
    avoided = set(archetype.avoided_moves) if archetype else set()
    lines = [_without_avoided_moves(cfg.knob_line(knobs, name), avoided)
             for name in ("concealment_propensity", "session_affect")]
    return (archetype.description if archetype else None), (list(archetype.avoids_prose) if archetype else []), lines


def build_member(cfg: Config, row: MemberRow, situation: Situation, knobs: Knobs, spec: CallSpec, *,
                 models: dict | None = None, memory_text: str | None = None,
                 time_context: str | None = None) -> Member:
    """The member for one session. `knobs` are this session's (concealment and affect move across
    sessions); `memory_text` is the running log as the member carries it, None in session 1; `models`
    are what it is called through (see `Member.models`)."""
    # An archetype removes moves from the repertoire wholesale: the tool enum, the parse-boundary
    # check, and every mention in the knob or style prose that would tell the member to reach for one
    # by name (knobs.yaml and style2.yaml name moves the tool enum may not offer at all).
    archetype = cfg.archetypes.archetypes[row.archetype] if row.archetype else None
    avoided = set(archetype.avoided_moves) if archetype else set()

    knob_lines = cfg.knob_lines(knobs, exclude=("verbosity",))
    knob_lines, verbosity_line, style_rule, style_examples, write_cue = _style(cfg, knobs, knob_lines)
    knob_lines = [_without_avoided_moves(line, avoided) for line in knob_lines]
    secondary_note = _without_avoided_moves(cfg.style2.secondary_note, avoided)
    repertoire = {name: desc for name, desc in cfg.moves.moves.items() if name not in avoided}
    move_lines = [f"`{name}` — {desc}" for name, desc in repertoire.items()]
    # The depth moves the member can reach, least to most.
    depth_moves = [cfg.moves.canonical_move_by_level[level] for level in ("minimize", "partial", "full")
                   if cfg.moves.canonical_move_by_level.get(level) in repertoire]
    # The agenda is shown in a fixed per-member order, without ids.
    checkpoints = list(situation.checkpoints)
    random.Random(f"{row.member_id}\x00agenda").shuffle(checkpoints)
    ctx = {
        "formulation": row.formulation,
        "situation": situation,
        "knob_lines": knob_lines,
        "verbosity_line": verbosity_line,
        "style_rule": style_rule,
        "style_examples": style_examples,
        "secondary_note": secondary_note,
        "memory_text": memory_text,
        "time_context": time_context,
        "move_lines": move_lines,
        "depth_moves": depth_moves,
        "archetype_text": archetype.description if archetype else None,
        "archetype_avoids": archetype.avoids_prose if archetype else [],
        "agenda_goal": situation.goal,
        "agenda_checkpoints": checkpoints,
    }
    # De-dashed after rendering, so the profile and situation text lose theirs too.
    system_prompt = render("member_system", **ctx).replace(" —", ",")
    # What the template put into the system prompt, by variable, for the session record.
    injections = {
        **{k: v for k, v in ctx.items() if k not in ("situation", "secondary_note")},
        **{f"situation.{k}": getattr(situation, k) for k in ("event", "automatic_thoughts", "behaviors", "physical")},
        "archetype": row.archetype,
        "archetype_avoided": sorted(avoided),
        "write_cue": write_cue,
    }
    return Member(
        tool=move_tool(repertoire),
        vocab=frozenset(repertoire),
        injections=injections,
        system_prompt=system_prompt,
        accepted_text=f"{ACCEPTED}. {write_cue}",
        spec=spec,
        models=models or {},
    )


class Todos:
    """The session's agenda: the situation's checkpoints, ticked off or added to by the member.

    Items keep short ids, but the member never sees them and the answers do not echo the list. A
    completion is matched to an open item by containment, then token overlap, because the
    member names items in its own words."""

    def __init__(self, situation: Situation) -> None:
        self.goal = situation.goal
        self.items: list[dict] = [
            {"id": f"t{i + 1}", "text": c, "status": "open", "added_turn": None, "completed_turn": None}
            for i, c in enumerate(situation.checkpoints)]

    @property
    def active(self) -> bool:
        return self.goal is not None

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())

    def _resolve(self, todo: str) -> dict | None:
        norm = self._norm(todo)
        if not norm:
            return None
        for item in self.items:
            if item["status"] != "open":
                continue
            other = self._norm(item["text"])
            if norm in other or other in norm:
                return item
        tokens = set(norm.split())
        best, score = None, 0.0
        for item in self.items:
            if item["status"] != "open":
                continue
            others = set(self._norm(item["text"]).split())
            overlap = len(tokens & others) / max(1, len(tokens | others))
            if overlap > score:
                best, score = item, overlap
        return best if score >= 0.5 else None

    def handler(self, turn: int, rec: TurnRecord):
        def on_todo(name: str, todo: str) -> str:
            todo = todo.strip()
            if not todo:
                return "nothing named; say what you mean."
            if name == "add_todo":
                if self._resolve(todo) is not None:
                    # The member cannot see its list, so it is told the item is already there rather
                    # than left to add it again.
                    return "that is already on your list."
                self.items.append({"id": f"t{len(self.items) + 1}", "text": todo, "status": "open",
                                   "added_turn": turn, "completed_turn": None})
                rec.todos_added.append(todo)
                return "added."
            match = self._resolve(todo)
            if match is None:
                # Not on the list: the member still said a thing it needed to say, so it is recorded
                # as its own item, done.
                self.items.append({"id": f"t{len(self.items) + 1}", "text": todo, "status": "done",
                                   "added_turn": turn, "completed_turn": turn})
                rec.todos_completed.append(todo)
                return "noted."
            match["status"], match["completed_turn"] = "done", turn
            rec.todos_completed.append(match["text"])
            return "noted."
        return on_todo

    def snapshot(self) -> dict | None:
        return {"goal": self.goal, "todos": self.items} if self.active else None


async def member_turn(cfg: Config, member: Member, counselor_text: str, counselor_meta: dict, turn: int,
                      session_id: str, session_seed: int, blank_record, todos: Todos, *,
                      guard: Guard | None = None, ledger: Ledger | None = None) -> TurnRecord:
    """One member turn: commit (asked again if the move is malformed), hear "accepted", speak. With a `guard`, a
    vetoed move is answered with its feedback and committed again, then clamped (guard.py); `ledger` is the
    session so far, which the Guard rules over."""
    rec = blank_record(turn, "commit")
    rec.counselor_utterance = counselor_text

    move, call = await member.commit(seed=rec.turn_seed)
    rec.calls = [call]

    def answered(text: str) -> bool:
        if not member.answer(text):
            return False
        rec.calls[-1]["tool_result"] = text
        return True

    tries = 0
    while move is None and tries < member.spec.max_retries - 1:
        # Only a move the model actually produced is worth answering. A transport failure was
        # retried inside `commit`; there is no call to answer, and the turn fails.
        if not answered(f"That did not come through as a usable move — {call['error']}. Call commit_move again."):
            break
        tries += 1
        move, call = await member.commit(seed=turn_seed(session_seed, session_id, turn, f"commit.retry.{tries}"))
        rec.calls.append({**call, "schema_retry": tries})
    if move is None:
        err = MemberError(f"commit_move failed on turn {turn}: {call['error']}")
        err.turn_record = rec
        raise err

    raw, verdicts, retries, fallback = move, [], 0, False
    if guard is not None:
        verdict = guard.check(move, ledger, turn)
        verdicts.append(verdict)
        # Resampling under a veto is rejection sampling: every attempt's verdict is kept.
        while verdict.vetoed and retries < cfg.guard.max_retries:
            retries += 1
            answered(verdict.feedback)
            resampled, call = await member.commit(seed=turn_seed(session_seed, session_id, turn,
                                                                 f"commit.veto.{retries}"))
            rec.calls.append({**call, "guard_retry": retries})
            if resampled is None:
                break  # a malformed resample: the last good move stands, and the clamp settles it
            move = resampled
            verdict = guard.check(move, ledger, turn)
            verdicts.append(verdict)
        if verdict.vetoed:
            move, fallback = guard.clamp(move, verdict), True

    rec.final_move = move
    rec.inferred_level = cfg.moves.level_of(move.move, move.secondary_move)
    rec.move_class = cfg.moves.class_of(move.move)

    answered(member.accepted_text)
    try:
        utterance, speak_calls = await member.speak(seed=turn_seed(session_seed, session_id, turn, "speak"),
                                                    on_todo=todos.handler(turn, rec) if todos.active else None)
    except MemberError as e:
        rec.calls.extend(e.calls)
        e.turn_record = rec
        raise
    rec.calls.extend(speak_calls)
    rec.utterance = utterance
    speak_call = speak_calls[-1]
    rec.timings = {"commit_ms": sum(c.get("ms", 0) for c in rec.calls if c["name"] != "speak"),
                   "speak_ms": speak_call.get("ms", 0)}
    commit_calls = [c for c in rec.calls if c["name"] == TOOL_NAME]
    rec.meta = {"commit": commit_calls[-1]["meta"] if commit_calls else {},
                "speak": speak_call["meta"], "counselor": counselor_meta}
    rec.flags = _flags(rec)
    if guard is not None:
        rec.meta["guard"] = {"raw_move": raw.model_dump(), "raw_inferred_level": guard.level_of(raw),
                             "verdicts": [v.model_dump() for v in verdicts], "retries": retries, "fallback": fallback}
        rec.flags += [flag for flag, on in (("guard_fallback", fallback), ("guard_conflict", verdicts[-1].conflict),
                                            ("guard_veto", retries > 0)) if on]
    return rec


def _flags(rec: TurnRecord) -> list[str]:
    flags = []
    if any(c.get("hygiene") for c in rec.calls):
        flags.append("utterance_hygiene")
    if any(c["name"] == STRAY_COMMIT for c in rec.calls):
        flags.append("stray_commit")
    if any(c.get("failed_attempts") for c in rec.calls):
        flags.append("commit_retry")
    return flags
