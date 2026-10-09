"""One member through N sessions under a virtual clock.

Before the first session the member's schedule is drawn: the gaps between sessions, the calendar
anchor, the beat of each stretch in between, and the concealment and affect per session. The draws
are seeded by the member and the seed alone, so every counselor meets the same schedule.

Between sessions the patient model makes two calls. Compaction folds the session into one running
log, which is the whole of memory: the member carries it into the next session and the counselor
walks in with the same log under `# Memory`. `memory_mode` changes what each side carries:
`compacted` (the default) is the log, on both sides; `none` gives the counselor the session-1 text in
every session, as if it kept nothing, while the member keeps the log; `full` gives both sides every
earlier session verbatim instead of the log, the transcript compaction reads, oldest first, the oldest
dropped first past `FULL_MEMORY_MAX_CHARS`, so the counselor never remembers more than the member;
`split` gives the counselor objective case notes of its own, compacted separately by the patient model
(`log_counselor.md`, prompt `prompts/ablation/compact_counselor.j2`, outside the fingerprint), while the member
carries every earlier session verbatim as under `full`; split and full differ in the counselor's memory alone.
Compaction of the shared log runs in every mode, so every arc has it. The mode is a run setting outside the
fingerprint, so every mode meets the same schedule. Evolution writes the next session's situation from the
member's formulation, the last situation, the last conversation, the time that has passed and the
drawn beat; its opening message is the member's first line next time. Under `full` and `split` it also reads every
conversation before the last one, verbatim, oldest first, the same sessions the member carries
(`prompts/ablation/evolve_situation_full.j2`, the benchmark prompt plus that one section); with a single earlier
session that is the benchmark prompt itself, so session 2 is written the same way in every mode.

A checkpoint after every session lets a member that died mid-run resume at the session it died in.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import NamedTuple

from mindeval import llm
from mindeval.config import REGISTER_KNOBS, CallSpec, Config, EpisodeFile, provider_seed, render, turn_seed
from mindeval.models import ArcRecord, ArcShape, EpisodeRef, MemberRow, SessionRecord, Situation
from mindeval.session import Counselor, input_hash, read_trace, run_session

STATE_FILE = "_state.json"
STATE_VERSION = 1
#: Compaction reads a whole session and reasons before writing, and it is the one call with no smaller
#: correct answer: a missing log costs every later session, on both sides. Its timeout is generous.
COMPACT_TIMEOUT = 1800.0


def member_seed(member_id: str, base: int) -> int:
    """A member's arc seed: stable in the member, not its position, so any subset reproduces."""
    return int.from_bytes(hashlib.sha256(f"{member_id}\x00{base}".encode()).digest()[:4], "big")


def arc_id_of(row: MemberRow, arc_seed: int, fingerprint: str) -> str:
    material = "\x00".join([row.member_id, row.situation.situation_id, input_hash(row), str(arc_seed), fingerprint])
    return hashlib.sha256(material.encode()).hexdigest()[:12]


# -- the schedule ----------------------------------------------------------------------------------
# One substream per (session, quantity): adding a session never moves an earlier draw.

_CONCEALMENT_ORDER = ("high", "medium", "low")


def _gaps(ep_cfg: EpisodeFile, arc_seed: int, arc_id: str, n: int) -> list[float]:
    bands = ep_cfg.inter_episode_gap_hours.bands
    gaps = []
    for ep in range(1, n):
        band = random.Random(turn_seed(arc_seed, arc_id, ep, "gap_band")).choices(bands, weights=[b.p for b in bands])[0]
        lo, hi = band.range
        gaps.append(random.Random(turn_seed(arc_seed, arc_id, ep, "gap_uniform")).uniform(lo, hi))
    return gaps


def _concealment(ep_cfg: EpisodeFile, arc_seed: int, arc_id: str, start: str, n: int) -> list[str]:
    """Opens toward `low` or holds, never closes: one dwell draw per level visited."""
    schedule: list[str] = []
    level, step = start, 0
    while len(schedule) < n:
        dist = ep_cfg.concealment_schedule.dwell.get(level, {"never": 1.0})
        keys = sorted(dist)
        drawn = random.Random(turn_seed(arc_seed, arc_id, step, "conceal_dwell")).choices(
            keys, weights=[dist[k] for k in keys])[0]
        if drawn == "never" or level == _CONCEALMENT_ORDER[-1]:
            schedule.extend([level] * (n - len(schedule)))
            break
        schedule.extend([level] * min(int(drawn), n - len(schedule)))
        level = _CONCEALMENT_ORDER[_CONCEALMENT_ORDER.index(level) + 1]
        step += 1
    return schedule[:n]


def _affect(ep_cfg: EpisodeFile, arc_seed: int, member_id: str, ep: int) -> str:
    vocab = ep_cfg.session_affect.vocabulary
    names = sorted(vocab)
    return random.Random(turn_seed(arc_seed, member_id, ep, "affect")).choices(
        names, weights=[vocab[n] for n in names])[0]


def draw_shape(ep_cfg: EpisodeFile, row: MemberRow, arc_seed: int, arc_id: str, n: int) -> ArcShape:
    """Session 1 keeps the affect and concealment its situation was written for."""
    weekday = random.Random(turn_seed(arc_seed, arc_id, 0, "anchor_weekday")).randrange(7)
    bands = ep_cfg.time_anchor.start_hour_bands
    rng = random.Random(turn_seed(arc_seed, arc_id, 0, "anchor_hour"))
    lo, hi = rng.choices(bands, weights=[b.p for b in bands])[0].range
    hour = rng.uniform(lo, hi)
    vocab = ep_cfg.episode_beats.vocabulary
    names = sorted(vocab)
    beats = [random.Random(turn_seed(arc_seed, arc_id, ep, "beat")).choices(names, weights=[vocab[x].p for x in names])[0]
             for ep in range(1, n)]
    return ArcShape(
        n_episodes=n, gaps_hours=_gaps(ep_cfg, arc_seed, arc_id, n), anchor_weekday=weekday, anchor_hour=hour,
        beats=beats, concealment=_concealment(ep_cfg, arc_seed, arc_id, row.knobs.concealment_propensity, n),
        affect=[row.knobs.session_affect] + [_affect(ep_cfg, arc_seed, row.member_id, ep) for ep in range(1, n)])


# -- the calendar ----------------------------------------------------------------------------------

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class Anchor(NamedTuple):
    weekday: int
    hour: float


def _part_of_day(hour: float) -> str:
    h = hour % 24
    if h < 5:
        return "the middle of the night"
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    if h < 21:
        return "evening"
    return "late evening"


def describe_ts(anchor: Anchor, ts_hours: float) -> str:
    """'Thursday evening' for a virtual time."""
    absolute = anchor.hour + ts_hours
    return f"{WEEKDAYS[(anchor.weekday + int(absolute // 24)) % 7]} {_part_of_day(absolute)}"


def describe_gap(hours: float) -> str:
    if hours < 1.5:
        return "about an hour"
    if hours < 24:
        return f"about {max(2, round(hours))} hours"
    days = hours / 24
    if days < 1.75:
        return "about a day"
    if days < 13:
        return f"about {round(days)} days"
    return f"about {round(days / 7)} weeks"


def describe_now(anchor: Anchor, now_ts: float, prev_ts: float | None = None) -> str:
    now = f"It is {describe_ts(anchor, now_ts)}."
    if prev_ts is None:
        return now
    return (f"{now} Your last conversation was {describe_gap(now_ts - prev_ts)} ago, "
            f"on {describe_ts(anchor, prev_ts)}.")


# -- memory ----------------------------------------------------------------------------------------

_ENTRY = re.compile(r"^-\s+(.+)$")


def render_log(entries: list[str]) -> str:
    return "\n".join(f"- {e}" for e in entries)


def parse_log(text: str) -> list[str]:
    """The log's entries. A line that is not an entry means the model did not return a log."""
    entries = []
    for line in (ln.rstrip() for ln in text.strip().splitlines()):
        if not line.strip():
            continue
        m = _ENTRY.match(line)
        if not m:
            raise ValueError(f"not an entry line: {line[:60]!r}")
        entries.append(m.group(1).strip())
    return entries


def member_memory(log: list[str]) -> str | None:
    """The log as the member carries it; nothing before the first compaction."""
    return render("memory_log", reader="member", log=render_log(log)).rstrip() if log else None


def counselor_memory(log: list[str]) -> str:
    """The same log framed as the counselor's notes. Never empty: the first session says there are no
    earlier ones, so the counselor knows from the start that sessions continue."""
    return render("memory_log", reader="counsellor", log=render_log(log)).rstrip()


#: What each side carries between sessions (see the module docstring).
MEMORY_MODES = ("compacted", "none", "full", "split")
#: `split`: the prompt the counselor's own notes are compacted with, and the file they are written to.
SPLIT = {"counselor": ("ablation/compact_counselor", "log_counselor.md")}
#: The modes in which the member, and the writer of the next situation, read every earlier session verbatim.
TRANSCRIPT_MODES = ("full", "split")
#: Those modes' situation writer: the benchmark's prompt with the earlier conversations added.
EVOLVE_FULL = "ablation/evolve_situation_full"
#: The cap on `full` memory, about 30k tokens: whole sessions are kept newest first while they fit.
FULL_MEMORY_MAX_CHARS = 120_000
_FULL_INTRO = {
    "counsellor": "Transcripts of your previous sessions with this member, oldest first. \"Counsellor\" is you; "
                  "\"Member\" is the member.",
    "member": "Your earlier conversations with the AI counsellor, word for word, oldest first. \"Member\" is you.",
}


def _full_blocks(sessions: list, label: str, reserved: int, max_chars: int) -> tuple[int, list[str]]:
    """One block per session, numbered from 1; the oldest are left out whole once `max_chars` would be passed.
    Returns how many were left out and the blocks kept."""
    blocks = [f"Session {i + 1}:\n{_conversation(s.member_messages, label)}" for i, s in enumerate(sessions)]
    kept, size = [], reserved
    for block in reversed(blocks):
        if kept and size + len(block) + 2 > max_chars:
            break
        kept.insert(0, block)
        size += len(block) + 2
    return len(blocks) - len(kept), kept


def _dropped_note(dropped: int) -> list[str]:
    return ([f"({dropped} earlier session{'s are' if dropped > 1 else ' is'} not included: the transcripts are "
             "too long to carry.)"] if dropped else [])


def full_memory(sessions: list, reader: str, max_chars: int = FULL_MEMORY_MAX_CHARS) -> str:
    """Every earlier session verbatim, as compaction reads it, oldest first, framed for `reader` ("member" or
    "counsellor"). Past `max_chars` the oldest sessions are left out whole, and the text says how many."""
    intro = _FULL_INTRO[reader]
    dropped, kept = _full_blocks(sessions, "Member", len(intro), max_chars)
    return "\n\n".join([intro, *_dropped_note(dropped), *kept])


def full_history(earlier: list, last, max_chars: int = FULL_MEMORY_MAX_CHARS) -> str:
    """The sessions before `last` for the situation writer under `full`: the ones the member carries into the next
    session, so the same oldest are left out, numbered as in the member's memory."""
    dropped, _ = _full_blocks([*earlier, last], "Member", len(_FULL_INTRO["member"]), max_chars)
    blocks = [f"Session {i + 1}:\n{_conversation(s.member_messages, 'Them')}"
              for i, s in enumerate(earlier) if i >= dropped]
    return "\n\n".join([*_dropped_note(dropped), *blocks])


def _check_mode(mode: str) -> None:
    if mode not in MEMORY_MODES:
        raise ValueError(f"unknown memory mode {mode!r}; choose one of {MEMORY_MODES}")


def counselor_memory_for(mode: str, log: list[str], sessions: list, ep: int,
                         split: dict[str, list[str]] | None = None) -> str:
    """The counselor's `# Memory` text for session `ep` (0-based) under `mode`."""
    _check_mode(mode)
    if mode == "compacted":
        return counselor_memory(log if ep > 0 else [])
    if mode == "none" or ep == 0:
        return counselor_memory([])
    if mode == "split":
        return counselor_memory(split["counselor"])
    return full_memory(sessions[:ep], "counsellor")


def member_memory_for(mode: str, log: list[str], sessions: list, ep: int,
                      split: dict[str, list[str]] | None = None) -> str | None:
    """What the member carries into session `ep` (0-based) under `mode`: nothing in session 1."""
    _check_mode(mode)
    if ep == 0:
        return None
    if mode in TRANSCRIPT_MODES:
        return full_memory(sessions[:ep], "member")
    return member_memory(log)


def _conversation(messages: list[dict], member_label: str) -> str:
    return "\n".join(f'{"Counsellor" if m["role"] == "user" else member_label}: {m["content"]}'
                     for m in messages if m.get("content"))


def compact_log(prior: list[str], session: SessionRecord, patient: CallSpec, *, max_entries: int,
                arc_seed: int, arc_id: str, episode_id: int, template: str = "compact_log",
                **context) -> list[str]:
    """Fold one closed session into the log and return the whole log. Raises after three attempts.
    `template` other than the shared log's draws its seed from a stream of its own."""
    prompt = render(template, prior_log=render_log(prior),
                    transcript=_conversation(session.member_messages, "Member"), max_entries=max_entries, **context)
    kind = "compact_log" if template == "compact_log" else f"compact_log:{template}"
    seed = provider_seed(turn_seed(arc_seed, arc_id, episode_id, kind))
    err = None
    for attempt in range(3):
        try:
            text, _ = llm.call_messages([{"role": "user", "content": prompt}], patient.model, patient.api_base,
                                        patient.temperature, patient.max_tokens, COMPACT_TIMEOUT, seed=seed + attempt,
                                        params=patient.params, api_key=patient.api_key)
            entries = parse_log(text)[:max_entries]
            if not entries:
                raise ValueError("empty log")
            return entries
        except Exception as e:  # noqa: BLE001 — retry on anything, report on exhaustion
            err = e
    raise RuntimeError(f"log compaction failed after session {episode_id + 1}: {err}")


def _situation_block(s: Situation) -> str:
    lines = [f"Event: {s.event}", "Thoughts: " + "; ".join(s.automatic_thoughts), "Did: " + "; ".join(s.behaviors)]
    if s.physical:
        lines.append("Body: " + "; ".join(s.physical))
    return "\n".join(lines)


def evolve_situation(cfg: Config, row: MemberRow, prev: Situation, last: SessionRecord, gap_hours: float,
                     patient: CallSpec, *, arc_seed: int, arc_id: str, episode_id: int, now_line: str,
                     beat: str, earlier: list | None = None) -> Situation:
    """The next session's situation, written by the patient model. Raises after three attempts. `earlier`, under
    `full`: the sessions before `last`, which the writer then reads too (`full_history`)."""
    leftover = [i["text"] for i in (last.agenda or {}).get("todos", []) if i["status"] == "open"]
    context = dict(formulation=row.formulation, prev_situation=_situation_block(prev),
                   transcript=_conversation(last.member_messages, "Them"), gap=describe_gap(gap_hours),
                   leftover_todos=leftover, now_line=now_line,
                   beat_gloss=cfg.episode.episode_beats.vocabulary[beat].gloss,
                   register_lines=[cfg.knob_line(row.knobs, k) for k in REGISTER_KNOBS])
    if earlier:
        prompt = render(EVOLVE_FULL, earlier=full_history(earlier, last), **context)
    else:
        prompt = render("evolve_situation", **context)
    seed = provider_seed(turn_seed(arc_seed, arc_id, episode_id, "evolve_situation"))
    err = None
    for attempt in range(3):
        try:
            raw, _ = llm.call_messages([{"role": "user", "content": prompt}], patient.model, patient.api_base,
                                       patient.temperature, patient.max_tokens, patient.timeout, seed=seed + attempt,
                                       params=patient.params, api_key=patient.api_key)
            if not raw.strip():
                raise ValueError("empty model output")
            parsed = llm.parse_json(raw)
            return Situation(situation_id=f"{prev.situation_id.split('-ep')[0]}-ep{episode_id}",
                             **{k: parsed[k] for k in ("event", "automatic_thoughts", "behaviors", "opening_message")},
                             physical=parsed.get("physical") or [],
                             goal=str(parsed.get("goal") or "") or None,
                             checkpoints=[str(c) for c in (parsed.get("checkpoints") or [])][:4])
        except Exception as e:  # noqa: BLE001 — includes validation errors
            err = e
    raise RuntimeError(f"writing the situation for session {episode_id + 1} failed: {err}")


def _situation_file(s: Situation, episode_id: int) -> str:
    source = ("the profile file" if episode_id == 0
              else "the patient model, from the last session and the time since")
    return f"# Session {episode_id + 1} situation, from {source}.\n" + json.dumps(
        s.model_dump(), indent=2, ensure_ascii=False) + "\n"


# -- checkpoints -----------------------------------------------------------------------------------

def _save_state(out: Path, *, arc_id: str, fingerprint: str, done: int, clock: float, log: list[str],
                episodes: list[EpisodeRef], split: dict[str, list[str]] | None = None) -> None:
    payload = {"version": STATE_VERSION, "arc_id": arc_id, "fingerprint": fingerprint, "episodes_done": done,
               "clock": clock, "log": log, "episodes": [e.model_dump() for e in episodes]}
    if split is not None:
        payload["split"] = split
    tmp = out / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out / STATE_FILE)


def _load_state(out: Path, *, arc_id: str, fingerprint: str):
    path = out / STATE_FILE
    if not path.is_file():
        return None
    ck = json.loads(path.read_text(encoding="utf-8"))
    if ck.get("arc_id") != arc_id or ck.get("fingerprint") != fingerprint:
        raise ValueError(f"{path} belongs to a different member, seed or configuration")
    done = int(ck["episodes_done"])
    sessions = []
    for i in range(done):
        _, session = read_trace(out / f"ep{i:03d}" / "trace.jsonl")
        if session is None:
            raise ValueError(f"{path} says {done} session(s) finished but session {i + 1} has no record")
        sessions.append(session)
    return (done, float(ck["clock"]), list(ck["log"]), [EpisodeRef(**e) for e in ck["episodes"]], sessions,
            ck.get("split"))


def set_aside_partial(arc_dir: Path) -> str | None:
    """Move an unfinished session's directory to `partial/` (never delete it) so the member resumes
    from its last finished session. Returns where it went, or None."""
    state = arc_dir / STATE_FILE
    if state.is_file():
        done = int(json.loads(state.read_text()).get("episodes_done", 0))
    elif arc_dir.is_dir() and not (arc_dir / "arc.json").is_file():
        done = 0
    else:
        return None
    stale = arc_dir / f"ep{done:03d}"
    if not stale.exists():
        return None
    aside = arc_dir / "partial" / f"ep{done:03d}-{time.strftime('%Y%m%dT%H%M%S')}"
    aside.parent.mkdir(exist_ok=True)
    stale.rename(aside)
    return str(aside)


# -- the run ---------------------------------------------------------------------------------------

def run_arc(cfg: Config, row: MemberRow, counselor: Counselor, *, patient: CallSpec, sessions: int,
            max_turns: int, arc_seed: int, out: Path, memory_mode: str = "compacted") -> ArcRecord:
    _check_mode(memory_mode)
    arc_id = arc_id_of(row, arc_seed, cfg.fingerprint)
    shape = draw_shape(cfg.episode, row, arc_seed, arc_id, sessions)
    anchor = Anchor(shape.anchor_weekday, shape.anchor_hour)
    spacing_hours = cfg.episode.turn_spacing_minutes / 60
    out.mkdir(parents=True, exist_ok=True)

    start, clock, log, episodes, done_sessions, split = 0, 0.0, [], [], [], None
    situation = row.situation
    resumed = _load_state(out, arc_id=arc_id, fingerprint=cfg.fingerprint)
    if resumed is not None:
        start, clock, log, episodes, done_sessions, split = resumed
        if start > 0:
            situation = Situation.model_validate(done_sessions[-1].situation)

    def compact_split(prior: dict[str, list[str]], record: SessionRecord, ep: int) -> dict[str, list[str]]:
        logs = {}
        for side, (template, name) in SPLIT.items():
            logs[side] = compact_log(prior[side], record, patient, max_entries=cfg.patient.log_max_entries,
                                     arc_seed=arc_seed, arc_id=arc_id, episode_id=ep, template=template,
                                     formulation=row.formulation)
            (out / f"ep{ep:03d}" / name).write_text(render_log(logs[side]) + "\n", encoding="utf-8")
        return logs

    if memory_mode == "split" and split is None:
        # Resumed from a checkpoint made in another mode (a fork after session 1): the counselor's notes are caught up
        # on the sessions already finished, in order.
        split = {side: [] for side in SPLIT}
        for i, record in enumerate(done_sessions):
            split = compact_split(split, record, i)

    for ep in range(start, sessions):
        ep_dir = out / f"ep{ep:03d}"
        trace_path = ep_dir / "trace.jsonl"
        if trace_path.exists():
            raise FileExistsError(f"{trace_path} already exists; move it aside before resuming")
        gap = shape.gaps_hours[ep - 1] if ep > 0 else None
        evolve_ms = None
        if ep > 0:
            t0 = time.monotonic()
            situation = evolve_situation(cfg, row, situation, done_sessions[-1], gap, patient, arc_seed=arc_seed,
                                         arc_id=arc_id, episode_id=ep, now_line=f"It is now {describe_ts(anchor, clock)}.",
                                         beat=shape.beats[ep - 1],
                                         earlier=done_sessions[:-1] if memory_mode in TRANSCRIPT_MODES else None)
            evolve_ms = round((time.monotonic() - t0) * 1000, 1)
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "situation.yaml").write_text(_situation_file(situation, ep), encoding="utf-8")

        ep_seed = turn_seed(arc_seed, arc_id, ep, "episode_seed")
        knobs = row.knobs.model_copy(update={"concealment_propensity": shape.concealment[ep],
                                             "session_affect": shape.affect[ep]})
        prev_close = episodes[-1].closed_ts if episodes else None
        record = run_session(cfg, row, situation, knobs, counselor, patient=patient, max_turns=max_turns,
                             seed=ep_seed, trace_path=trace_path,
                             memory_text=member_memory_for(memory_mode, log, done_sessions, ep, split),
                             counselor_memory=counselor_memory_for(memory_mode, log, done_sessions, ep, split),
                             time_context=describe_now(anchor, clock, prev_close))
        turns, _ = read_trace(trace_path)
        turn_ts = [clock + i * spacing_hours for i in range(len(turns))]
        closed = turn_ts[-1] if turn_ts else clock

        t0 = time.monotonic()
        log = compact_log(log, record, patient, max_entries=cfg.patient.log_max_entries, arc_seed=arc_seed,
                          arc_id=arc_id, episode_id=ep)
        compact_ms = round((time.monotonic() - t0) * 1000, 1)
        if memory_mode == "split":
            split = compact_split(split, record, ep)
        rendered = render_log(log)
        (ep_dir / "log.md").write_text(rendered + "\n", encoding="utf-8")
        episodes.append(EpisodeRef(
            episode_id=ep, run_id=record.run_id, session_id=record.session_id, situation_id=situation.situation_id,
            trace_path=str(trace_path.relative_to(out)), ep_seed=ep_seed, opened_ts=clock, closed_ts=closed,
            turn_ts=turn_ts, gap_hours_since_prev=gap, termination=record.termination, evolve_ms=evolve_ms,
            compact_ms=compact_ms, log_sha=hashlib.sha256(rendered.encode()).hexdigest()[:12],
            log_entries=len(log)))
        done_sessions.append(record)
        if ep < sessions - 1:
            clock = closed + shape.gaps_hours[ep]
        _save_state(out, arc_id=arc_id, fingerprint=cfg.fingerprint, done=ep + 1, clock=clock, log=log,
                    episodes=episodes, split=split)

    arc = ArcRecord(arc_id=arc_id, arc_seed=arc_seed, profile_id=row.member_id,
                    situation_id=row.situation.situation_id, fingerprint=cfg.fingerprint, shape=shape,
                    episodes=episodes)
    tmp = out / "arc.json.tmp"
    tmp.write_text(json.dumps(arc.model_dump(), indent=2, sort_keys=True) + "\n")
    tmp.replace(out / "arc.json")  # arc.json is the completion signal, so it appears whole or not at all
    (out / STATE_FILE).unlink(missing_ok=True)
    return arc
