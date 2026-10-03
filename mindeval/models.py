"""The types that cross a boundary: the profile file's rows, a committed move, and the trace records.

Every model forbids unknown keys, so a field that is not declared here does not exist on the wire.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

KnobLevel = Literal["low", "medium", "high"]
VerbosityLevel = Literal["low", "medium"]
#: How the person writes. Not ordinal: `formal` is different from `casual`, not more of it.
WritingStyle = Literal["formal", "plain", "casual"]
#: Ekman's six.
SessionAffect = Literal["anger", "disgust", "fear", "joy", "sadness", "surprise"]
DisclosureLevel = Literal["withhold", "deflect", "minimize", "partial", "full"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Knobs(_Strict):
    """The behavioural dials, one sentence each in the member prompt. Field order is render order."""

    concealment_propensity: KnobLevel
    frustration_reactivity: KnobLevel
    session_affect: SessionAffect
    verbosity: VerbosityLevel
    writing_style: WritingStyle


class Situation(_Strict):
    """What has just happened in the member's life, going into one session. The member states it
    plainly when asked; `opening_message` is their first line, and `goal` and `checkpoints` seed the
    agenda they came in with."""

    situation_id: str
    event: str
    automatic_thoughts: list[str]
    behaviors: list[str]
    physical: list[str] = Field(default_factory=list)
    opening_message: str
    goal: str | None = None
    checkpoints: list[str] = Field(default_factory=list, max_length=4)


class MemberRow(_Strict):
    """One line of the profile file.

    `formulation` is opaque text pasted into the member prompt verbatim. `knobs` hold for session 1
    (its situation was written for that affect and concealment); later sessions draw affect and open
    concealment on a schedule (config/episode.yaml). `archetype` names an entry in
    config/archetypes.yaml, or is null."""

    member_id: str
    formulation: str
    knobs: Knobs
    archetype: str | None = None
    situation: Situation


class Move(_Strict):
    """What the member commits to before any words exist: the arguments of `commit_move`."""

    reasoning: str
    move: str
    #: A second act riding along, or the depth a flat move carried. Recorded, never required.
    secondary_move: str | None = None
    #: The member is leaving: the words of this turn are its last and the session ends after them.
    end_conversation: bool = False

    @model_validator(mode="after")
    def _distinct(self) -> Move:
        if self.secondary_move == self.move:
            raise ValueError("secondary_move must differ from move (or be null)")
        return self


class TurnRecord(_Strict):
    """One line of `trace.jsonl`. Turn 0 is the member's opening message; turn k >= 1 is the
    counselor's k-th reply and the member's answer to it."""

    type: Literal["turn"] = "turn"
    session_id: str
    run_id: str
    turn: int
    seed: int
    turn_seed: int
    fingerprint: str
    knobs: Knobs

    counselor_utterance: str | None = None
    #: The disclosure rung of the committed move (or of its secondary when the move carries none).
    inferred_level: DisclosureLevel | None = None
    move_class: str | None = None
    todos_completed: list[str] = Field(default_factory=list)
    todos_added: list[str] = Field(default_factory=list)
    final_move: Move | None = None

    utterance: str = ""
    flags: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)
    #: Every model call the member side made this turn, in order, with what came back.
    calls: list[dict] = Field(default_factory=list)


class SessionRecord(_Strict):
    """The last line of `trace.jsonl`: everything needed to read the session on its own."""

    type: Literal["session"] = "session"
    session_id: str
    run_id: str
    seed: int
    fingerprint: str
    profile_id: str
    situation_id: str
    knobs: Knobs
    archetype: str | None = None
    #: {"goal": str, "todos": [{"id", "text", "status", "added_turn", "completed_turn"}]}.
    agenda: dict | None = None
    termination: Literal["max_turns", "member_terminated", "error"]
    error: str | None = None
    flags: list[str] = Field(default_factory=list)
    #: The spoken turns, member-POV: the counselor is `user`, the member `assistant`.
    member_messages: list[dict] = Field(default_factory=list)
    member_system_prompt: str = ""
    counselor_system_prompt: str | None = None
    formulation: str = ""
    situation: dict = Field(default_factory=dict)
    setup: dict = Field(default_factory=dict)


class ArcShape(_Strict):
    """A member's drawn schedule, fixed before the first session."""

    n_episodes: int = Field(ge=1)
    #: Hours of silence before each session after the first.
    gaps_hours: list[float]
    anchor_weekday: int
    anchor_hour: float
    #: The shape of the stretch before each session after the first.
    beats: list[str]
    #: The concealment level in force per session.
    concealment: list[KnobLevel]
    #: The affect nearest the surface per session.
    affect: list[SessionAffect]

    @model_validator(mode="after")
    def _lengths(self) -> ArcShape:
        n = self.n_episodes
        if not (len(self.gaps_hours) == len(self.beats) == n - 1 and len(self.concealment) == len(self.affect) == n):
            raise ValueError("schedule lengths do not match the number of sessions")
        return self


class EpisodeRef(_Strict):
    """One session as the arc records it."""

    episode_id: int
    run_id: str
    session_id: str
    situation_id: str
    trace_path: str
    ep_seed: int
    #: Virtual hours since the arc started.
    opened_ts: float
    closed_ts: float
    turn_ts: list[float]
    gap_hours_since_prev: float | None = None
    termination: str
    evolve_ms: float | None = None
    compact_ms: float | None = None
    log_sha: str | None = None
    log_entries: int | None = None


class ArcRecord(_Strict):
    """`arc.json`, written when a member's last session closes."""

    schema_version: str = "1"
    arc_id: str
    arc_seed: int
    profile_id: str
    situation_id: str
    fingerprint: str
    shape: ArcShape
    episodes: list[EpisodeRef]
