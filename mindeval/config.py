"""The configuration that defines the simulated member, loaded once and fingerprinted.

Everything under `config/` and `prompts/` is part of the benchmark: the fingerprint is a hash over
those files, recorded on every trace line and in `run.json`, so an edit to any of them makes a
different benchmark and a run cannot be resumed across it. Call settings (models, endpoints,
timeouts) come from the command line and are not fingerprinted.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import AfterValidator, BaseModel, Field, ValidationInfo, model_validator

from mindeval.guard import GuardFile
from mindeval.models import KnobLevel, Knobs, SessionAffect, VerbosityLevel, WritingStyle, _Strict

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config"
PROMPTS_DIR = PACKAGE_DIR / "prompts"
#: Bump to deliberately change every derived per-call seed — the one documented way to re-draw the
#: benchmark on purpose. Bump SEED_FINGERPRINT alongside it: arc/run ids are pinned there, and left
#: alone they would keep their names while the draws underneath them moved.
SEED_VERSION = "1"
#: The namespace arc_id_of() and _run_id() draw from, in place of `cfg.fingerprint`, so a run's ids
#: and every seed drawn under them derive from this and not from the config's bytes — an edit that
#: leaves the data alone (a comment, a rename) keeps every arc. It is MindEval2 0.2.0's fingerprint
#: (its config/prompts, unedited), so arcs and seeds here match runs made with it. Change it, with
#: SEED_VERSION above, to re-draw the benchmark on purpose.
SEED_FINGERPRINT = "7962e7caed157571"
KNOB_NAMES = tuple(Knobs.model_fields)
#: The knobs that describe how the member writes; the only ones the situation writer is shown.
REGISTER_KNOBS = ("verbosity", "writing_style")
LEVELS = ("withhold", "deflect", "minimize", "partial", "full")


class MovesFile(_Strict):
    moves: dict[str, str]
    classes: dict[str, list[str]]
    level_by_move: dict[str, str]
    canonical_move_by_level: dict[str, str]
    #: Visible negative affect; the Guard's F4 exempts backsliding right after one.
    negative_expressive: list[str] = []

    @model_validator(mode="after")
    def _consistent(self) -> MovesFile:
        classed = [m for members in self.classes.values() for m in members]
        if sorted(classed) != sorted(self.moves):
            raise ValueError("every move must sit in exactly one class")
        tables = set(self.level_by_move) | set(self.canonical_move_by_level.values()) | set(self.negative_expressive)
        if not tables <= set(self.moves) or not set(self.level_by_move.values()) <= set(LEVELS):
            raise ValueError("level tables name unknown moves or levels")
        return self

    def class_of(self, move: str) -> str | None:
        return next((name for name, members in self.classes.items() if move in members), None)

    def level_of(self, move: str, secondary: str | None) -> str | None:
        """The disclosure rung a committed move carries; the secondary decides when the move has none."""
        return self.level_by_move.get(move) or self.level_by_move.get(secondary)


class Archetype(_Strict):
    name: str
    description: str
    avoided_moves: list[str]
    avoids_prose: list[str] = []


class ArchetypesFile(_Strict):
    archetypes: dict[str, Archetype]


class KnobText(_Strict):
    concealment_propensity: dict[KnobLevel, str]
    frustration_reactivity: dict[KnobLevel, str]
    session_affect: dict[SessionAffect, str]
    verbosity: dict[VerbosityLevel, str]
    writing_style: dict[WritingStyle, str]


class Style2File(_Strict):
    rule: str
    diction: dict[WritingStyle, str] = {}
    orthography: dict[WritingStyle, str]
    verbosity: dict[VerbosityLevel, str]
    examples_intro: str
    examples: dict[VerbosityLevel, dict[WritingStyle, list[str]]]
    write_note: str
    write_cue: dict[VerbosityLevel, str]
    write_rule: dict[WritingStyle, str]
    secondary_note: str | None = None


def _sums_to_one(dist, info: ValidationInfo):
    """A list of bands, or a {name: p} dict, whose probabilities (`.p` or the value) sum to one."""
    ps = [getattr(x, "p", x) for x in (dist.values() if isinstance(dist, dict) else dist)]
    if abs(sum(ps) - 1.0) > 1e-6:
        raise ValueError(f"{info.field_name} probabilities sum to {sum(ps)}, not 1")
    return dist


class Band(_Strict):
    range: tuple[float, float]
    p: float = Field(gt=0)


class Beat(_Strict):
    p: float = Field(gt=0)
    gloss: str


Bands = Annotated[list[Band], AfterValidator(_sums_to_one)]
Dist = Annotated[dict[str, float], AfterValidator(_sums_to_one)]


class GapSpec(_Strict):
    bands: Bands


class AffectSpec(_Strict):
    vocabulary: Annotated[dict[SessionAffect, float], AfterValidator(_sums_to_one)]


class ConcealmentSpec(_Strict):
    #: starting level -> {sessions at this level -> probability}; "never" holds it for good.
    dwell: dict[str, Dist]


class TimeAnchorSpec(_Strict):
    start_hour_bands: Bands


class BeatsSpec(_Strict):
    vocabulary: Annotated[dict[str, Beat], AfterValidator(_sums_to_one)]

class EpisodeFile(_Strict):
    turn_spacing_minutes: float = Field(gt=0)
    inter_episode_gap_hours: GapSpec
    session_affect: AffectSpec
    concealment_schedule: ConcealmentSpec
    time_anchor: TimeAnchorSpec
    episode_beats: BeatsSpec


class PatientProvider(_Strict):
    match: str
    params: dict = {}
    max_tokens: int | None = Field(default=None, gt=0)
    force_move: bool = True

    @model_validator(mode="after")
    def _regex(self) -> PatientProvider:
        re.compile(self.match)
        return self


class PatientFile(_Strict):
    temperature: float = 1.0
    max_retries: int = Field(default=3, ge=1)
    log_max_entries: int = Field(default=30, gt=0)
    providers: list[PatientProvider] = []

    def provider(self, model: str) -> PatientProvider:
        """The first entry whose `match` is found in the model name; a model none matches gets no params
        and a forced move."""
        return next((p for p in self.providers if re.search(p.match, model)), PatientProvider(match=""))


@dataclass(frozen=True)
class Config:
    moves: MovesFile
    archetypes: ArchetypesFile
    knob_text: KnobText
    style2: Style2File
    episode: EpisodeFile
    patient: PatientFile
    guard: GuardFile
    fingerprint: str

    def knob_line(self, knobs: Knobs, name: str) -> str:
        return getattr(self.knob_text, name)[getattr(knobs, name)]

    def knob_lines(self, knobs: Knobs, *, exclude: tuple[str, ...] = ()) -> list[str]:
        return [self.knob_line(knobs, name) for name in KNOB_NAMES if name not in exclude]


@dataclass(frozen=True)
class CallSpec:
    """One side's model, endpoint and call policy. `temperature` and `max_tokens` of None are not
    sent; `params` is merged into the litellm request. `force_move` (the patient only) names
    `commit_move` in `tool_choice` rather than offering it as `auto`. `api_key` is never recorded."""

    model: str
    api_base: str | None
    temperature: float | None
    max_tokens: int | None
    timeout: float
    max_retries: int
    params: dict = field(default_factory=dict)
    force_move: bool = True
    api_key: str | None = field(default=None, repr=False)


def _fingerprint() -> str:
    h = hashlib.sha256()
    h.update(SEED_VERSION.encode())
    for path in sorted(CONFIG_DIR.glob("*.yaml")) + sorted(PROMPTS_DIR.glob("*.j2")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _read(name: str, model: type[BaseModel]):
    return model.model_validate(yaml.safe_load((CONFIG_DIR / name).read_text()))


@lru_cache(maxsize=1)
def load_config() -> Config:
    moves = _read("moves.yaml", MovesFile)
    archetypes = _read("archetypes.yaml", ArchetypesFile)
    for aid, arch in archetypes.archetypes.items():
        unknown = set(arch.avoided_moves) - set(moves.moves)
        if unknown:
            raise ValueError(f"archetypes.yaml: {aid} avoids moves not in moves.yaml: {sorted(unknown)}")
    return Config(moves=moves, archetypes=archetypes, knob_text=_read("knobs.yaml", KnobText),
                  style2=_read("style2.yaml", Style2File), episode=_read("episode.yaml", EpisodeFile),
                  patient=_read("patient.yaml", PatientFile), guard=_read("guard.yaml", GuardFile),
                  fingerprint=_fingerprint())


@lru_cache(maxsize=1)
def _env() -> Environment:
    # StrictUndefined: a variable the caller forgot to pass raises instead of leaving a hole.
    return Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), undefined=StrictUndefined,
                       trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, autoescape=False)


def render(name: str, **context) -> str:
    """Render `prompts/<name>.j2`."""
    return _env().get_template(f"{name}.j2").render(**context)


def turn_seed(seed: int, key: str, turn: int, kind: str) -> int:
    """A per-call seed from the call's identity, so no call's draw depends on how many came before."""
    material = f"{SEED_VERSION}\x00{seed}\x00{key}\x00{turn}\x00{kind}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def provider_seed(value: int) -> int:
    """Providers that accept a `seed` generally want a signed 32-bit int."""
    return value % (2 ** 31)
