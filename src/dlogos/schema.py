"""Shared domain model for dLogos.

Every other subpackage imports its types from here and must NOT redefine them.
Kept deliberately import-light: only pydantic v2 + stdlib ``datetime`` /
``enum``. No heavy/optional deps, so importing this module is always cheap.

Design references (see the PoC spec, §6): a Claim is *reified* — it carries
stance, sentiment, confidence, and a source span so it can later be
contradicted or superseded. Predicates are drawn from a *controlled
vocabulary* enforced at extraction time (a closed enum), not normalized in a
post-hoc pass. Facts loaded into the graph are *bitemporal*: both an
event-time (when it was said) and an ingestion-time (when we processed it),
with validity intervals rather than snapshots.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Stance(str, Enum):
    """How a speaker holds a claim — what makes belief-shift queryable."""

    asserts = "asserts"
    disputes = "disputes"
    hedges = "hedges"
    predicts = "predicts"
    retracts = "retracts"


class EntityType(str, Enum):
    """Domain-general entity categories (no fixed topic list)."""

    person = "person"
    organization = "organization"
    concept = "concept"
    work = "work"  # book, paper, product, episode, etc.


class Predicate(str, Enum):
    """Controlled predicate vocabulary, enforced at extraction time.

    A *closed* enum — the extractor must emit one of these normalized
    relations. This replaces a separate post-hoc normalization pass: there is
    no "free predicate then normalize" stage; the vocabulary is the contract.
    """

    expects = "expects"
    rates_positive = "rates_positive"
    rates_negative = "rates_negative"
    predicts = "predicts"
    recommends = "recommends"
    criticizes = "criticizes"
    compares = "compares"
    explains = "explains"
    attributes = "attributes"
    forecasts = "forecasts"
    endorses = "endorses"
    rejects = "rejects"
    questions = "questions"
    agrees = "agrees"
    disagrees = "disagrees"


# --------------------------------------------------------------------------- #
# Core value objects
# --------------------------------------------------------------------------- #
class SourceSpan(BaseModel):
    """A pointer back into a transcript: which episode and where in time.

    The ``t_start``/``t_end`` audio offsets are what the eval's
    speaker-verified citation check reads — "who is speaking at this
    timestamp", not merely "was the topic present".
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    t_start: float = Field(ge=0.0, description="Audio start offset, seconds.")
    t_end: float = Field(ge=0.0, description="Audio end offset, seconds.")
    transcript_offset: int | None = Field(
        default=None, description="Optional character offset into the transcript."
    )


class Entity(BaseModel):
    """A thing a claim is about or mentions.

    ``canonical_id`` is filled by the resolution stage (subject-entity
    clustering / Wikidata match); it is ``None`` straight off the extractor.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: EntityType
    canonical_id: str | None = Field(
        default=None, description="Filled by resolution; None until resolved."
    )
    qid: str | None = Field(
        default=None,
        description="Wikidata QID anchor (person/org), filled by resolution; "
        "None until anchored.",
    )


class SpeakerRef(BaseModel):
    """Reference to who is speaking.

    ``label`` is the per-episode diarization label (e.g. ``SPEAKER_00``).
    ``resolved_id`` / ``name`` are filled by cross-episode speaker identity
    (host-anchored gallery + recurring-guest resolution).
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    resolved_id: str | None = Field(
        default=None, description="Canonical Speaker id; None until resolved."
    )
    name: str | None = Field(
        default=None, description="Human-readable name; None until resolved."
    )


class TranscriptSegment(BaseModel):
    """One diarized, time-stamped utterance."""

    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(description="Per-episode diarization label.")
    text: str
    t_start: float = Field(ge=0.0)
    t_end: float = Field(ge=0.0)


class Transcript(BaseModel):
    """A full episode transcript: ordered diarized segments + metadata."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    language: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    duration_s: float = Field(ge=0.0)


class Episode(BaseModel):
    """The source unit; every derived fact traces back here."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    show_id: str
    guid: str
    title: str
    published_at: datetime
    audio_url: str
    audio_sha256: str | None = None
    transcript_ref: str | None = None


class ExtractedClaim(BaseModel):
    """The extraction output schema (§6): a reified, stance-tagged claim.

    Emitted by the open-weight extractor, conforming to the controlled
    predicate vocabulary. ``sentiment`` is a signed scalar in [-1, 1];
    ``confidence`` is the model's confidence in the extraction in [0, 1].
    """

    model_config = ConfigDict(extra="forbid")

    speaker: SpeakerRef
    predicate: Predicate
    subject_entity: Entity
    object: str = Field(description="Free text / entity / value the claim is about.")
    stance: Stance
    sentiment: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: SourceSpan


# --------------------------------------------------------------------------- #
# Bitemporal base for graph edges
# --------------------------------------------------------------------------- #
class BitemporalFact(BaseModel):
    """Mixin/base carrying the two independent time axes (§6).

    - ``event_time`` — when it was said (episode publish/recording date).
    - ``ingestion_time`` — when we processed it.
    - ``valid_from`` / ``valid_to`` — the validity interval. A fact that stops
      being true is *invalidated* (``valid_to`` set, ``invalidated=True``),
      never deleted; history is preserved without snapshotting.
    """

    model_config = ConfigDict(extra="forbid")

    event_time: datetime
    ingestion_time: datetime
    valid_from: datetime
    valid_to: datetime | None = None
    invalidated: bool = False
