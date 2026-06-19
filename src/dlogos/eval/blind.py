"""Deterministic blinding of arm identity for credible scoring (spec §9).

So the grader cannot reflexively reward "the dLogos one," the four answers per
query are shuffled and de-labeled before scoring. Blinding is:

- **Seed-deterministic** -- same seed + same answers -> same anonymous order
  and labels (HARD CONSTRAINT: deterministic tests, no real randomness).
- **Reversible** -- an :class:`UnblindMap` recovers which arm produced each
  anonymized answer after scoring is recorded.

The per-query shuffle uses a stable RNG seeded by ``(seed, query_id)`` so two
queries do not share an ordering (which would let a grader learn the layout).
The arm field is *stripped* from the answer the grader sees, replaced by an
opaque label like ``"answer_A"``.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dlogos.eval.arms import Answer

# Opaque labels handed to the blinded grader, in positional order.
_LABELS = tuple(f"answer_{c}" for c in "ABCDEFGH")


class BlindedAnswer(BaseModel):
    """An answer as the grader sees it: opaque label, no arm identity.

    ``text`` and ``citations`` survive (the grader needs them); ``arm`` is set
    to the empty string so the true source cannot leak through the model.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    answer: Answer = Field(
        description="Arm-stripped answer (its .arm field is blanked)."
    )


class UnblindMap(BaseModel):
    """Recovers the true arm for each opaque label, for one query."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    seed: int
    label_to_arm: dict[str, str]

    def arm_for(self, label: str) -> str:
        """Return the true arm name behind ``label`` (raises if unknown)."""

        return self.label_to_arm[label]


class BlindedQuery(BaseModel):
    """The blinded bundle for one query: shuffled answers + the unblind map."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    seed: int
    blinded: list[BlindedAnswer]
    unblind: UnblindMap


def _seed_for(seed: int, query_id: str) -> int:
    """Derive a stable per-query seed from (seed, query_id).

    Uses numpy's SeedSequence so the derivation is portable and does not depend
    on Python's salted ``hash`` (which varies between processes).
    """

    qbytes = query_id.encode("utf-8")
    # Fold the query id bytes into a single integer deterministically.
    qint = int.from_bytes(qbytes, "big") if qbytes else 0
    ss = np.random.SeedSequence([int(seed), qint])
    return int(ss.generate_state(1)[0])


def blind_answers(
    query_id: str, answers: list[Answer], *, seed: int
) -> BlindedQuery:
    """Shuffle + anonymize the answers for one query, deterministically.

    Returns a :class:`BlindedQuery` carrying the de-labeled answers (in shuffled
    order, labeled ``answer_A``, ``answer_B``, ...) and the :class:`UnblindMap`
    to reverse it. Same ``seed`` and same ``answers`` always yield the same
    order and label assignment.

    Raises ``ValueError`` if there are more answers than available labels.
    """

    if len(answers) > len(_LABELS):
        raise ValueError(
            f"too many answers ({len(answers)}) for {len(_LABELS)} labels"
        )

    rng = np.random.default_rng(_seed_for(seed, query_id))
    order = rng.permutation(len(answers)).tolist()

    blinded: list[BlindedAnswer] = []
    label_to_arm: dict[str, str] = {}
    for pos, src_idx in enumerate(order):
        label = _LABELS[pos]
        src = answers[src_idx]
        # Strip the arm identity from what the grader sees.
        stripped = src.model_copy(update={"arm": ""})
        blinded.append(BlindedAnswer(label=label, answer=stripped))
        label_to_arm[label] = src.arm

    return BlindedQuery(
        query_id=query_id,
        seed=seed,
        blinded=blinded,
        unblind=UnblindMap(
            query_id=query_id, seed=seed, label_to_arm=label_to_arm
        ),
    )


def unblind_scores(
    unblind: UnblindMap, scores_by_label: dict[str, float]
) -> dict[str, float]:
    """Map blinded per-label scores back to per-arm scores.

    ``scores_by_label`` is what the grader recorded against opaque labels;
    the result keys are the true arm names. Raises ``KeyError`` if a label has
    no entry in the unblind map.
    """

    return {unblind.arm_for(label): score for label, score in scores_by_label.items()}
