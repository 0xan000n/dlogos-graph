"""Tests for the dLogos MCP handler functions (spec §8, §9).

These exercise the handlers against a *fake* RetrievalSurface with NO ``mcp``
import — the whole point of the handler/server split is that the tool logic is
testable with the core dependency group only. Determinism comes from a fixed
set of :class:`RetrievableClaim`\\ s and the pure ``consensus_over_time`` over
the conftest synthetic claims.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from dlogos.mcp.server import (
    BeliefHistoryResult,
    ConsensusTrendResult,
    ProvenanceResult,
    RetrievalSurface,
    SearchDialogueResult,
    WhoDiscussedResult,
    belief_history_handler,
    consensus_trend_handler,
    provenance_lookup_handler,
    search_dialogue_handler,
    who_discussed_handler,
)
from dlogos.retrieval.consensus import consensus_over_time
from dlogos.retrieval.hybrid import RetrievableClaim, RetrievalResult
from dlogos.schema import SourceSpan, Stance


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _claim(
    claim_id: str,
    text: str,
    *,
    speaker_id: str | None,
    subject_id: str | None = "apple",
    stance: Stance | None = Stance.asserts,
    episode_id: str = "ep-0001",
    t_start: float = 4.5,
    t_end: float = 10.0,
    event_time: datetime | None = None,
) -> RetrievableClaim:
    return RetrievableClaim(
        claim_id=claim_id,
        text=text,
        speaker_id=speaker_id,
        subject_id=subject_id,
        stance=stance,
        source_span=SourceSpan(
            episode_id=episode_id, t_start=t_start, t_end=t_end
        ),
        event_time=event_time,
    )


class FakeRetrievalSurface:
    """Deterministic RetrievalSurface backed by fixed claims + real consensus.

    ``search`` returns the claims in a fixed order (wrapped as RetrievalResults
    with descending scores), honoring an event-time window when given.
    ``consensus`` runs the pure :func:`consensus_over_time` over injected
    ExtractedClaims. ``provenance`` resolves by claim id.
    """

    def __init__(self, claims, *, consensus_claims=None, event_times=None) -> None:
        self._claims = list(claims)
        self._by_id = {c.claim_id: c for c in self._claims}
        self._consensus_claims = consensus_claims or []
        self._event_times = event_times or {}
        self.search_calls: list[tuple] = []

    def search(self, query, *, top_k=10, since=None, until=None):
        self.search_calls.append((query, top_k, since, until))
        rows = self._claims
        if since is not None or until is not None:
            rows = [
                c
                for c in rows
                if c.event_time is not None
                and (since is None or c.event_time >= since)
                and (until is None or c.event_time <= until)
            ]
        rows = rows[:top_k]
        n = len(rows)
        return [
            RetrievalResult(claim=c, score=float(n - i))
            for i, c in enumerate(rows)
        ]

    def consensus(self, subject, *, window_days=30):
        from datetime import timedelta

        return consensus_over_time(
            self._consensus_claims,
            self._event_times,
            subject=subject,
            bucket=timedelta(days=window_days),
        )

    def provenance(self, claim_ref):
        claim = self._by_id.get(claim_ref)
        if claim is None:
            return None
        return RetrievalResult(claim=claim, score=1.0)


def _surface_with_search():
    return FakeRetrievalSurface(
        [
            _claim(
                "c1",
                "Apple hardware has plateaued",
                speaker_id="spk-analyst",
                event_time=_dt(2026, 1, 10),
            ),
            _claim(
                "c2",
                "Apple services growth offsets hardware",
                speaker_id="spk-host",
                episode_id="ep-0002",
                event_time=_dt(2026, 2, 15),
            ),
            # A second hit from the same speaker — must be de-duped by who_discussed.
            _claim(
                "c3",
                "Apple still leads on silicon",
                speaker_id="spk-analyst",
                episode_id="ep-0003",
                event_time=_dt(2026, 4, 1),
            ),
            # A hit with no resolved speaker — who_discussed must skip it.
            _claim(
                "c4",
                "unattributed aside about Apple",
                speaker_id=None,
                episode_id="ep-0004",
                event_time=_dt(2026, 5, 20),
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# The whole point: handlers do not require the mcp package
# --------------------------------------------------------------------------- #
def test_handlers_do_not_import_mcp() -> None:
    surface = _surface_with_search()
    search_dialogue_handler(surface, "Apple")
    assert "mcp.server" not in sys.modules or True  # our own package is dlogos.mcp
    # The real assertion: the third-party `mcp` package must NOT be loaded.
    assert "mcp" not in sys.modules


def test_search_dialogue_flattens_hits_in_rank_order() -> None:
    surface = _surface_with_search()
    res = search_dialogue_handler(surface, "Apple hardware", top_k=3)
    assert isinstance(res, SearchDialogueResult)
    assert res.query == "Apple hardware"
    assert [h.claim_id for h in res.hits] == ["c1", "c2", "c3"]
    # Scores descend with rank; spans + speaker carried through.
    assert res.hits[0].score >= res.hits[1].score >= res.hits[2].score
    assert res.hits[0].speaker_id == "spk-analyst"
    assert res.hits[0].episode_id == "ep-0001"
    assert res.hits[0].t_start == 4.5
    assert res.hits[0].stance == "asserts"
    # top_k is honored.
    assert len(res.hits) == 3
    assert surface.search_calls[-1][1] == 3


def test_search_dialogue_applies_temporal_window() -> None:
    surface = _surface_with_search()
    res = search_dialogue_handler(
        surface, "Apple", since=_dt(2026, 2, 1), until=_dt(2026, 4, 30)
    )
    ids = {h.claim_id for h in res.hits}
    assert ids == {"c2", "c3"}  # Jan and May excluded by the window
    assert res.since == _dt(2026, 2, 1)
    assert res.until == _dt(2026, 4, 30)


def test_who_discussed_dedupes_speakers_and_skips_unattributed() -> None:
    surface = _surface_with_search()
    res = who_discussed_handler(surface, "Apple")
    assert isinstance(res, WhoDiscussedResult)
    # spk-analyst appears twice (c1, c3) -> once; spk-host once; c4 has no
    # speaker -> dropped entirely.
    assert res.speakers == ["spk-analyst", "spk-host"]
    assert len(res.mentions) == 2
    # First mention per speaker is kept (c1 for the analyst).
    analyst = next(m for m in res.mentions if m.speaker_id == "spk-analyst")
    assert analyst.episode_id == "ep-0001"
    assert analyst.snippet == "Apple hardware has plateaued"


def test_who_discussed_since_passes_through_to_search() -> None:
    surface = _surface_with_search()
    who_discussed_handler(surface, "Apple", since=_dt(2026, 3, 1))
    # since is forwarded to the surface as the temporal filter.
    last = surface.search_calls[-1]
    assert last[2] == _dt(2026, 3, 1)


def test_consensus_trend_flattens_the_trend(
    synthetic_claims, claim_event_times
) -> None:
    surface = FakeRetrievalSurface(
        [],
        consensus_claims=synthetic_claims,
        event_times=claim_event_times,
    )
    res = consensus_trend_handler(surface, "Apple", window_days=30)
    assert isinstance(res, ConsensusTrendResult)
    assert res.subject == "Apple"
    # Several attributed speakers across the timeline.
    assert "spk-analyst" in res.all_speakers
    assert "spk-host" in res.all_speakers
    assert "spk-guest-b" in res.all_speakers
    # Direction is one of the enum string values.
    assert res.direction in {
        "rising",
        "falling",
        "flat",
        "mixed",
        "insufficient",
    }
    # Points carry per-bucket structure.
    assert res.points
    assert all(p.start <= p.end for p in res.points)
    populated = [p for p in res.points if p.claim_count > 0]
    assert populated and all(p.speakers for p in populated)


def test_belief_history_filters_to_one_person(
    synthetic_claims, claim_event_times
) -> None:
    surface = FakeRetrievalSurface(
        [],
        consensus_claims=synthetic_claims,
        event_times=claim_event_times,
    )
    res = belief_history_handler(surface, "spk-analyst", "Apple")
    assert isinstance(res, BeliefHistoryResult)
    assert res.person == "spk-analyst"
    assert res.found is True
    # The analyst contributes to >= 1 bucket (ep-0001 negative, ep-0003 expects).
    assert res.points
    # Every point reflects only the analyst's claims being present.
    assert all(p.claim_count >= 1 for p in res.points)
    assert res.direction in {"rising", "falling", "flat", "insufficient"}


def test_belief_history_unknown_person_is_not_found(
    synthetic_claims, claim_event_times
) -> None:
    surface = FakeRetrievalSurface(
        [],
        consensus_claims=synthetic_claims,
        event_times=claim_event_times,
    )
    res = belief_history_handler(surface, "spk-nobody", "Apple")
    assert res.found is False
    assert res.points == []
    assert res.direction == "insufficient"


def test_provenance_lookup_resolves_and_misses() -> None:
    surface = _surface_with_search()
    hit = provenance_lookup_handler(surface, "c2")
    assert isinstance(hit, ProvenanceResult)
    assert hit.found is True
    assert hit.claim_id == "c2"
    assert hit.episode_id == "ep-0002"
    assert hit.speaker_id == "spk-host"
    assert hit.t_start == 4.5

    miss = provenance_lookup_handler(surface, "does-not-exist")
    assert miss.found is False
    assert miss.claim_id is None
    assert miss.episode_id is None


def test_fake_surface_satisfies_the_protocol() -> None:
    # Structural check: the fake is a valid RetrievalSurface.
    assert isinstance(_surface_with_search(), RetrievalSurface)
