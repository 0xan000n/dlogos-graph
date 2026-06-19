"""Tests for recurring-guest resolution.

Deterministic: a fake :class:`WikidataClient` maps known names to fixed QIDs,
so no network is touched. The headline cases — a repeated guest resolves across
two episodes to one stable id, while a one-off guest stays per-episode — are
covered directly, alongside the intro-pattern regex and the metadata signal.
"""

from __future__ import annotations

import pytest

from dlogos.schema import SpeakerRef, Transcript, TranscriptSegment
from dlogos.speakers.guests import (
    GuestCandidate,
    GuestResolver,
    HttpxWikidataClient,
    WikidataClient,
    WikidataMatch,
    apply_guest_resolution,
    extract_intro_names,
)


# --------------------------------------------------------------------------- #
# Fakes + helpers
# --------------------------------------------------------------------------- #
class FakeWikidata:
    """Maps known person names to fixed QIDs; everything else misses."""

    _TABLE = {
        "jane doe": WikidataMatch(qid="Q111", label="Jane Doe", description="economist"),
        "sam patel": WikidataMatch(qid="Q222", label="Sam Patel", description="AI researcher"),
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def lookup(self, name: str, context=None) -> WikidataMatch | None:
        self.calls.append((name, tuple(context or [])))
        return self._TABLE.get(name.strip().lower())


def _transcript(episode_id: str, intro_text: str, guest_text: str) -> Transcript:
    return Transcript(
        episode_id=episode_id,
        language="en",
        duration_s=30.0,
        segments=[
            TranscriptSegment(speaker="SPEAKER_00", text=intro_text, t_start=0.0, t_end=5.0),
            TranscriptSegment(speaker="SPEAKER_01", text=guest_text, t_start=5.0, t_end=15.0),
        ],
    )


@pytest.fixture
def wikidata() -> FakeWikidata:
    return FakeWikidata()


# --------------------------------------------------------------------------- #
# Intro-pattern extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("My guest today is Jane Doe.", "Jane Doe"),
        ("Joining me is Sam Patel today.", "Sam Patel"),
        ("I'm here with Maria Garcia on the show.", "Maria Garcia"),
        ("Today I'm joined by John Smith.", "John Smith"),
        ("Our guest is Lee O'Brien.", "Lee O'Brien"),
        ("Please welcome Ada Lovelace back.", "Ada Lovelace"),
    ],
)
def test_intro_patterns_extract_name(phrase: str, expected: str) -> None:
    t = _transcript("ep", phrase, "Thanks for having me.")
    assert extract_intro_names(t) == [expected]


def test_no_intro_returns_empty() -> None:
    t = _transcript("ep", "Welcome back to the show, everyone.", "Great to be here.")
    assert extract_intro_names(t) == []


def test_intro_dedupes_repeated_name() -> None:
    t = Transcript(
        episode_id="ep",
        language="en",
        duration_s=10.0,
        segments=[
            TranscriptSegment(speaker="SPEAKER_00", text="My guest today is Jane Doe.", t_start=0.0, t_end=4.0),
            TranscriptSegment(speaker="SPEAKER_00", text="Again, my guest today is Jane Doe.", t_start=4.0, t_end=8.0),
        ],
    )
    assert extract_intro_names(t) == ["Jane Doe"]


# --------------------------------------------------------------------------- #
# Headline: repeated guest resolves across two episodes
# --------------------------------------------------------------------------- #
def test_repeated_guest_resolves_across_two_episodes(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Jane Doe.", "Thanks."),
        show_id="show-1",
        guest_label="SPEAKER_01",
        context=["finance"],
    )
    resolver.add_episode(
        _transcript("ep-2", "Joining me is Jane Doe.", "Good to be back."),
        show_id="show-2",
        guest_label="SPEAKER_03",
        context=["finance"],
    )

    results = {r.name: r for r in resolver.resolve()}
    jane = results["Jane Doe"]
    assert jane.is_resolved
    assert jane.resolved.speaker_id == "guest-Q111"
    assert jane.resolved.wikidata_qid == "Q111"
    assert jane.resolved.is_host is False
    # Resolved across both episodes / both shows.
    assert set(jane.episode_ids) == {"ep-1", "ep-2"}
    assert set(jane.resolved.show_ids) == {"show-1", "show-2"}


# --------------------------------------------------------------------------- #
# Headline: one-off guest stays per-episode (long tail)
# --------------------------------------------------------------------------- #
def test_one_off_guest_stays_per_episode(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Sam Patel.", "Hello."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    results = {r.name: r for r in resolver.resolve()}
    sam = results["Sam Patel"]
    assert not sam.is_resolved  # only one appearance < min_appearances
    assert sam.resolved is None


def test_recurring_but_unknown_to_wikidata_stays_per_episode(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    for ep in ("ep-1", "ep-2"):
        resolver.add_episode(
            _transcript(ep, "My guest today is Nobody Famous.", "Hi."),
            show_id="show-1",
            guest_label="SPEAKER_01",
        )
    results = {r.name: r for r in resolver.resolve()}
    nf = results["Nobody Famous"]
    # Recurs (2 episodes) but Wikidata misses → not promoted (require_wikidata).
    assert not nf.is_resolved
    assert nf.wikidata is None


def test_require_wikidata_false_promotes_recurring_unknown() -> None:
    resolver = GuestResolver(wikidata=FakeWikidata(), min_appearances=2, require_wikidata=False)
    for ep in ("ep-1", "ep-2"):
        resolver.add_episode(
            _transcript(ep, "My guest today is Nobody Famous.", "Hi."),
            show_id="show-1",
            guest_label="SPEAKER_01",
        )
    nf = {r.name: r for r in resolver.resolve()}["Nobody Famous"]
    assert nf.is_resolved
    assert nf.resolved.wikidata_qid is None
    assert nf.resolved.speaker_id == "guest-nobody-famous"


# --------------------------------------------------------------------------- #
# Signals: metadata, both-signal flags, context passed to Wikidata
# --------------------------------------------------------------------------- #
def test_metadata_only_signal_counts(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    for ep in ("ep-1", "ep-2"):
        resolver.add_episode(
            _transcript(ep, "Welcome back, everyone.", "Glad to be here."),
            show_id="show-1",
            metadata_names=["Jane Doe"],
            guest_label="SPEAKER_01",
        )
    jane = {r.name: r for r in resolver.resolve()}["Jane Doe"]
    assert jane.is_resolved
    assert all(a.from_metadata and not a.from_intro for a in jane.appearances)


def test_both_signals_flagged(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=1)
    added = resolver.add_episode(
        _transcript("ep-1", "My guest today is Jane Doe.", "Hi."),
        show_id="show-1",
        metadata_names=["Jane Doe"],
        guest_label="SPEAKER_01",
    )
    assert len(added) == 1
    assert added[0].from_metadata and added[0].from_intro


def test_context_forwarded_to_wikidata(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    for ep in ("ep-1", "ep-2"):
        resolver.add_episode(
            _transcript(ep, "My guest today is Jane Doe.", "Hi."),
            show_id="show-1",
            guest_label="SPEAKER_01",
            context=["finance", "economics"],
        )
    resolver.resolve()
    # The disambiguation context reached the client.
    assert any("finance" in ctx for _, ctx in wikidata.calls)


def test_unresolved_guest_not_looked_up(wikidata: FakeWikidata) -> None:
    """A one-off name must not trigger a (costly) Wikidata call."""

    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Sam Patel.", "Hi."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    resolver.resolve()
    assert wikidata.calls == []


# --------------------------------------------------------------------------- #
# Writing resolution back onto SpeakerRefs
# --------------------------------------------------------------------------- #
def test_apply_guest_resolution_writes_back(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Jane Doe.", "Hi."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    resolver.add_episode(
        _transcript("ep-2", "Joining me is Jane Doe.", "Back again."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    jane = {r.name: r for r in resolver.resolve()}["Jane Doe"]

    refs = [
        SpeakerRef(label="SPEAKER_00"),  # host turn — untouched
        SpeakerRef(label="SPEAKER_01"),  # the guest — should be filled
    ]
    apply_guest_resolution(refs, jane, episode_id="ep-1")

    assert refs[0].resolved_id is None
    assert refs[1].resolved_id == "guest-Q111"
    assert refs[1].name == "Jane Doe"


def test_apply_guest_resolution_noop_when_unresolved(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Sam Patel.", "Hi."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    sam = {r.name: r for r in resolver.resolve()}["Sam Patel"]
    refs = [SpeakerRef(label="SPEAKER_01")]
    apply_guest_resolution(refs, sam, episode_id="ep-1")
    assert refs[0].resolved_id is None


def test_resolution_for_other_episode_is_none(wikidata: FakeWikidata) -> None:
    resolver = GuestResolver(wikidata=wikidata, min_appearances=2)
    for ep in ("ep-1", "ep-2"):
        resolver.add_episode(
            _transcript(ep, "My guest today is Jane Doe.", "Hi."),
            show_id="show-1",
            guest_label="SPEAKER_01",
        )
    jane = {r.name: r for r in resolver.resolve()}["Jane Doe"]
    assert jane.resolution_for("ep-1") is not None
    assert jane.resolution_for("ep-999") is None


# --------------------------------------------------------------------------- #
# Protocol conformance + lazy httpx default
# --------------------------------------------------------------------------- #
def test_fake_satisfies_protocol(wikidata: FakeWikidata) -> None:
    assert isinstance(wikidata, WikidataClient)


def test_httpx_client_does_not_import_at_module_load() -> None:
    # Constructing the default client must not require httpx at import time;
    # httpx is imported lazily inside lookup(). Construction alone is cheap.
    client = HttpxWikidataClient()
    assert client.endpoint.startswith("https://")
    assert isinstance(client, WikidataClient)
