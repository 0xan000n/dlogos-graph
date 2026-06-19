"""Tests for recurring-guest resolution.

Deterministic: guest resolution depends on the SINGLE Wikidata module
(:mod:`dlogos.resolution.wikidata`). We inject a real
:class:`~dlogos.resolution.wikidata.WikidataLinker` over a fake
:class:`~dlogos.resolution.wikidata.WikidataClient` (canned candidate lists),
so no network is touched and the consolidation path is exercised end-to-end.
The headline cases — a repeated guest resolves across two episodes to one
stable QID-derived id, while a one-off guest stays per-episode — are covered
directly, alongside the intro-pattern regex and the metadata signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dlogos.resolution.wikidata import WikidataClient, WikidataLinker, WikidataMatch
from dlogos.schema import EntityType, SpeakerRef, Transcript, TranscriptSegment
from dlogos.speakers.guests import (
    GuestCandidate,
    GuestResolver,
    apply_guest_resolution,
    extract_intro_names,
)


# --------------------------------------------------------------------------- #
# Fakes + helpers
# --------------------------------------------------------------------------- #
class FakeWikidataClient:
    """Deterministic, offline Wikidata client returning canned candidates.

    Implements the single canonical ``WikidataClient`` protocol (``search`` ->
    list of candidate dicts). Records calls so tests can assert that unresolved
    (one-off) guests never incur a lookup and that domain context is forwarded.
    """

    _DB: dict[str, list[dict[str, Any]]] = {
        "jane doe": [
            {"id": "Q111", "label": "Jane Doe", "description": "economist"},
        ],
        "sam patel": [
            {"id": "Q222", "label": "Sam Patel", "description": "AI researcher"},
        ],
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, EntityType | None]] = []

    def search(
        self, name: str, *, entity_type: EntityType | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        self.calls.append((name, entity_type))
        return list(self._DB.get(name.strip().lower(), []))


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
def client() -> FakeWikidataClient:
    return FakeWikidataClient()


@pytest.fixture
def linker(client: FakeWikidataClient) -> WikidataLinker:
    """A real linker over the fake client — the single Wikidata module."""

    return WikidataLinker(client)


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
# Headline: a recurring guest resolves to one stable QID across two episodes
# --------------------------------------------------------------------------- #
def test_repeated_guest_resolves_qid_via_resolution_wikidata(
    linker: WikidataLinker,
) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=2)
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
    # The stable id is derived from the Wikidata QID returned by the single
    # resolution.wikidata module.
    assert jane.resolved.speaker_id == "guest-Q111"
    assert jane.resolved.wikidata_qid == "Q111"
    assert jane.resolved.is_host is False
    # The match carried on the resolution is the canonical WikidataMatch type.
    assert isinstance(jane.wikidata, WikidataMatch)
    assert jane.wikidata.qid == "Q111"
    # Resolved across both episodes / both shows.
    assert set(jane.episode_ids) == {"ep-1", "ep-2"}
    assert set(jane.resolved.show_ids) == {"show-1", "show-2"}


# --------------------------------------------------------------------------- #
# Headline: one-off guest stays per-episode (long tail)
# --------------------------------------------------------------------------- #
def test_one_off_guest_stays_per_episode(linker: WikidataLinker) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Sam Patel.", "Hello."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    results = {r.name: r for r in resolver.resolve()}
    sam = results["Sam Patel"]
    assert not sam.is_resolved  # only one appearance < min_appearances
    assert sam.resolved is None


def test_recurring_but_unknown_to_wikidata_stays_per_episode(
    linker: WikidataLinker,
) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=2)
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
    linker = WikidataLinker(FakeWikidataClient())
    resolver = GuestResolver(wikidata=linker, min_appearances=2, require_wikidata=False)
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
def test_metadata_only_signal_counts(linker: WikidataLinker) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=2)
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


def test_both_signals_flagged(linker: WikidataLinker) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=1)
    added = resolver.add_episode(
        _transcript("ep-1", "My guest today is Jane Doe.", "Hi."),
        show_id="show-1",
        metadata_names=["Jane Doe"],
        guest_label="SPEAKER_01",
    )
    assert len(added) == 1
    assert added[0].from_metadata and added[0].from_intro


def test_context_forwarded_to_wikidata(client: FakeWikidataClient) -> None:
    # The fake client only sees the name + type via ``search`` (context is used
    # inside the linker's matcher). To assert context reached the resolution
    # layer, use a context-sensitive candidate set where the domain hint picks a
    # different QID than relevance ordering would.
    class ContextClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, EntityType | None]] = []

        def search(
            self, name: str, *, entity_type: EntityType | None = None, limit: int = 5
        ) -> list[dict[str, Any]]:
            self.calls.append((name, entity_type))
            if name.strip().lower() == "jane doe":
                return [
                    {"id": "Q999", "label": "Jane Doe", "description": "actor"},
                    {"id": "Q111", "label": "Jane Doe", "description": "finance economist"},
                ]
            return []

    cc = ContextClient()
    resolver = GuestResolver(wikidata=WikidataLinker(cc), min_appearances=2)
    for ep in ("ep-1", "ep-2"):
        resolver.add_episode(
            _transcript(ep, "My guest today is Jane Doe.", "Hi."),
            show_id="show-1",
            guest_label="SPEAKER_01",
            context=["finance", "economics"],
        )
    jane = {r.name: r for r in resolver.resolve()}["Jane Doe"]
    # The "finance" domain hint disambiguated to the economist (Q111), not the
    # relevance-first actor (Q999) — context reached the linker's matcher.
    assert jane.resolved.wikidata_qid == "Q111"
    assert cc.calls and cc.calls[0][1] == EntityType.person


def test_unresolved_guest_not_looked_up(client: FakeWikidataClient) -> None:
    """A one-off name must not trigger a (costly) Wikidata call."""

    resolver = GuestResolver(wikidata=WikidataLinker(client), min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Sam Patel.", "Hi."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    resolver.resolve()
    assert client.calls == []


# --------------------------------------------------------------------------- #
# Writing resolution back onto SpeakerRefs
# --------------------------------------------------------------------------- #
def test_apply_guest_resolution_writes_back(linker: WikidataLinker) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=2)
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


def test_apply_guest_resolution_noop_when_unresolved(linker: WikidataLinker) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=2)
    resolver.add_episode(
        _transcript("ep-1", "My guest today is Sam Patel.", "Hi."),
        show_id="show-1",
        guest_label="SPEAKER_01",
    )
    sam = {r.name: r for r in resolver.resolve()}["Sam Patel"]
    refs = [SpeakerRef(label="SPEAKER_01")]
    apply_guest_resolution(refs, sam, episode_id="ep-1")
    assert refs[0].resolved_id is None


def test_resolution_for_other_episode_is_none(linker: WikidataLinker) -> None:
    resolver = GuestResolver(wikidata=linker, min_appearances=2)
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
# Single-module guarantee: guests depends on the one Wikidata module and does
# not define a second Wikidata client/implementation.
# --------------------------------------------------------------------------- #
def test_fake_client_satisfies_canonical_protocol(client: FakeWikidataClient) -> None:
    assert isinstance(client, WikidataClient)


def test_guests_does_not_define_a_second_wikidata_client() -> None:
    """Grep guard: only ONE module defines a Wikidata client.

    guests.py must re-use :mod:`dlogos.resolution.wikidata`, never define its
    own ``class WikidataClient`` / ``class HttpxWikidataClient`` / Wikidata
    ``WikidataMatch``. We scan the source for class definitions to lock the
    consolidation in.
    """

    src_root = Path(__file__).resolve().parents[2] / "src" / "dlogos"
    guests_src = (src_root / "speakers" / "guests.py").read_text()
    # No local class definitions of any Wikidata type in guests.py.
    for forbidden in (
        "class WikidataClient",
        "class HttpxWikidataClient",
        "class WikidataMatch",
        "class WikidataLinker",
    ):
        assert forbidden not in guests_src, (
            f"guests.py must not define {forbidden!r}; it must import the single "
            "Wikidata module dlogos.resolution.wikidata"
        )

    # Across the whole source tree exactly one module defines a Wikidata client
    # protocol/class — resolution/wikidata.py.
    definers = sorted(
        p.relative_to(src_root).as_posix()
        for p in src_root.rglob("*.py")
        if "class WikidataClient" in p.read_text()
    )
    assert definers == ["resolution/wikidata.py"], definers


def test_guests_imports_from_resolution_wikidata() -> None:
    src_root = Path(__file__).resolve().parents[2] / "src" / "dlogos"
    guests_src = (src_root / "speakers" / "guests.py").read_text()
    assert "from dlogos.resolution.wikidata import" in guests_src
