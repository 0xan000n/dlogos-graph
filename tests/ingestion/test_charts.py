"""Tests for chart-driven manifest assembly and the high-velocity subset."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from dlogos.ingestion.charts import (
    DEEP_SUBSET_MAX,
    DOMAINS,
    ChartShow,
    PodcastIndexChartSource,
    build_manifest_from_charts,
    flag_high_velocity_subset,
)
from dlogos.ingestion.podcast_index import PodcastIndexClient


@dataclass
class FakeChartSource:
    """Deterministic in-memory chart source keyed by domain."""

    by_domain: dict[str, list[ChartShow]] = field(default_factory=dict)

    def feeds_for_domain(self, domain: str, *, limit: int) -> list[ChartShow]:
        return list(self.by_domain.get(domain, []))[:limit]


def _show(show_id, domain, rank, *, epm=0.0, hosts=()):
    return ChartShow(
        show_id=show_id,
        feed_url=f"https://feeds/{show_id}.xml",
        domain=domain,
        rank=rank,
        title=show_id,
        known_hosts=tuple(hosts),
        episodes_per_month=epm,
    )


def test_build_manifest_assembles_rows_per_domain() -> None:
    source = FakeChartSource(
        by_domain={
            "technology": [_show("s1", "technology", 0), _show("s2", "technology", 1)],
            "finance": [_show("s3", "finance", 0)],
        }
    )
    m = build_manifest_from_charts(
        source, domains=("technology", "finance"), deep_min=0, deep_max=0
    )
    assert {r.show_id for r in m.rows} == {"s1", "s2", "s3"}
    assert m.get("s3").domains == ["finance"]


def test_multi_domain_show_is_merged_into_one_row() -> None:
    # s1 charts in BOTH technology and product → single row, both domains.
    source = FakeChartSource(
        by_domain={
            "technology": [_show("s1", "technology", 0, hosts=("Alice",))],
            "product": [_show("s1", "product", 3, hosts=("Bob",))],
        }
    )
    m = build_manifest_from_charts(
        source, domains=("technology", "product"), deep_min=0, deep_max=0
    )
    rows = [r for r in m.rows if r.show_id == "s1"]
    assert len(rows) == 1
    assert rows[0].domains == ["technology", "product"]
    # Host union preserved across domains.
    assert set(rows[0].known_hosts) == {"Alice", "Bob"}


def test_high_velocity_subset_picks_top_cadence() -> None:
    rows = [
        _show("hot1", "technology", 0, epm=30.0),
        _show("hot2", "finance", 1, epm=20.0),
        _show("slow", "science", 2, epm=1.0),
        _show("dead", "politics", 3, epm=0.0),
    ]
    deep = flag_high_velocity_subset(rows, min_size=0, max_size=2)
    assert deep == {"hot1", "hot2"}


def test_high_velocity_dedupes_show_across_domains_by_best_signal() -> None:
    rows = [
        _show("multi", "technology", 0, epm=5.0),
        _show("multi", "product", 1, epm=25.0),  # stronger signal wins
        _show("other", "finance", 0, epm=10.0),
    ]
    deep = flag_high_velocity_subset(rows, min_size=0, max_size=1)
    assert deep == {"multi"}


def test_build_manifest_flags_deep_subset() -> None:
    source = FakeChartSource(
        by_domain={
            "technology": [_show("fast", "technology", 0, epm=40.0)],
            "finance": [_show("slow", "finance", 0, epm=0.5)],
        }
    )
    m = build_manifest_from_charts(
        source, domains=("technology", "finance"), deep_min=0, deep_max=1
    )
    assert m.get("fast").deep_backfill is True
    assert m.get("slow").deep_backfill is False


def test_deep_subset_capped_at_max() -> None:
    rows = [_show(f"s{i}", "technology", i, epm=float(100 - i)) for i in range(50)]
    deep = flag_high_velocity_subset(rows, min_size=15, max_size=DEEP_SUBSET_MAX)
    assert len(deep) == DEEP_SUBSET_MAX


def test_eight_domains_constant() -> None:
    assert DOMAINS == (
        "science",
        "technology",
        "philosophy",
        "engineering",
        "product",
        "business",
        "finance",
        "politics",
    )


def test_live_chart_source_adapts_podcast_index_feeds() -> None:
    # Mock the PI category endpoint; verify ChartShow adaptation, no real net.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "feeds": [
                    {
                        "id": 42,
                        "url": "https://f/42.xml",
                        "title": "Deep Tech",
                        "author": "Carol",
                        "episodesPerMonth": 12,
                        "episodeCount": 240,
                    }
                ]
            },
        )

    client = PodcastIndexClient(
        api_key="k",
        api_secret="s",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    src = PodcastIndexChartSource(client=client)
    shows = src.feeds_for_domain("technology", limit=5)
    assert len(shows) == 1
    assert shows[0].show_id == "pi-42"
    assert shows[0].feed_url == "https://f/42.xml"
    assert shows[0].known_hosts == ("Carol",)
    assert shows[0].episodes_per_month == 12.0
