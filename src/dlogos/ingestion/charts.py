"""Build the corpus manifest from public category charts (spec §4).

Selection is chart-driven, not hand-built, so the corpus is reproducible and
defensible as "the top shows" with coverage spanning the eight domains. We
rank by chart position within each domain, take a target number of shows per
domain to reach ~200, and flag a ~15–25 show high-velocity subset for the
deep-tier (~18–24 month) backfill.

Network access (hitting the Podcast Index for category feeds) lives in
:meth:`ChartSource.feeds_for_domain` on the live source only; the pure manifest
assembly (:func:`build_manifest_from_charts`) takes already-fetched chart rows,
so it is fully testable without any network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dlogos.ingestion.manifest import CorpusManifest, ManifestRow
from dlogos.ingestion.podcast_index import PodcastIndexClient

# The eight knowledge-economy domains (spec §3, line 56). Ordering is stable so
# the assembled manifest is reproducible.
DOMAINS: tuple[str, ...] = (
    "science",
    "technology",
    "philosophy",
    "engineering",
    "product",
    "business",
    "finance",
    "politics",
)

# Default corpus target — ~200 shows spread across the eight domains.
DEFAULT_TARGET_TOTAL = 200
# Deep-tier high-velocity subset size band (spec §4b).
DEEP_SUBSET_MIN = 15
DEEP_SUBSET_MAX = 25


@dataclass(frozen=True)
class ChartShow:
    """One chart entry for a show, ranked within a domain.

    ``episodes_per_month`` and ``recent_episode_count`` are publish-cadence
    signals (from feed metadata) used to pick the high-velocity deep subset.
    """

    show_id: str
    feed_url: str
    domain: str
    rank: int
    title: str = ""
    known_hosts: tuple[str, ...] = ()
    episodes_per_month: float = 0.0
    recent_episode_count: int = 0


class ChartSource(Protocol):
    """Anything that can yield ranked chart rows for a domain.

    The live implementation hits the Podcast Index; tests pass an in-memory
    fake, so :func:`build_manifest_from_charts` never needs the network.
    """

    def feeds_for_domain(self, domain: str, *, limit: int) -> list[ChartShow]: ...


@dataclass
class PodcastIndexChartSource:
    """Live chart source backed by the Podcast Index category listing.

    Maps each abstract domain to one or more Podcast Index categories, fetches
    the category feeds, and adapts them into :class:`ChartShow` rows. The only
    network call is inside :meth:`feeds_for_domain`.
    """

    client: PodcastIndexClient
    # Domain → Podcast Index category names. A domain may map to several
    # categories; ranks are assigned by listing order within the domain.
    domain_categories: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "science": ("Science",),
            "technology": ("Technology",),
            "philosophy": ("Philosophy",),
            "engineering": ("Technology", "Science"),
            "product": ("Business", "Technology"),
            "business": ("Business",),
            "finance": ("Investing", "Business"),
            "politics": ("Politics", "News"),
        }
    )

    def feeds_for_domain(self, domain: str, *, limit: int) -> list[ChartShow]:
        categories = self.domain_categories.get(domain, (domain.capitalize(),))
        rows: list[ChartShow] = []
        seen_feeds: set[str] = set()
        rank = 0
        for cat in categories:
            feeds = self.client.feeds_by_category(cat, max_results=limit)
            for f in feeds:
                feed_url = str(f.get("url") or f.get("originalUrl") or "")
                if not feed_url or feed_url in seen_feeds:
                    continue
                seen_feeds.add(feed_url)
                show_id = _show_id_from_feed(f, domain)
                hosts = _hosts_from_feed(f)
                rows.append(
                    ChartShow(
                        show_id=show_id,
                        feed_url=feed_url,
                        domain=domain,
                        rank=rank,
                        title=str(f.get("title", "")),
                        known_hosts=hosts,
                        episodes_per_month=float(_episodes_per_month(f)),
                        recent_episode_count=int(f.get("episodeCount", 0) or 0),
                    )
                )
                rank += 1
                if rank >= limit:
                    break
            if rank >= limit:
                break
        return rows


def _show_id_from_feed(feed: dict, domain: str) -> str:
    """Derive a stable show id from feed metadata (prefer the PI feed id)."""

    pi_id = feed.get("id")
    if pi_id is not None:
        return f"pi-{pi_id}"
    podcast_guid = feed.get("podcastGuid")
    if podcast_guid:
        return f"guid-{podcast_guid}"
    # Fall back to a slug of the title scoped by domain.
    title = str(feed.get("title", "untitled")).strip().lower()
    slug = "".join(c if c.isalnum() else "-" for c in title).strip("-") or "untitled"
    return f"{domain}-{slug}"


def _hosts_from_feed(feed: dict) -> tuple[str, ...]:
    """Best-effort host extraction from common feed fields."""

    author = feed.get("author") or feed.get("ownerName")
    if author:
        return (str(author),)
    return ()


def _episodes_per_month(feed: dict) -> float:
    """Cadence estimate from feed metadata, defensive to missing fields."""

    val = feed.get("episodesPerMonth")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def flag_high_velocity_subset(
    rows: list[ChartShow],
    *,
    min_size: int = DEEP_SUBSET_MIN,
    max_size: int = DEEP_SUBSET_MAX,
) -> set[str]:
    """Select the deep-tier high-velocity subset by publish cadence.

    Returns the set of ``show_id``s to mark ``deep_backfill=True``. Shows are
    ranked by ``episodes_per_month`` then ``recent_episode_count`` (both higher
    = more revisits of a topic, the signal the temporal-shift demo needs), with
    chart ``rank`` as a stable tiebreaker. We take up to ``max_size``; if fewer
    than ``min_size`` shows carry a positive cadence signal we still return at
    most what is available (callers reviewing the manifest can top it up).
    """

    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")

    # Deduplicate to one entry per show (a show can chart in several domains);
    # keep the entry with the strongest cadence signal.
    best: dict[str, ChartShow] = {}
    for r in rows:
        cur = best.get(r.show_id)
        if cur is None or _velocity_key(r) > _velocity_key(cur):
            best[r.show_id] = r

    ranked = sorted(best.values(), key=_velocity_sort_key)
    # Prefer shows with any positive cadence signal first.
    with_signal = [r for r in ranked if _velocity_key(r) > (0.0, 0)]
    chosen = with_signal[:max_size]
    if len(chosen) < min_size:
        # Backfill from the remaining ranked shows to reach min_size if we can.
        remaining = [r for r in ranked if r.show_id not in {c.show_id for c in chosen}]
        chosen = chosen + remaining[: max(0, min_size - len(chosen))]
    return {r.show_id for r in chosen}


def _velocity_key(show: ChartShow) -> tuple[float, int]:
    return (show.episodes_per_month, show.recent_episode_count)


def _velocity_sort_key(show: ChartShow) -> tuple[float, int, int]:
    # Higher cadence first (negate), then lower chart rank as a stable tiebreak.
    return (-show.episodes_per_month, -show.recent_episode_count, show.rank)


def build_manifest_from_charts(
    source: ChartSource,
    *,
    domains: tuple[str, ...] = DOMAINS,
    target_total: int = DEFAULT_TARGET_TOTAL,
    deep_min: int = DEEP_SUBSET_MIN,
    deep_max: int = DEEP_SUBSET_MAX,
    per_domain_limit: int | None = None,
) -> CorpusManifest:
    """Assemble a :class:`CorpusManifest` from ranked chart rows.

    Pure given a ``source``: pass a fake source in tests for full determinism.
    A show charting in multiple domains is merged into one manifest row whose
    ``domains`` lists every domain it appears in (so coverage is recorded
    without duplicating the show). The deep-backfill flag is set on the
    high-velocity subset chosen across the full pulled set.
    """

    per_domain = (
        per_domain_limit
        if per_domain_limit is not None
        else _per_domain_quota(target_total, len(domains))
    )

    all_rows: list[ChartShow] = []
    # Merge multi-domain shows: show_id → (row, set of domains, host union).
    merged: dict[str, tuple[ChartShow, list[str], list[str]]] = {}
    order: list[str] = []

    for domain in domains:
        for show in source.feeds_for_domain(domain, limit=per_domain):
            all_rows.append(show)
            if show.show_id not in merged:
                merged[show.show_id] = (show, [show.domain], list(show.known_hosts))
                order.append(show.show_id)
            else:
                _, domain_tags, hosts = merged[show.show_id]
                if show.domain not in domain_tags:
                    domain_tags.append(show.domain)
                for h in show.known_hosts:
                    if h not in hosts:
                        hosts.append(h)

    deep_ids = flag_high_velocity_subset(
        all_rows, min_size=deep_min, max_size=deep_max
    )

    rows: list[ManifestRow] = []
    for show_id in order:
        show, domain_tags, hosts = merged[show_id]
        rows.append(
            ManifestRow(
                show_id=show_id,
                feed_url=show.feed_url,
                domains=domain_tags,
                known_hosts=hosts,
                deep_backfill=show_id in deep_ids,
            )
        )
    return CorpusManifest(rows=rows)


def _per_domain_quota(target_total: int, n_domains: int) -> int:
    """Even per-domain quota that reaches at least ``target_total`` overall."""

    if n_domains <= 0:
        raise ValueError("need at least one domain")
    # Ceil division so the union meets/exceeds the target before de-dup.
    return -(-target_total // n_domains)
