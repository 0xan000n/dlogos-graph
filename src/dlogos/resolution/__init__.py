"""Lightweight resolution (spec §7.4a) — runs as a batch BEFORE graph load.

This is the deliberately small slice of entity resolution the
temporal-consensus claim depends on, run as **our own batch module** over the
full extracted-claim set *before* the graph load — so the Graphiti bulk
loader can bypass Graphiti's per-add LLM node-dedup (§7.5/§7.6).

Two public levers live here:

- :mod:`dlogos.resolution.subjects` — subject-entity embedding clustering, so
  "consensus about X" does not fragment across surface forms
  (*Apple* / *iPhone* / *Apple hardware*). Assigns a shared ``canonical_id``.
- :mod:`dlogos.resolution.wikidata` — lightweight Wikidata linking, attaching a
  stable QID to known people/orgs (the recurring-guest / known-entity lever).

The third lever — the controlled predicate vocabulary — is enforced *upstream*
at extraction time (the closed :class:`dlogos.schema.Predicate` enum); there is
no separate post-hoc predicate-normalization pass, so nothing for it lives here.

Heavy/optional deps (FlagEmbedding, sentence-transformers, httpx clients to
real services) are imported LAZILY inside functions; embedders and HTTP
clients are INJECTED so unit tests run on core deps with no network.
"""

from __future__ import annotations

from dlogos.resolution.subjects import (
    Embedder,
    EntityCluster,
    SubjectResolution,
    cluster_entities,
    resolve_subjects,
)
from dlogos.resolution.wikidata import (
    WikidataClient,
    WikidataLinker,
    WikidataMatch,
    link_entities,
)

__all__ = [
    "Embedder",
    "EntityCluster",
    "SubjectResolution",
    "cluster_entities",
    "resolve_subjects",
    "WikidataClient",
    "WikidataLinker",
    "WikidataMatch",
    "link_entities",
]
