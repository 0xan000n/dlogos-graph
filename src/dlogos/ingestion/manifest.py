"""The corpus manifest (spec §4): the reviewed input artifact for backfill.

One row per show — ``feed_url``, canonical ``show_id``, ``domains`` tag(s),
``known_hosts`` (for voiceprint-gallery seeding) with optional per-host
``reference_audio`` refs, and a ``deep_backfill`` flag marking the ~15–25
high-velocity subset that carries the temporal-shift demos.

The manifest is a versioned JSON config. ``load_manifest`` / ``save_manifest``
round-trip it losslessly through Pydantic v2 so it can be hand-reviewed,
checked into the repo, and reloaded deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Manifest schema version — bump when the row shape changes incompatibly.
MANIFEST_VERSION = 1


class ManifestRow(BaseModel):
    """A single show in the corpus manifest."""

    model_config = ConfigDict(extra="forbid")

    show_id: str = Field(description="Canonical, stable show identifier.")
    feed_url: str = Field(description="Clean RSS feed URL (from Podcast Index).")
    domains: list[str] = Field(
        default_factory=list,
        description="One or more of the eight domain tags (see charts.DOMAINS).",
    )
    known_hosts: list[str] = Field(
        default_factory=list,
        description="Host names; seeds the host-anchored voiceprint gallery (§7.3).",
    )
    reference_audio: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional per-host reference-audio ref, keyed by host name "
            "(must be one of ``known_hosts``). Seeds the voiceprint gallery's "
            "sample refs (§7.3); hosts without a ref still define a canonical "
            "speaker but cannot be voiceprint-matched."
        ),
    )
    deep_backfill: bool = Field(
        default=False,
        description="True for the ~15–25 high-velocity deep-tier subset (§4b).",
    )

    @field_validator("show_id", "feed_url")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("show_id and feed_url must be non-empty")
        return v

    @model_validator(mode="after")
    def _reference_audio_keys_are_known_hosts(self) -> "ManifestRow":
        unknown = set(self.reference_audio) - set(self.known_hosts)
        if unknown:
            raise ValueError(
                "reference_audio keys must be listed in known_hosts; "
                f"unknown host(s): {sorted(unknown)!r}"
            )
        return self


class CorpusManifest(BaseModel):
    """The full versioned manifest: an ordered list of unique-show rows."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=MANIFEST_VERSION)
    rows: list[ManifestRow] = Field(default_factory=list)

    @field_validator("rows")
    @classmethod
    def _unique_show_ids(cls, rows: list[ManifestRow]) -> list[ManifestRow]:
        seen: set[str] = set()
        for r in rows:
            if r.show_id in seen:
                raise ValueError(f"duplicate show_id in manifest: {r.show_id!r}")
            seen.add(r.show_id)
        return rows

    # -- convenience views -------------------------------------------------- #
    def deep_backfill_rows(self) -> list[ManifestRow]:
        """The deep-tier (~18–24 month) high-velocity subset."""

        return [r for r in self.rows if r.deep_backfill]

    def rows_for_domain(self, domain: str) -> list[ManifestRow]:
        """All shows tagged with ``domain``."""

        return [r for r in self.rows if domain in r.domains]

    def get(self, show_id: str) -> ManifestRow | None:
        """Look up a single row by ``show_id`` (None if absent)."""

        for r in self.rows:
            if r.show_id == show_id:
                return r
        return None


def save_manifest(manifest: CorpusManifest, path: str | Path) -> None:
    """Write the manifest to ``path`` as pretty, stable JSON."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    p.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> CorpusManifest:
    """Load and validate a manifest from ``path``."""

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return CorpusManifest.model_validate(data)
