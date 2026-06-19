"""Tests for the corpus manifest: round-trip, uniqueness, and views."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dlogos.ingestion.manifest import (
    MANIFEST_VERSION,
    CorpusManifest,
    ManifestRow,
    load_manifest,
    save_manifest,
)


def _manifest() -> CorpusManifest:
    return CorpusManifest(
        rows=[
            ManifestRow(
                show_id="pi-1",
                feed_url="https://f/1.xml",
                domains=["technology", "product"],
                known_hosts=["Alice Host"],
                deep_backfill=True,
            ),
            ManifestRow(
                show_id="pi-2",
                feed_url="https://f/2.xml",
                domains=["finance"],
                known_hosts=[],
                deep_backfill=False,
            ),
        ]
    )


def test_manifest_round_trip(tmp_path) -> None:
    m = _manifest()
    path = tmp_path / "corpus.json"
    save_manifest(m, path)
    loaded = load_manifest(path)
    assert loaded == m
    assert loaded.version == MANIFEST_VERSION


def test_saved_manifest_is_valid_json_with_expected_shape(tmp_path) -> None:
    path = tmp_path / "corpus.json"
    save_manifest(_manifest(), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == MANIFEST_VERSION
    assert {r["show_id"] for r in raw["rows"]} == {"pi-1", "pi-2"}
    assert raw["rows"][0]["deep_backfill"] is True


def test_save_creates_parent_dirs(tmp_path) -> None:
    path = tmp_path / "nested" / "deep" / "corpus.json"
    save_manifest(_manifest(), path)
    assert path.exists()


def test_duplicate_show_id_rejected() -> None:
    with pytest.raises(ValidationError):
        CorpusManifest(
            rows=[
                ManifestRow(show_id="dup", feed_url="https://a"),
                ManifestRow(show_id="dup", feed_url="https://b"),
            ]
        )


def test_empty_show_id_or_feed_rejected() -> None:
    with pytest.raises(ValidationError):
        ManifestRow(show_id="", feed_url="https://a")
    with pytest.raises(ValidationError):
        ManifestRow(show_id="x", feed_url="   ")


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ManifestRow(show_id="x", feed_url="https://a", bogus=1)


def test_deep_backfill_and_domain_views() -> None:
    m = _manifest()
    assert [r.show_id for r in m.deep_backfill_rows()] == ["pi-1"]
    assert [r.show_id for r in m.rows_for_domain("finance")] == ["pi-2"]
    assert [r.show_id for r in m.rows_for_domain("technology")] == ["pi-1"]
    assert m.get("pi-2").feed_url == "https://f/2.xml"
    assert m.get("missing") is None
