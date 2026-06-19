"""Tests for WhisperX HF-token resolution from configuration.

Deterministic and dependency-light: ``WhisperXBackend`` imports ``whisperx`` /
``torch`` only inside :meth:`transcribe`, so constructing it and calling
``_resolve_hf_token`` touches no heavy dep and no network. We drive the config
read by injecting a fake ``settings`` object onto the lazily-imported
``dlogos.config`` module, asserting the real wiring (not a ``None`` stub).
"""

from __future__ import annotations

from dataclasses import dataclass

import dlogos.config as config_module
from dlogos.asr.whisperx_backend import WhisperXBackend
from dlogos.config import Settings


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@dataclass
class _FakeSettings:
    hf_token: str = ""


def _set_settings(monkeypatch, hf_token: str) -> None:
    """Swap the module-level ``settings`` singleton for a fake one."""

    monkeypatch.setattr(config_module, "settings", _FakeSettings(hf_token=hf_token))


# --------------------------------------------------------------------------- #
# The config field exists
# --------------------------------------------------------------------------- #
def test_settings_exposes_hf_token_field() -> None:
    s = Settings()
    # Default is empty (no real credential required to construct).
    assert s.hf_token == ""
    assert "hf_token" in Settings.model_fields


def test_hf_token_reads_from_env_alias(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert Settings().hf_token == "hf_from_env"


# --------------------------------------------------------------------------- #
# Resolution from settings (real wiring)
# --------------------------------------------------------------------------- #
def test_resolve_hf_token_reads_from_settings(monkeypatch) -> None:
    _set_settings(monkeypatch, "hf_secret_xyz")
    backend = WhisperXBackend(diarize=True)  # no explicit hf_token
    assert backend._resolve_hf_token() == "hf_secret_xyz"


def test_blank_settings_token_resolves_to_none(monkeypatch) -> None:
    _set_settings(monkeypatch, "")
    backend = WhisperXBackend()
    # Blank config → None so pyannote can fall back to its own env.
    assert backend._resolve_hf_token() is None


def test_whitespace_only_settings_token_resolves_to_none(monkeypatch) -> None:
    _set_settings(monkeypatch, "   ")
    backend = WhisperXBackend()
    assert backend._resolve_hf_token() is None


# --------------------------------------------------------------------------- #
# Explicit constructor token wins over settings
# --------------------------------------------------------------------------- #
def test_explicit_token_overrides_settings(monkeypatch) -> None:
    _set_settings(monkeypatch, "hf_from_settings")
    backend = WhisperXBackend(hf_token="hf_explicit")
    assert backend._resolve_hf_token() == "hf_explicit"


def test_explicit_empty_string_token_is_respected(monkeypatch) -> None:
    # An explicitly-passed "" is not None, so it wins over settings as-is
    # (caller intent: send an empty token, do not fall back).
    _set_settings(monkeypatch, "hf_from_settings")
    backend = WhisperXBackend(hf_token="")
    assert backend._resolve_hf_token() == ""
