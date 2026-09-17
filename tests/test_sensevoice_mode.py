"""
tests/test_sensevoice_mode.py
------------------------------
Where SenseVoice runs is decided by SENSEVOICE_MODE, falling back to whether
SENSEVOICE_REMOTE_URL is set. The case that matters: a local-ASR deployment
sharing a .env that sets a remote URL must still run locally.
"""

import pytest

from workers.verbal_worker import _resolve_remote_url

URL = "https://example.ngrok-free.dev/transcribe"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("SENSEVOICE_MODE", raising=False)
    monkeypatch.delenv("SENSEVOICE_REMOTE_URL", raising=False)


def test_unset_mode_follows_url(monkeypatch):
    assert _resolve_remote_url() is None
    monkeypatch.setenv("SENSEVOICE_REMOTE_URL", URL)
    assert _resolve_remote_url() == URL


def test_local_mode_ignores_a_configured_url(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_REMOTE_URL", URL)
    monkeypatch.setenv("SENSEVOICE_MODE", "local")
    assert _resolve_remote_url() is None


def test_mode_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_REMOTE_URL", URL)
    monkeypatch.setenv("SENSEVOICE_MODE", " Local ")
    assert _resolve_remote_url() is None


def test_remote_mode_uses_url(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_REMOTE_URL", URL)
    monkeypatch.setenv("SENSEVOICE_MODE", "remote")
    assert _resolve_remote_url() == URL


def test_remote_mode_without_url_is_an_error_not_a_silent_local_load(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_MODE", "remote")
    with pytest.raises(ValueError, match="SENSEVOICE_REMOTE_URL"):
        _resolve_remote_url()


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_MODE", "colab")
    with pytest.raises(ValueError, match="expected one of"):
        _resolve_remote_url()
