"""
core/drive_download.py
------------------------
Fetch a Google Drive file's bytes by its shareable link, via the Drive API
(not the /preview embed used for playback elsewhere in the dashboard, which
isn't a direct-download URL and can't be fetched programmatically).

Used by BulkOrchestrator when a manifest entry's local `path` doesn't exist
yet — e.g. the dashboard's Bulk Upload tab, where entries only have a
`drive_url` and no pre-staged local file (unlike `analyze bulk` from the
CLI, where videos are manually staged in advance).
"""

from __future__ import annotations

import os
import re

import requests

_DRIVE_ID_RE = re.compile(r"/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_QUERY_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")
_DRIVE_API_URL = "https://www.googleapis.com/drive/v3/files/{file_id}"
_CHUNK_SIZE = 1024 * 1024


def extract_file_id(drive_url: str) -> str:
    m = _DRIVE_ID_RE.search(drive_url) or _DRIVE_ID_QUERY_RE.search(drive_url)
    if not m:
        raise ValueError(f"Could not extract a Drive file ID from {drive_url!r}")
    return m.group(1)


def download_drive_file(drive_url: str, dest_path: str, api_key: str | None = None) -> None:
    """
    Download a publicly-shared ("anyone with the link") Google Drive file to
    *dest_path*. Streams to a temp file first and only renames it into place
    on success, so a failed/interrupted download never leaves a partial file
    at the real destination.
    """
    api_key = api_key or os.environ.get("GOOGLE_DRIVE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_DRIVE_API_KEY not configured — cannot fetch from Google Drive"
        )

    file_id = extract_file_id(drive_url)
    url = _DRIVE_API_URL.format(file_id=file_id)
    tmp_path = f"{dest_path}.part"

    try:
        with requests.get(
            url, params={"alt": "media", "key": api_key}, stream=True, timeout=60
        ) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
