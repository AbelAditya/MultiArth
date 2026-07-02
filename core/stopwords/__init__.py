from __future__ import annotations
from pathlib import Path

_DIR = Path(__file__).parent


def load_stopwords(lang_code: str) -> set[str]:
    """
    Return the curated stop word set for *lang_code*, or an empty set if
    no file exists for that language.  Lines starting with '#' are comments.
    """
    path = _DIR / f"{lang_code}.txt"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as fh:
        return {
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#")
        }
