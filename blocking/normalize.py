"""
PRD 6.1 normalization: lowercase, strip punctuation/HTML entities, parse
dates into a canonical form with explicit handling of ambiguous/missing
years, resolve venue names through the alias table where known (falling
back to the raw string so fuzzy match still has something to work with).
"""
import html
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dateutil import parser as dateparser

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from venue_aliases import VENUE_ALIASES  # noqa: E402

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PUNCT_RE = re.compile(r"[^\w\s]")

# The catalog spans the 2025-26 NBA season and 2025 NFL season, i.e.
# real dates fall roughly in this window. When a listing's date has no
# year, we can't safely guess one -- picking wrong is worse than not
# filtering at all -- so callers skip the ES date filter in that case
# rather than have this module invent a year.
CATALOG_YEAR_RANGE = (2025, 2026)


def normalize_text(raw):
    if not raw:
        return ""
    s = html.unescape(raw)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_venue(raw_venue):
    if not raw_venue:
        return ""
    stripped = html.unescape(raw_venue).strip()
    resolved = VENUE_ALIASES.get(stripped, stripped)
    return normalize_text(resolved)


def normalize_date(raw_date):
    """Returns (datetime_or_None, year_was_missing: bool)."""
    if not raw_date or not raw_date.strip():
        return None, True
    year_missing = _YEAR_RE.search(raw_date) is None
    try:
        # dateutil needs SOME default; if the year is genuinely absent we
        # still parse month/day, but callers must treat the year as unreliable.
        default = datetime(CATALOG_YEAR_RANGE[1], 1, 1)
        dt = dateparser.parse(raw_date, default=default, fuzzy=True)
        return dt, year_missing
    except (ValueError, OverflowError):
        return None, True


def date_range_filter(dt, window_days=3):
    if dt is None:
        return None
    return dt - timedelta(days=window_days), dt + timedelta(days=window_days)
