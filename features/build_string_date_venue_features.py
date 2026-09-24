"""
PRD 6.3 (phase 3a): string/date/venue feature extraction on top of
features/build_pairs.py's candidate_pairs.csv.

Reads data/processed/candidate_pairs.csv (101,044 rows, one per
listing/candidate pair from blocking) and writes
features/string_date_venue_features.csv with the same number of rows,
keyed by (listing_id, candidate_event_id) plus label/split copied
through unchanged, plus this module's feature columns.

Scope: string similarity, date proximity, venue match, and a source
one-hot -- explicitly NOT embeddings (owned by a parallel agent) and NOT
the LightGBM model (a later phase). See docs/phase3a_string_features.md
for the write-up.
"""
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import jellyfish
from dateutil import parser as dateparser
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent.parent / "blocking"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from normalize import normalize_date, normalize_text, normalize_venue  # noqa: E402
from venue_aliases import VENUE_ALIASES  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_PATH = Path(__file__).parent / "string_date_venue_features.csv"

_TOKEN_RE = re.compile(r"\w+")

# Sentinel used for day_diff when the listing date can't be parsed at all --
# paired with date_missing=1 so a model can distinguish "far apart" from
# "unknown" rather than silently treating them the same.
DATE_MISSING_SENTINEL = 999

SOURCES = ["official_feed", "partner_feed_a", "scraped_feed_b", "scraped_feed_c"]


def tokenize(s):
    return set(_TOKEN_RE.findall(s)) if s else set()


def jaccard(a_tokens, b_tokens):
    if not a_tokens and not b_tokens:
        return 0.0
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def build_venue_coord_lookup():
    """venue_canonical (normalized) -> (lat, lon), from the events catalog.

    candidate_pairs.csv only carries lat/lon for the CANDIDATE side; the
    listing side has no coordinates of its own. If the listing's raw venue
    (after alias resolution) matches a known catalog venue, we can look up
    that venue's coordinates and still compute a geo-distance feature.
    When it doesn't resolve, geo_distance is left missing (flagged), not
    guessed.
    """
    lookup = {}
    with (DATA_DIR / "events_catalog.csv").open() as f:
        for row in csv.DictReader(f):
            key = normalize_text(row["venue_canonical"])
            if key and key not in lookup:
                try:
                    lookup[key] = (float(row["venue_lat"]), float(row["venue_lon"]))
                except (TypeError, ValueError):
                    pass
    return lookup


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def string_features(raw_title, cand_home, cand_away):
    listing_raw = raw_title or ""
    listing_norm = normalize_text(raw_title)

    team_raw = f"{cand_home} {cand_away}"
    team_raw_rev = f"{cand_away} {cand_home}"
    team_norm = normalize_text(team_raw)
    team_norm_rev = normalize_text(team_raw_rev)

    lev_raw = max(
        fuzz.ratio(listing_raw, team_raw), fuzz.ratio(listing_raw, team_raw_rev)
    ) / 100.0
    lev_norm = max(
        fuzz.ratio(listing_norm, team_norm), fuzz.ratio(listing_norm, team_norm_rev)
    ) / 100.0

    jw_raw = max(
        jellyfish.jaro_winkler_similarity(listing_raw, team_raw),
        jellyfish.jaro_winkler_similarity(listing_raw, team_raw_rev),
    )
    jw_norm = max(
        jellyfish.jaro_winkler_similarity(listing_norm, team_norm),
        jellyfish.jaro_winkler_similarity(listing_norm, team_norm_rev),
    )

    jac_raw = jaccard(tokenize(listing_raw), tokenize(team_raw))
    jac_norm = jaccard(tokenize(listing_norm), tokenize(team_norm))

    tsr_raw = fuzz.token_sort_ratio(listing_raw, team_raw) / 100.0
    tsr_norm = fuzz.token_sort_ratio(listing_norm, team_norm) / 100.0

    return {
        "str_lev_ratio_raw": lev_raw,
        "str_lev_ratio_norm": lev_norm,
        "str_jaro_winkler_raw": jw_raw,
        "str_jaro_winkler_norm": jw_norm,
        "str_token_jaccard_raw": jac_raw,
        "str_token_jaccard_norm": jac_norm,
        "str_token_sort_ratio_raw": tsr_raw,
        "str_token_sort_ratio_norm": tsr_norm,
    }


def date_features(raw_date, cand_event_date):
    dt, year_missing = normalize_date(raw_date)

    try:
        cand_dt = dateparser.parse(cand_event_date)
    except (ValueError, OverflowError, TypeError):
        cand_dt = None

    if dt is None or cand_dt is None:
        return {
            "date_missing": 1,
            "date_year_inferred": 1 if year_missing else 0,
            "date_day_diff": DATE_MISSING_SENTINEL,
            "date_exact_match": 0,
            "date_same_weekday": 0,
        }

    day_diff = abs((dt.date() - cand_dt.date()).days)
    return {
        "date_missing": 0,
        "date_year_inferred": 1 if year_missing else 0,
        "date_day_diff": day_diff,
        "date_exact_match": 1 if day_diff == 0 else 0,
        "date_same_weekday": 1 if dt.weekday() == cand_dt.weekday() else 0,
    }


def venue_features(raw_venue, cand_venue_canonical, cand_lat, cand_lon, venue_coord_lookup):
    if not raw_venue or not raw_venue.strip():
        return {
            "venue_missing": 1,
            "venue_exact_match": 0,
            "venue_alias_match": 0,
            "venue_fuzzy_score": 0.0,
            "venue_geo_distance_km": -1.0,
            "venue_geo_distance_missing": 1,
        }

    raw_venue_stripped = raw_venue.strip()
    norm_listing_venue_raw_compare = normalize_text(raw_venue)  # no alias resolution
    norm_cand_venue = normalize_text(cand_venue_canonical)

    # alias-resolved + normalized, reusing blocking/normalize.py's normalize_venue
    norm_listing_venue = normalize_venue(raw_venue)

    exact_match = 1 if (
        norm_listing_venue_raw_compare and norm_listing_venue_raw_compare == norm_cand_venue
    ) else 0

    alias_target = VENUE_ALIASES.get(raw_venue_stripped)
    alias_match = 1 if (
        alias_target is not None and normalize_text(alias_target) == norm_cand_venue
    ) else 0

    fuzzy_score = fuzz.ratio(norm_listing_venue, norm_cand_venue) / 100.0

    geo_distance = -1.0
    geo_missing = 1
    coords = venue_coord_lookup.get(norm_listing_venue)
    if coords is not None and cand_lat not in (None, "") and cand_lon not in (None, ""):
        try:
            lat2, lon2 = float(cand_lat), float(cand_lon)
            geo_distance = haversine_km(coords[0], coords[1], lat2, lon2)
            geo_missing = 0
        except (TypeError, ValueError):
            pass

    return {
        "venue_missing": 0,
        "venue_exact_match": exact_match,
        "venue_alias_match": alias_match,
        "venue_fuzzy_score": fuzzy_score,
        "venue_geo_distance_km": round(geo_distance, 3) if geo_distance != -1.0 else -1.0,
        "venue_geo_distance_missing": geo_missing,
    }


def source_features(source):
    return {f"source_{s}": (1 if source == s else 0) for s in SOURCES}


def main():
    venue_coord_lookup = build_venue_coord_lookup()

    in_path = DATA_DIR / "candidate_pairs.csv"
    fieldnames = None
    n = 0
    with in_path.open() as in_f, OUT_PATH.open("w", newline="") as out_f:
        reader = csv.DictReader(in_f)
        writer = None

        for row in reader:
            feats = {}
            feats.update(string_features(row["raw_title"], row["cand_home_team"], row["cand_away_team"]))
            feats.update(date_features(row["raw_date"], row["cand_event_date"]))
            feats.update(
                venue_features(
                    row["raw_venue"],
                    row["cand_venue_canonical"],
                    row["cand_venue_lat"],
                    row["cand_venue_lon"],
                    venue_coord_lookup,
                )
            )
            feats.update(source_features(row["source"]))

            out_row = {
                "listing_id": row["listing_id"],
                "candidate_event_id": row["candidate_event_id"],
                "label": row["label"],
                "split": row["split"],
                **feats,
            }

            if writer is None:
                fieldnames = list(out_row.keys())
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(out_row)
            n += 1
            if n % 20000 == 0:
                print(f"  {n} rows processed", flush=True)

    print(f"Wrote {n} feature rows to {OUT_PATH}")
    print(f"Feature columns ({len(fieldnames) - 4}): {[c for c in fieldnames if c not in ('listing_id', 'candidate_event_id', 'label', 'split')]}")


if __name__ == "__main__":
    main()
