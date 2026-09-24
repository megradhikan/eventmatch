"""
PRD 6.2: index the (indexed-slice) canonical catalog in Elasticsearch with
fuzzy matching on team/artist names and venue, ready for candidate blocking.
"""
import csv
import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_text, normalize_venue  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from team_abbreviations import TEAM_ABBREVIATIONS  # noqa: E402

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")
INDEX_NAME = "events"
CATALOG_PATH = Path(__file__).parent.parent / "data" / "processed" / "events_catalog.csv"

MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "default": {"type": "standard"},
            }
        }
    },
    "mappings": {
        "properties": {
            "event_id": {"type": "keyword"},
            "sport": {"type": "keyword"},
            "home_team": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "away_team": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "search_text": {"type": "text"},
            "venue_canonical": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "event_date": {"type": "date"},
            "venue_lat": {"type": "float"},
            "venue_lon": {"type": "float"},
            "split": {"type": "keyword"},
        }
    },
}


def build_actions(rows):
    for row in rows:
        home_abbr = TEAM_ABBREVIATIONS.get(row["home_team"], "")
        away_abbr = TEAM_ABBREVIATIONS.get(row["away_team"], "")
        # Abbreviations are indexed as extra tokens (not a substitute for the
        # full name) so abbreviation-only listings like "DAL @ ATL" survive
        # fuzzy matching -- edit distance alone can't bridge "dal" to
        # "dallas mavericks", this was the largest source of blocking misses.
        search_text = normalize_text(
            f"{row['home_team']} {row['away_team']} {home_abbr} {away_abbr}"
        )
        yield {
            "_index": INDEX_NAME,
            "_id": row["event_id"],
            "_source": {
                "event_id": row["event_id"],
                "sport": row["sport"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "search_text": search_text,
                "venue_canonical": row["venue_canonical"],
                "venue_canonical_norm": normalize_venue(row["venue_canonical"]),
                "event_date": row["event_date"][:10],
                "venue_lat": float(row["venue_lat"]) if row["venue_lat"] else None,
                "venue_lon": float(row["venue_lon"]) if row["venue_lon"] else None,
                "split": row["split"],
            },
        }


def main():
    es = Elasticsearch(ES_HOST)
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    # venue_canonical_norm added dynamically -- extend mapping
    mapping = MAPPING.copy()
    mapping["mappings"]["properties"]["venue_canonical_norm"] = {"type": "text"}
    es.indices.create(index=INDEX_NAME, **mapping)

    with CATALOG_PATH.open() as f:
        rows = list(csv.DictReader(f))

    success, errors = helpers.bulk(es, build_actions(rows), stats_only=False, raise_on_error=False)
    print(f"Indexed {success} events into '{INDEX_NAME}'")
    if errors:
        print(f"Errors: {len(errors)} (showing first 3)")
        for e in errors[:3]:
            print(e)

    es.indices.refresh(index=INDEX_NAME)
    count = es.count(index=INDEX_NAME)["count"]
    print(f"Index doc count: {count}")


if __name__ == "__main__":
    main()
