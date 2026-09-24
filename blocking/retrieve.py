"""
PRD 6.2: for each incoming listing, retrieve top-K candidates from
Elasticsearch instead of comparing against the full catalog. Fuzzy match
on team/venue text, date range filter (+/-3 days) when a reliable date
is available.
"""
import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent))
from normalize import date_range_filter, normalize_date, normalize_text, normalize_venue  # noqa: E402

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")
INDEX_NAME = "events"
DATE_WINDOW_DAYS = 3


def get_client():
    return Elasticsearch(ES_HOST)


def build_query(raw_title, raw_date, raw_venue, size=20):
    search_text = normalize_text(raw_title)
    venue_norm = normalize_venue(raw_venue) if raw_venue else ""

    should = [{"match": {"search_text": {"query": search_text, "fuzziness": "AUTO"}}}]
    if venue_norm:
        should.append(
            {"match": {"venue_canonical_norm": {"query": venue_norm, "fuzziness": "AUTO", "boost": 0.5}}}
        )

    query = {"bool": {"should": should, "minimum_should_match": 1}}

    dt, year_missing = normalize_date(raw_date) if raw_date else (None, True)
    if dt is not None and not year_missing:
        lo, hi = date_range_filter(dt, DATE_WINDOW_DAYS)
        query["bool"]["filter"] = [
            {"range": {"event_date": {"gte": lo.strftime("%Y-%m-%d"), "lte": hi.strftime("%Y-%m-%d")}}}
        ]
    # If the year is missing/ambiguous, we deliberately do NOT filter by date --
    # guessing wrong would drop the true candidate; text/venue similarity alone
    # still does the retrieval work (see blocking/normalize.py docstring).

    return {"size": size, "query": query}


def retrieve_candidates(es, raw_title, raw_date, raw_venue, size=20):
    body = build_query(raw_title, raw_date, raw_venue, size=size)
    resp = es.search(index=INDEX_NAME, **body)
    return [
        {"event_id": hit["_source"]["event_id"], "score": hit["_score"]}
        for hit in resp["hits"]["hits"]
    ]
