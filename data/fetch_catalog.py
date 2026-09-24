"""
Pull a real canonical event catalog from ESPN's public scoreboard API
(no auth required) for the completed 2025-26 NBA season and the completed
2025 NFL regular season, then attach venue coordinates from a hand-curated
static lookup (see venue_coords.py). Output: data/raw/events_catalog.csv.

This is real schedule data (team names, venues, dates) used per PRD 5.1;
noise is applied downstream in noise_generator.py, never here.

Note: an earlier version of this script live-geocoded venues via OpenStreetMap
Nominatim, but its 1 req/sec public rate limit made a one-time ~80-venue batch
lookup unreliably slow, so venue coordinates are now a static table instead.
"""
import csv
import json
import time
from pathlib import Path

import requests

from venue_coords import VENUE_COORDS

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

# ESPN's endpoint 403s on some custom User-Agent strings; the requests default works fine.
HEADERS = {}
NBA_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
NFL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def _parse_events(payload, sport):
    out = []
    for ev in payload.get("events", []):
        comp = ev["competitions"][0]
        venue = comp.get("venue", {})
        competitors = comp["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")
        out.append(
            {
                "event_id": f"{sport}_{ev['id']}",
                "sport": sport,
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "venue_canonical": venue.get("fullName", ""),
                "venue_city": venue.get("address", {}).get("city", ""),
                "venue_state": venue.get("address", {}).get("state", ""),
                "event_date": ev["date"],  # ISO 8601 UTC
            }
        )
    return out


def fetch_nba_season(anchor_date="20251021"):
    r = requests.get(NBA_SCOREBOARD, params={"dates": anchor_date}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    calendar = r.json()["leagues"][0]["calendar"]
    dates = [d[:10].replace("-", "") for d in calendar]
    print(f"NBA: {len(dates)} game dates in season")

    events = []
    for i, d in enumerate(dates):
        resp = requests.get(NBA_SCOREBOARD, params={"dates": d}, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        events.extend(_parse_events(resp.json(), "nba"))
        if i % 20 == 0:
            print(f"  NBA date {i+1}/{len(dates)} -> {len(events)} events so far", flush=True)
        time.sleep(0.05)
    return events


def fetch_nfl_season(year=2025, weeks=range(1, 19)):
    events = []
    for wk in weeks:
        resp = requests.get(
            NFL_SCOREBOARD,
            params={"dates": year, "seasontype": 2, "week": wk},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        wk_events = _parse_events(resp.json(), "nfl")
        events.extend(wk_events)
        print(f"  NFL week {wk} -> {len(wk_events)} games", flush=True)
        time.sleep(0.05)
    return events


def main():
    raw_cache = RAW_DIR / "events_raw.json"
    if raw_cache.exists():
        print(f"Using cached raw events from {raw_cache}")
        all_events = json.loads(raw_cache.read_text())
    else:
        print("Fetching NBA 2025-26 season...")
        nba_events = fetch_nba_season()
        print(f"NBA total: {len(nba_events)} events")

        print("Fetching NFL 2025 regular season...")
        nfl_events = fetch_nfl_season()
        print(f"NFL total: {len(nfl_events)} events")

        all_events = nba_events + nfl_events
        raw_cache.write_text(json.dumps(all_events, indent=2))

    missing_coords = set()
    for ev in all_events:
        coords = VENUE_COORDS.get(ev["venue_canonical"])
        if coords:
            ev["venue_lat"], ev["venue_lon"] = coords
        else:
            ev["venue_lat"], ev["venue_lon"] = "", ""
            missing_coords.add(ev["venue_canonical"])

    if missing_coords:
        print(f"WARNING: no coordinates for venues: {sorted(missing_coords)}")

    fieldnames = [
        "event_id", "sport", "home_team", "away_team", "venue_canonical",
        "venue_city", "venue_state", "event_date", "venue_lat", "venue_lon",
    ]
    out_path = RAW_DIR / "events_catalog.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_events)

    print(f"Wrote {len(all_events)} events -> {out_path}")


if __name__ == "__main__":
    main()
