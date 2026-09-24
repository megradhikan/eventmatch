"""
PRD 5.2: takes each canonical event and produces N noisy listing variants
simulating messy multi-feed scraped data. Ground truth (listing_id ->
true_event_id or NO_MATCH) is known because we generate the noise, and is
stored separately from anything fed to the model.

Noise categories implemented:
  - name noise: reordering, abbreviation, char-level typos, casing, punctuation drift
  - date noise: format drift, missing year, rare wrong-day errors
  - venue noise: alias substitution, abbreviation, occasional omission
  - structural noise: missing fields, extra whitespace/HTML entities, near-dup listings
  - hard negatives: held-out events (real, but not in the indexed catalog slice)
                     and fabricated events with no real catalog counterpart at all
"""
import csv
import html
import random
import re
from dataclasses import dataclass
from pathlib import Path

from venue_aliases import VENUE_ALIASES

RAW_DIR = Path(__file__).parent / "raw"
PROC_DIR = Path(__file__).parent / "processed"
PROC_DIR.mkdir(exist_ok=True)

RNG_SEED = 42

TEAM_ABBREVIATIONS = {
    "Los Angeles Lakers": "LAL", "Boston Celtics": "BOS", "Golden State Warriors": "GSW",
    "Los Angeles Clippers": "LAC", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Denver Nuggets": "DEN", "Phoenix Suns": "PHX", "Dallas Mavericks": "DAL",
    "Philadelphia 76ers": "PHI", "New York Knicks": "NYK", "Brooklyn Nets": "BKN",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Atlanta Hawks": "ATL",
    "Toronto Raptors": "TOR", "Memphis Grizzlies": "MEM", "New Orleans Pelicans": "NOP",
    "Sacramento Kings": "SAC", "Minnesota Timberwolves": "MIN", "Oklahoma City Thunder": "OKC",
    "Portland Trail Blazers": "POR", "Utah Jazz": "UTA", "San Antonio Spurs": "SAS",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Detroit Pistons": "DET",
    "Charlotte Hornets": "CHA", "Washington Wizards": "WAS", "Orlando Magic": "ORL",
    "Kansas City Chiefs": "KC", "Buffalo Bills": "BUF", "San Francisco 49ers": "SF",
    "Dallas Cowboys": "DAL", "Philadelphia Eagles": "PHI", "Green Bay Packers": "GB",
    "New England Patriots": "NE", "Seattle Seahawks": "SEA", "Baltimore Ravens": "BAL",
    "Detroit Lions": "DET", "Miami Dolphins": "MIA", "New York Jets": "NYJ",
    "New York Giants": "NYG", "Pittsburgh Steelers": "PIT", "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE", "Tennessee Titans": "TEN", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Houston Texans": "HOU", "Denver Broncos": "DEN",
    "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Los Angeles Rams": "LAR",
    "Arizona Cardinals": "ARI", "Minnesota Vikings": "MIN", "Chicago Bears": "CHI",
    "Atlanta Falcons": "ATL", "Carolina Panthers": "CAR", "New Orleans Saints": "NO",
    "Tampa Bay Buccaneers": "TB", "Washington Commanders": "WAS",
}

HTML_ENTITY_MAP = {" ": "&nbsp;", "'": "&#39;", "&": "&amp;"}
KEYBOARD_NEIGHBORS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "ou", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def char_typo(s, rng, n=1):
    s = list(s)
    for _ in range(n):
        if len(s) < 3:
            break
        idx = rng.randrange(len(s))
        c = s[idx].lower()
        op = rng.choice(["sub", "del", "swap"])
        if op == "sub" and c in KEYBOARD_NEIGHBORS:
            s[idx] = rng.choice(KEYBOARD_NEIGHBORS[c])
        elif op == "del":
            del s[idx]
        elif op == "swap" and idx < len(s) - 1:
            s[idx], s[idx + 1] = s[idx + 1], s[idx]
    return "".join(s)


def noisy_name(home, away, rng, noise_level):
    """Returns a raw_title string with name-level noise applied."""
    order = rng.choice(["home_vs_away", "away_at_home", "away_vs_home"])
    h, a = home, away
    if noise_level >= 1 and rng.random() < 0.5:
        h = TEAM_ABBREVIATIONS.get(h, h)
        a = TEAM_ABBREVIATIONS.get(a, a)
    if order == "home_vs_away":
        title = f"{h} vs {a}"
    elif order == "away_at_home":
        title = f"{a} at {h}"
    else:
        title = f"{a} vs {h}"

    if noise_level >= 2 and rng.random() < 0.4:
        title = char_typo(title, rng, n=rng.randint(1, 2))
    if noise_level >= 1 and rng.random() < 0.3:
        title = title.upper() if rng.random() < 0.5 else title.lower()
    if noise_level >= 2 and rng.random() < 0.3:
        title = title.replace(" vs ", " v. ").replace(" at ", " @ ")
    return title


def noisy_date(iso_date, rng, noise_level):
    """Returns (raw_date_string, wrong_day_flag). iso_date is 'YYYY-MM-DDTHH:MMZ'."""
    from datetime import datetime, timedelta

    dt = datetime.strptime(iso_date[:10], "%Y-%m-%d")
    wrong_day = False
    if noise_level >= 2 and rng.random() < 0.08:
        dt = dt + timedelta(days=rng.choice([-2, -1, 1, 2]))
        wrong_day = True

    fmt = rng.choice(["iso", "us_slash", "us_slash_2digit", "month_name", "month_name_noyear"])
    if fmt == "iso":
        s = dt.strftime("%Y-%m-%d")
    elif fmt == "us_slash":
        s = dt.strftime("%-m/%-d/%Y") if hasattr(dt, "strftime") else dt.strftime("%m/%d/%Y")
    elif fmt == "us_slash_2digit":
        s = dt.strftime("%-m/%-d/%y")
    elif fmt == "month_name":
        s = f"{MONTHS[dt.month - 1]} {dt.day}, {dt.year}"
    else:  # month_name_noyear -- missing year, must be inferred from context
        s = f"{MONTHS[dt.month - 1]} {dt.day}"
    return s, wrong_day


def noisy_venue(venue, rng, noise_level):
    v = venue
    if noise_level >= 1 and rng.random() < 0.35:
        aliases = [a for a, canon in VENUE_ALIASES.items() if canon == venue]
        if aliases:
            v = rng.choice(aliases)
    if noise_level >= 2 and rng.random() < 0.15:
        v = "".join(w[0] for w in v.split() if w[0].isupper()) or v
    if noise_level >= 1 and rng.random() < 0.1:
        return ""  # venue omitted entirely
    return v


def apply_structural_noise(title, date_str, venue, rng, noise_level):
    if noise_level >= 2 and rng.random() < 0.2:
        title = "  " + title.replace(" ", "  ", 1) + " "
    if noise_level >= 2 and rng.random() < 0.15:
        for ch, ent in HTML_ENTITY_MAP.items():
            if ch in title and rng.random() < 0.5:
                title = title.replace(ch, ent, 1)
                break
    return title, date_str, venue


@dataclass
class Source:
    name: str
    noise_level: int  # 0=clean, 1=moderate, 2=noisy
    missing_field_rate: float


SOURCES = [
    Source("official_feed", 0, 0.01),
    Source("partner_feed_a", 1, 0.05),
    Source("scraped_feed_b", 2, 0.15),
    Source("scraped_feed_c", 2, 0.20),
]


def generate_listing(event, source, rng, listing_id):
    title = noisy_name(event["home_team"], event["away_team"], rng, source.noise_level)
    date_str, wrong_day = noisy_date(event["event_date"], rng, source.noise_level)
    venue = noisy_venue(event["venue_canonical"], rng, source.noise_level)
    title, date_str, venue = apply_structural_noise(title, date_str, venue, rng, source.noise_level)

    if rng.random() < source.missing_field_rate:
        field = rng.choice(["venue", "date"])
        if field == "venue":
            venue = ""
        else:
            date_str = ""

    return {
        "listing_id": listing_id,
        "raw_title": title,
        "raw_date": date_str,
        "raw_venue": venue,
        "source": source.name,
        "wrong_day_noise": wrong_day,
    }


def fabricate_fake_event(rng, idx, real_events):
    """A genuinely new event with no real catalog counterpart: real team names
    paired up in a combination that never actually happened, on a fabricated
    date. Used to test NO_MATCH generalization beyond 'real event we didn't index'."""
    sample = rng.sample(real_events, 2)
    home = sample[0]["home_team"]
    away = sample[1]["away_team"] if sample[1]["away_team"] != home else sample[1]["home_team"]
    year = rng.choice([2027, 2028])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return {
        "event_id": f"fake_{idx}",
        "sport": sample[0]["sport"],
        "home_team": home,
        "away_team": away,
        "venue_canonical": sample[0]["venue_canonical"],
        "venue_city": sample[0]["venue_city"],
        "venue_state": sample[0]["venue_state"],
        "event_date": f"{year}-{month:02d}-{day:02d}T19:00Z",
        "venue_lat": sample[0]["venue_lat"],
        "venue_lon": sample[0]["venue_lon"],
    }


def main():
    rng = random.Random(RNG_SEED)

    with (RAW_DIR / "events_catalog.csv").open() as f:
        events = list(csv.DictReader(f))
    print(f"Loaded {len(events)} real events")

    # Deterministic 80/20 split into indexed (catalog) vs held-out (hard negatives).
    shuffled = events[:]
    rng.shuffle(shuffled)
    split_point = int(len(shuffled) * 0.8)
    indexed_events = shuffled[:split_point]
    held_out_events = shuffled[split_point:]
    print(f"Indexed (in catalog): {len(indexed_events)} | Held-out (NO_MATCH source): {len(held_out_events)}")

    # Event-level train/val/test split (by event, not listing -- PRD 6.4).
    # Applies only to indexed events, since those are the ones with a real
    # positive label a ranker can learn to hit.
    idx_shuffled = indexed_events[:]
    rng.shuffle(idx_shuffled)
    n = len(idx_shuffled)
    train_cut, val_cut = int(n * 0.7), int(n * 0.85)
    split_map = {}
    for e in idx_shuffled[:train_cut]:
        split_map[e["event_id"]] = "train"
    for e in idx_shuffled[train_cut:val_cut]:
        split_map[e["event_id"]] = "val"
    for e in idx_shuffled[val_cut:]:
        split_map[e["event_id"]] = "test"

    fake_events = [fabricate_fake_event(rng, i, events) for i in range(200)]

    listings = []
    ground_truth = []
    listing_counter = 0

    def add_listings(event, true_event_id, k):
        nonlocal listing_counter
        for _ in range(k):
            source = rng.choice(SOURCES)
            listing_id = f"L{listing_counter:06d}"
            listing_counter += 1
            listing = generate_listing(event, source, rng, listing_id)
            listings.append(listing)
            ground_truth.append({"listing_id": listing_id, "true_event_id": true_event_id or ""})

    for e in indexed_events:
        add_listings(e, e["event_id"], k=4)
    for e in held_out_events:
        add_listings(e, None, k=2)
    for e in fake_events:
        add_listings(e, None, k=2)

    print(f"Generated {len(listings)} listings "
          f"({sum(1 for g in ground_truth if g['true_event_id'])} positive, "
          f"{sum(1 for g in ground_truth if not g['true_event_id'])} NO_MATCH)")

    # events_catalog.csv: ONLY the indexed slice -- this is what Elasticsearch
    # indexes in Phase 2. Held-out/fake events are deliberately excluded so
    # listings.csv gives us real hard negatives.
    catalog_fields = ["event_id", "sport", "home_team", "away_team", "venue_canonical",
                       "venue_city", "venue_state", "event_date", "venue_lat", "venue_lon", "split"]
    with (PROC_DIR / "events_catalog.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=catalog_fields)
        w.writeheader()
        for e in indexed_events:
            row = {k: e.get(k, "") for k in catalog_fields}
            row["split"] = split_map[e["event_id"]]
            w.writerow(row)

    with (PROC_DIR / "held_out_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=catalog_fields[:-1])
        w.writeheader()
        for e in held_out_events + fake_events:
            w.writerow({k: e.get(k, "") for k in catalog_fields[:-1]})

    listing_fields = ["listing_id", "raw_title", "raw_date", "raw_venue", "source", "wrong_day_noise"]
    with (PROC_DIR / "listings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=listing_fields)
        w.writeheader()
        w.writerows(listings)

    with (PROC_DIR / "ground_truth.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["listing_id", "true_event_id"])
        w.writeheader()
        w.writerows(ground_truth)

    print(f"Wrote events_catalog.csv ({len(indexed_events)} events), "
          f"held_out_events.csv ({len(held_out_events) + len(fake_events)} events), "
          f"listings.csv ({len(listings)} rows), ground_truth.csv")


if __name__ == "__main__":
    main()
