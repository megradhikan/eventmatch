# EventMatch

Entity resolution pipeline for live event listings — resolves noisy, inconsistently
formatted event listings (multiple simulated feeds) to canonical event records.
Mirrors the "event matching" problem in marketplace/ticketing data platforms
(venue mapping, inventory reconciliation).

Pipeline: candidate blocking (Elasticsearch) → feature extraction (string similarity,
date/venue, sentence-transformer embeddings) → LightGBM ranker → calibrated confidence
tiers (auto-accept / agentic review / no-match) → LangChain agentic fallback for the
ambiguous band.

**Status: build in progress.** This README is being filled in phase by phase; see
[docs/PRD.md](docs/PRD.md) for the full spec and `docs/` for phase write-ups as they land.

## Repo layout

```
data/       canonical catalog, noise generator, synthetic labeled dataset
blocking/   Elasticsearch indexing + candidate retrieval, recall@K eval
features/   string/date/venue features, embedding pipeline
model/      LightGBM ranker, calibration, thresholds
agent/      LangChain agentic fallback for ambiguous-confidence listings
api/        FastAPI POST /match demo
eval/       metrics, ablations, failure analysis
docs/       PRD, write-ups, charts
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d elasticsearch
```
