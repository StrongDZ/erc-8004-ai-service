# ERC-8004 AI Service

Standalone Python service that owns all AI logic for the ERC-8004 benchmarking platform. The Go backend (`erc-8004-benchmarking-be`) calls this service over HTTP for any feedback record that falls through the rule-based classifier.

## Two concerns in one package

1. **HTTP service** under [`app/`](app/) — FastAPI exposing `/classify`, `/embed`, `/health`. This is what Go talks to in production.
2. **Research notebooks** under [`notebooks/`](notebooks/) — three benchmark approaches compared on the same gold set:

| # | Approach | Notebook | Inference cost | Train cost |
|---|---|---|---|---|
| A | Improved zero-shot LLM (XML prompt, smaller context) | `03_approach_a_zeroshot.ipynb` | high (LLM call/record) | none |
| B | Embedding + classical ML (bge + LogReg/k-NN/SVM) | `04_approach_b_embedding.ipynb` | low (~30ms) | small (~min on M4) |
| C | Few-shot LLM with embedding retrieval | `05_approach_c_fewshot.ipynb` | medium | small |

Both halves reuse [`shared/`](shared/) — `ollama_client.py`, `prompts.py`, `context_builder.py`, etc.

## Category schema

5 categories (down from 6 — `spam` and `noise` merged into `junk` because both are throw-away signals and `noise` has only 33 records in the corpus):

```
junk | service_feedback | config_feedback | app_specific | others
```

The LLM emits one of the first four (it never returns `others`); `others` is the rule-classifier's fallback bucket for rows that didn't match any rule.

## Folder layout

```
erc-8004-ai-service/
├── app/                       # FastAPI server (production path)
│   ├── main.py                #   entrypoint + lifespan
│   ├── schemas.py             #   Pydantic request/response models
│   ├── deps.py                #   singleton OllamaClient + SentenceTransformer
│   └── routers/
│       ├── classify.py        #   POST /classify
│       ├── embed.py           #   POST /embed
│       └── health.py          #   GET /health
├── shared/                    # Python modules reused by app/ AND notebooks/
│   ├── mongo_client.py
│   ├── ollama_client.py
│   ├── prompts.py             # V4 XML prompt + legacy ports
│   ├── context_builder.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── types.py
├── notebooks/                 # 7 research notebooks (00–06)
└── data/                      # gitignored: splits, embeddings, results
```

## Setup

Recommended — install `uv` once, then sync:

```bash
pip install uv                 # one-time
cd erc-8004-ai-service
uv venv --python 3.12          # ML wheels are most stable on 3.11/3.12
uv sync
source .venv/bin/activate
```

Fallback (pip):

```bash
cd erc-8004-ai-service
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.14 is too new for several ML wheels (torch, sentence-transformers). Use 3.11 or 3.12 — install via `brew install python@3.12`.

## Env

`.env` lives in this folder (no longer a symlink to the Go backend). Copy the keys from `erc-8004-benchmarking-be/.env`:

- `MONGO_URI`, `MONGO_DATABASE_ANALYZED_AGENTS`, `MONGO_COLLECTION_FEEDBACK_HISTORY`, `MONGO_COLLECTION_AGENTS`
- `LLM_BASE_URL` (defaults to `http://localhost:11434`) — native Ollama on host
- `AI_SERVICE_DEFAULT_MODEL` (defaults to `qwen2.5:3b`)
- `AI_SERVICE_DEFAULT_EMBED_MODEL` (defaults to `BAAI/bge-base-en-v1.5`)

## Run the HTTP service

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Classify a sample:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"tag1":"excellent","tag2":"fast","value_norm":0.95,
       "agent_description":"trading bot"}'
```

The Go backend reads `AI_SERVICE_URL` (default `http://localhost:8000`) — see `erc-8004-benchmarking-be/.env`.

## Run the research notebooks

1. `00_setup.ipynb` — sanity check Mongo + Ollama connection
2. `01_data_extraction.ipynb` — produces `data/splits/{train,val,test}.parquet`
3. `02_agent_summary.ipynb` — writes `agentSummary` field back to Mongo `agents` collection (one-off, ~min for ~1k unique agents)
4. `03_approach_a_zeroshot.ipynb`, `04_…`, `05_…` — each writes results to `data/results/<approach>.parquet`
5. `06_evaluation.ipynb` — loads all results, prints comparison table, saves plots
