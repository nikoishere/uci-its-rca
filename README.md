# ActivitySim Post-Run RCA Agent

Automatic root-cause analysis for failed [ActivitySim](https://activitysim.github.io/) simulations.

The tool reads the end of a log file backwards in fixed-size chunks, extracts the failure region and stack trace, pulls relevant YAML config context, queries a pgvector knowledge base of runbooks and resolved incidents, then calls GPT-4o to produce a concise, cited root-cause report — all without loading the full log into memory.

---

## Features

| Capability | Detail |
|---|---|
| **Memory-safe log parsing** | Reverse-chunk reader with carry buffer; constant memory regardless of log size |
| **YAML config extraction** | Surfaces `settings.yaml`, `network_los.yaml`, and other ActivitySim configs |
| **RAG knowledge base** | Runbooks and past incidents embedded into pgvector on PostgreSQL, source docs stored in S3 |
| **Structured LLM output** | GPT-4o Structured Outputs — always a validated Pydantic model, no JSON scraping |
| **Production resilience** | Rate limiting (token bucket), timeouts, exponential-backoff retries, graceful fallback |
| **Cost visibility** | Per-call and session-level latency + token cost metrics printed after every run |

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for PostgreSQL + pgvector)
- An OpenAI API key
- AWS credentials with S3 access (for the RAG document store)

---

## Setup

```bash
# 1. Clone and enter the project
cd its_rca

# 2. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY, AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY, and S3_BUCKET.

# 5. Start PostgreSQL + pgvector
docker-compose up -d

# 6. Create the schema (run once)
python main.py init-db
```

---

## Usage

### Analyse a failed simulation

```bash
# Basic (no YAML context, skips RAG if DB unreachable)
python main.py analyze /path/to/simulation.log

# With YAML config directory and Markdown output
python main.py analyze /path/to/simulation.log \
    --config-dir /path/to/configs \
    --output report.md

# Skip RAG lookup (useful when DB is not running)
python main.py analyze /path/to/simulation.log --no-rag

# Retrieve more RAG snippets
python main.py analyze /path/to/simulation.log --top-k 6
```

### Build the knowledge base

```bash
# Ingest a runbook
python main.py ingest docs/oom-runbook.md \
    --type runbook \
    --title "OOM on zone-skims load"

# Ingest a resolved incident report
python main.py ingest incidents/2024-11-keyerror.txt \
    --type incident \
    --title "KeyError trip_mode Nov 2024"
```

---

## Architecture

```
main.py (Click CLI)
│
├── analyze command
│   ├── rca/log_parser.py       ← reverse-chunk binary reader
│   ├── rca/yaml_extractor.py   ← YAML config surfacing
│   ├── rag/retriever.py        ← pgvector semantic search
│   ├── rca/agent.py            ← GPT-4o Structured Outputs
│   └── rca/report.py           ← Rich terminal + Markdown output
│
├── ingest command
│   ├── rag/s3_store.py         ← S3 upload
│   ├── rag/embedder.py         ← text-embedding-3-small
│   └── rag/ingest.py           ← chunk → embed → pgvector
│
└── init-db command
    └── rag/db.py               ← CREATE EXTENSION / TABLE / INDEX
```

### Log parser algorithm

1. Open file in binary mode, `seek` to end.
2. Read backwards in `CHUNK_SIZE` (default 8 KB) chunks.
3. A carry buffer holds the partial line at each chunk boundary so no line is silently truncated across chunks.
4. Each line is scanned for failure markers (`Traceback`, `ERROR`, `FATAL`, `MemoryError`, …).
5. Once the failure line is found, collect up to `CONTEXT_LINES` (default 50) lines of chronologically earlier context, then stop — skipping the rest of the file entirely.
6. The collected lines are reversed back to chronological order and a stack trace is extracted.

### RAG pipeline

```
Ingest:  file → S3 (source doc) → chunk → embed → pgvector
Retrieve: failure context → embed → cosine search → top-k citations
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o` | Any model supporting Structured Outputs |
| `OPENAI_TIMEOUT_S` | `60` | Per-request timeout in seconds |
| `OPENAI_MAX_RETRIES` | `3` | Retries on transient API errors |
| `RPM_LIMIT` | `10` | Outgoing requests per minute (token bucket) |
| `DATABASE_URL` | `postgresql://rca:rca@localhost:5432/rca_db` | pgvector connection string |
| `AWS_REGION` | `us-east-1` | S3 bucket region |
| `S3_BUCKET` | `its-rca-knowledge-base` | Bucket for source documents |
| `CHUNK_SIZE` | `8192` | Log parser chunk size in bytes |
| `CONTEXT_LINES` | `50` | Lines of context around the failure |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIM` | `1536` | Must match the chosen embedding model |
