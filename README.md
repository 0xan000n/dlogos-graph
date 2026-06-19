# dLogos PoC — Temporal Dialogue Knowledge Graph

dLogos is a **generalized, temporal, dialogue knowledge graph** built from ~200
podcasts: speakers, entities, and time-stamped, **stance-tagged claims**,
queryable for *who said what, when, and how the consensus moved*. The proof of
value is a four-arm head-to-head where a strong frontier model **plus dLogos**
answers temporal/consensus questions that the same model **alone** (or with web
search, or with naive vector-RAG over identical transcripts) cannot — winning
specifically on *temporal-consensus synthesis across attributed sources*. This
repository is the PoC pipeline that constructs the graph and runs that eval; it
redistributes nothing (the corpus, transcripts, and claims stay internal).

Full design: [`docs/superpowers/specs/2026-06-18-dlogos-dialogue-graph-poc-design.md`](docs/superpowers/specs/2026-06-18-dlogos-dialogue-graph-poc-design.md).
Build plan: [`docs/superpowers/plans/2026-06-18-dlogos-poc-build-plan.md`](docs/superpowers/plans/2026-06-18-dlogos-poc-build-plan.md).
Build report: [`BUILD_REPORT.md`](BUILD_REPORT.md).

---

## Architecture

```
 RSS feeds (Podcast Index GUIDs)
      │  fetch enclosure · content-hash dedupe · idempotent on episode GUID
      ▼
 Work queue ──► Object storage (raw audio)
      │
      ▼
 ASR + diarization + alignment            [asr/]   (WhisperX + pyannote | MockASRBackend)
      │  speaker-labeled, word-timestamped transcript; drop low-talk-time speakers
      ▼
 Cross-episode speaker identity           [speakers/]
      │  host-anchored voiceprint gallery + recurring-guest resolution
      │  (episode metadata + "my guest today is…" intro + Wikidata QID)
      ▼
 Chunk (overlap, speaker-labelled)        [extraction/chunking.py]
      ▼
 Open-weight extraction                    [extraction/]  (OpenAI-compatible endpoint)
      │  stance-tagged Claims + source spans
      │  CONTROLLED predicate vocabulary enforced AT EXTRACTION TIME (closed enum)
      ▼
 Lightweight resolution (batch, pre-load) [resolution/]
      │  subject-entity embedding clustering → canonical_id (Apple≈iPhone≈Apple-hw)
      │  recurring-guest → Wikidata QID
      ▼
 Bitemporal graph load                     [graph/]   (Graphiti/Neo4j | FakeGraphStore)
      │  reified Claim nodes · validity intervals · invalidate-not-delete
      │  BULK path BYPASSES Graphiti per-add LLM node-dedup (we resolved already)
      ▼
 Hybrid + temporal retrieval               [retrieval/]
      │  semantic + BM25 + graph traversal, RRF-fused, event-time filters
      │  + consensus-over-time primitive (bucketed by canonical_id)
      ▼
 MCP server  +  thin four-arm UI           [mcp/]  [ui/]
      │
      ▼
 Four-arm head-to-head eval                [eval/]   (the greenlight artifact)
   model-alone · +web-search · +naive-vector-RAG · +dLogos
   blind-scored · 2nd rater on a subset · pre-registered answer shapes
```

`pipeline.py` composes ingest → ASR → diarize/prune → speaker identity → chunk →
extract → resolve → bulk-load into one injectable orchestrator. `spike/` is the
de-risking harness that compares Graphiti-extracts (Approach A) vs
we-extract-then-load (Approach B) on quality **and** the throughput / $-per-episode
gate (spec §7.6).

---

## What runs offline now vs. what needs live infra

The entire pipeline runs **end-to-end offline on the core dependency group** via
injected fakes (mock ASR, fake in-memory graph store, fake embedder, fake async
LLM clients). All 382 unit tests run with `uv sync` + `uv run pytest` — **no GPU,
no Neo4j, no network, no API keys**. Heavy/optional deps are imported *lazily
inside functions*, so importing any module for tests never pulls them.

| Capability | Offline now (core deps) | Needs live infra to run for real |
|---|---|---|
| Shared schema, config, pipeline wiring | ✅ pure Pydantic | — |
| Ingestion: charts, manifest, queue, GUID-idempotent fetch dedupe | ✅ logic unit-tested (httpx mocked) | **Podcast Index API key**; live RSS/enclosure HTTP for real feeds |
| ASR + diarization + alignment | ✅ `MockASRBackend` (deterministic transcript) | **GPU** + `asr` extra (whisperx, pyannote.audio, torch); **HF token** for pyannote weights |
| Cross-episode speaker identity (hosts + recurring guests) | ✅ injected embedder + gallery; intro-parse + metadata | real voiceprint embedder; **Wikidata SPARQL** for guest QIDs |
| Chunking + controlled-predicate extraction | ✅ fake async chat client, schema-validated | **open-weight LLM endpoint** (OpenAI-compatible, e.g. vLLM/SGLang on rented GPU) |
| Subject-entity clustering → canonical_id | ✅ `FakeEmbedder` (deterministic) | **embedding endpoint / `embed` extra** (BGE-M3) for real surface forms |
| Recurring-guest Wikidata linking | ✅ httpx mocked | **Wikidata SPARQL endpoint** (live) |
| Bitemporal graph load + temporal helpers | ✅ `FakeGraphStore` (in-memory) | **Neo4j** (`docker compose up neo4j`) + `graph` extra (graphiti-core, neo4j) |
| Hybrid + temporal retrieval + consensus | ✅ in-memory retrieval store | the same Neo4j + embedding endpoint for corpus-scale retrieval |
| MCP server | ✅ handlers unit-tested directly | **`mcp` extra** to actually serve the tools (`build_server`) |
| Thin four-arm UI | ✅ pure render fns unit-tested | **`ui` extra** (gradio) to launch (`build_ui` / `python -m dlogos.ui.app`) |
| Four-arm eval (arms, rubric, blind, agreement, runner) | ✅ fake arms + fake rater, fully deterministic | **frontier model API key** (+ web-search-enabled endpoint for arm 2) to run real arms |
| Spike comparison + throughput/$ gate | ✅ fakes; gate logic unit-tested | open-weight endpoint + Neo4j to get real throughput/$ numbers |

Short version: **every module is exercised offline today.** "Live infra" means
the four external dependencies the spec names — **GPU ASR**, **Neo4j**, an
**open-weight LLM endpoint** (extraction + embeddings), and a **frontier eval
model** — plus the Podcast Index / Wikidata HTTP services.

---

## Quick start (offline — no infra required)

```bash
make sync          # uv sync — CORE deps only (heavy extras are NOT installed)
make test          # uv run pytest — 382 tests, fully offline & deterministic
```

or directly:

```bash
uv sync
uv run pytest
```

### Run the pipeline on a fixture (offline)

The end-to-end smoke test *is* a runnable example of the whole pipeline on a
fixture episode (ingest → … → bulk-load → dLogos arm → speaker-verified citation
check), wired entirely from offline fakes:

```bash
uv run pytest tests/test_e2e_smoke.py -q
```

`tests/test_e2e_smoke.py` shows the canonical wiring: `MockASRBackend` +
`FakeGraphStore` + a fake embedder + a fake async extraction client →
`Pipeline.run([...])` → `PipelineResult.build_retrieval_surface(...)` →
`ModelDLogosArm`. Copy that wiring to drive the pipeline on your own fixtures.

---

## Running it for real (live infra)

### 1. Configure secrets

```bash
cp .env.example .env       # then fill in the keys/URLs you have
```

`.env` / environment variables read by `dlogos.config.Settings`:

| Variable | Used by | Notes |
|---|---|---|
| `PODCAST_INDEX_KEY` / `PODCAST_INDEX_SECRET` | ingestion | feed resolution + episode GUIDs |
| `EXTRACTION_BASE_URL` / `EXTRACTION_API_KEY` / `EXTRACTION_MODEL` | extraction | OpenAI-compatible open-weight endpoint (e.g. DeepSeek-V3 via vLLM) |
| `EMBED_BASE_URL` / `EMBED_API_KEY` / `EMBED_MODEL` | resolution / retrieval | OpenAI-compatible embeddings (e.g. BGE-M3) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` (`NEO4J_AUTH` for compose) | graph | bitemporal store backend |
| `FRONTIER_BASE_URL` / `FRONTIER_API_KEY` / `FRONTIER_MODEL` | eval | the strong head-to-head baseline (all four arms) |
| `FRONTIER_WEB_SEARCH_*` | eval (arm 2) | optional separate endpoint for the web-search arm |
| `WIKIDATA_ENDPOINT` | resolution / speakers | SPARQL endpoint for guest QIDs (defaults to the public one) |

### 2. Start Neo4j

```bash
docker compose up -d neo4j      # or: make neo4j-up
# Bolt: bolt://localhost:7687   Browser: http://localhost:7474
```

### 3. Install the extras you need

The core sync deliberately omits heavy deps. Install only what you're running:

```bash
uv sync --extra graph      # graphiti-core, neo4j   (graph load + retrieval)
uv sync --extra asr        # whisperx, pyannote.audio, torch, faster-whisper (GPU)
uv sync --extra embed      # FlagEmbedding, sentence-transformers (local BGE-M3)
uv sync --extra mcp        # mcp (serve the MCP tools)
uv sync --extra ui         # gradio (the four-arm UI)
```

> Extras are large and may be unreachable in some environments — that is by
> design. Tests never need them.

### 4. Launch the four-arm UI

```bash
make ui
# = uv run --extra ui python -m dlogos.ui.app
```

`dlogos.ui.app.build_ui(arms)` builds a Gradio Blocks app: one query box over
four side-by-side panels (model-alone / +web-search / +naive-vector-RAG /
+dLogos). Only the dLogos panel surfaces speaker-verified citations (episode +
timestamp + attributed speaker). Wire the four real arms (`dlogos.eval.arms`)
with your frontier client + a retriever over the loaded graph.

### 5. Start the MCP server

```python
from dlogos.mcp.server import GraphRetrievalSurface, build_server
# `result` is a PipelineResult from Pipeline.run(...), `embedder` your real embedder
surface = result.build_retrieval_surface(embedder)
server = build_server(surface, name="dlogos")   # needs the `mcp` extra
server.run()                                     # stdio transport
```

Five tools are exposed: `search_dialogue`, `who_discussed`, `consensus_trend`,
`belief_history`, `provenance_lookup`. Each handler is a plain function (unit-
tested without `mcp`); `build_server` lazily imports `mcp` and registers thin
wrappers, so the corpus is queryable from Claude.

---

## Repository layout

```
src/dlogos/
  schema.py        shared domain model (Claim, Entity, Speaker, bitemporal base) — import from here
  config.py        pydantic-settings Settings + singleton
  pipeline.py      end-to-end orchestrator (injectable deps)
  ingestion/       podcast_index · charts · manifest · fetch · queue
  asr/             base · whisperx_backend · diarization · mock_backend
  speakers/        identity (host gallery) · guests (recurring-guest resolution)
  extraction/      chunking · predicates · extractor
  resolution/      subjects (canonical_id clustering) · wikidata
  graph/           store · loader · temporal · fake_store
  retrieval/       hybrid · consensus
  eval/            golden · arms · rubric · blind · agreement · runner
  mcp/             server (5 tools + handlers + GraphRetrievalSurface)
  ui/              app (four-arm side-by-side)
  spike/           run_comparison · score (the §7.6 de-risk gate)
tests/             mirrors src/ — 382 deterministic, offline tests
```

## Development conventions

- **Heavy/optional deps are imported lazily inside functions**, never at module
  top level. Importing any module for tests requires only the core group.
- **Tests are deterministic**: seeds, embedders, and clients are injected; no
  real randomness, no real network.
- Shared types live only in `src/dlogos/schema.py`; modules import from there.

```bash
make lint          # ruff check src tests (optional; ruff not in core deps)
```
