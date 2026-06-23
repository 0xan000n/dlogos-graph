# dLogos — Dialogue Knowledge Graph

A **resolved, bitemporal, multilingual knowledge graph built from spoken conversation** (podcasts). It turns dialogue into structured, queryable intelligence: *who said what, about whom, when — and how the consensus moved.*

The bet: a frontier model's knowledge is a frozen snapshot. Dialogue doesn't stop. dLogos is the live, sourced layer a model (or a product) calls when a question touches the present — across episodes, speakers, and languages.

> **The one rule:** this is **not** a vector index over transcript chunks. It is a graph of **resolved identities and claims over time**, where the expensive reasoning happens on the *write* path so reads are cheap. That's what avoids the "needle-in-a-haystack" collapse of large RAG.

---

## How it works

```
RSS / public transcript
  → transcribe (ASR + diarization)  OR  parse existing transcript   (text → speaker turns)
  → extract  : open-weight LLM → stance-tagged Claims with source spans   [the dominant cost]
  → resolve  : entities → ONE canonical node (Wikidata-anchored); speakers → ONE id across episodes
  → load     : reified Claim nodes + bitemporal edges into the graph
  → serve    : materialized views for product pages; constrained retrieval for "ask"
```

**Data model** — four node types; a **Claim is a node** (not an edge) so it carries stance/sentiment/confidence/source-span and can itself be contradicted or superseded:

```
Speaker ──asserts──▶ Claim ──about──▶ Entity         (the subject)
                      Claim ──disputes / agrees_with──▶ Claim
                      Claim ──supersedes──▶ Claim      (same speaker, later in time)
Speaker ──appears_in──▶ Episode
```

Every edge is **bitemporal** (event-time = when said, ingestion-time = when learned) and **invalidate-never-delete**, so belief-over-time is a first-class, append-only query. Transcript text and audio live in **object storage**, addressed by `(episode_id, span)` — the graph stays small and hot.

**Full architecture (build-ready spec):** [`docs/dlogos-graph-architecture.md`](docs/dlogos-graph-architecture.md) — including why/how it scales to millions of episodes across time and cultures.

---

## Status — honest

This is a **working proof-of-concept**, not production.

**Proven, end-to-end on 20 real episodes** (AI-safety / sensemaking podcasts):
- Incremental, **Wikidata-anchored entity resolution** — `OpenAI → 1 node (wd-Q21708200)`, all major people/orgs → one node across episodes.
- **Cross-episode speaker unification** — *Tristan Harris*: 5 appearances across 3 shows → **1 node**.
- Bitemporal load; cross-claim `disputes` / `supersedes`; consensus-over-time aggregation; a constrained retrieval surface (MCP) + a 4-arm eval harness.
- **762 unit tests green, fully offline** (no GPU / DB / keys): `uv run pytest`.

**Not yet built (the real next work):** the production serving layer (columnar/OLAP for claims + ANN index + materialized views with dirty-set freshness), concept-level resolution (concepts still fragment), word-level citation precision, and per-language pipelines. See spec §12.

---

## Quickstart

```bash
uv sync                 # core deps (no GPU/DB/keys needed)
uv run pytest -q        # 762 tests, fully offline

# build the multi-episode graph over public transcripts (needs a DeepInfra key for extraction/embeddings):
cp .env.example .env     # fill EXTRACTION_API_KEY / EMBED_API_KEY (DeepInfra)
uv sync --extra transcripts
uv run python scripts/run_kg_slice.py --transcripts docs/corpus/ai_sensemaking_20.json
# → out/graph.json + a per-entity fragmentation report

# view the resolved graph in a browser:
uv run python -m dlogos.ui.graph_app --graph out/graph.json --port 8765
```

Single-episode-from-audio smoke run and the full bring-up sequence are in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

---

## Layout

```
src/dlogos/
  ingestion/     RSS, audio fetch, public-transcript parsers + TranscriptBackend
  asr/           WhisperX/pyannote + hosted AssemblyAI backends; word-level re-segmentation
  extraction/    open-weight claim extraction (chunking, controlled predicate vocab, grounding)
  resolution/    canonical entity store, rules→embedding→LLM cascade, Wikidata anchoring (the moat)
  speakers/      cross-episode speaker identity (voiceprint / name-canonical, persistent)
  graph/         reified-claim store, bitemporal loader, cross-claim relation derivation
  retrieval/     hybrid (semantic + BM25 + graph) + consensus-over-time helper
  eval/          four-arm head-to-head harness + fragmentation metric
  mcp/  ui/      MCP server + a localhost graph viewer
scripts/         run_kg_slice.py (multi-episode), run_smoke_inmemory.py (single-episode)
docs/            architecture spec, runbook, the 20-episode corpus
```

---

*Build the resolution layer like it's the company, because it is. Transcription, extraction, storage, and serving are plumbing with known answers; the resolved, bitemporal, cross-lingual identity of who-said-what-about-whom-and-when is the asset that compounds and can't be cloned.*
