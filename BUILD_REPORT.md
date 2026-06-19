# dLogos PoC — Build Report

**Date:** 2026-06-18
**Test state:** `uv sync && uv run pytest` → **382 passed**, fully offline (core
dependency group only — no GPU, no Neo4j, no network, no API keys).
**Scope of this report:** an honest, per-subsystem account of what is
implemented, whether its unit tests are green, whether running it *for real*
needs live infra, and the known gaps/TODOs.

Legend: ✅ yes · ⚠️ partial / with caveat · 🔌 needs live infra to run for real.

---

## Summary

Every subsystem in the canonical layout exists and is unit-tested green on the
core deps. The pipeline composes end-to-end **offline** through injected fakes
(`tests/test_e2e_smoke.py` proves the full ingest→…→dLogos-arm chain with a
speaker-verified citation). The honest caveat throughout: **unit-tested green
means the logic and wiring are proven against fakes; it does not mean the stage
has been run against real podcast audio, a real open-weight model, a live Neo4j,
or a frontier eval model.** Those are the four named live-infra dependencies and
none of them is exercised by the test suite (by design — heavy deps are lazy and
mocked).

---

## Per-subsystem

### Foundation — `schema.py`, `config.py`, `pipeline.py`, tooling
- **Implemented?** ✅ Shared reified-Claim/bitemporal schema; pydantic-settings
  `Settings` + singleton; the end-to-end `Pipeline` orchestrator; `pyproject`
  (core + 5 optional extras), `Makefile`, `docker-compose.yml`, `.env.example`,
  `conftest.py` (`FakeEmbedder`).
- **Unit-tested green?** ✅ `test_schema.py`, `test_pipeline.py`,
  `test_e2e_smoke.py` (22).
- **Needs live infra?** No — pure Pydantic / wiring.
- **Known gaps/TODOs:** Pipeline drops one-off unresolved speakers from the load
  (kept in per-episode runs for audit) — acceptable per spec §7.3, but it means
  the long-tail speaker's claims never reach the graph unless a
  `fallback_speaker_id` hook is injected. `**pipeline_kwargs` is `type: ignore`d
  through `run_pipeline` (loose typing at that one seam).

### Ingestion — `podcast_index`, `charts`, `manifest`, `fetch`, `queue`
- **Implemented?** ✅ Typed Podcast Index client; chart→manifest builder;
  manifest record (with `deep_backfill` flag); content-hash + **GUID-idempotent**
  fetch dedupe; work-queue abstraction.
- **Unit-tested green?** ✅ `tests/ingestion/` (51), httpx mocked.
- **Needs live infra?** 🔌 **Podcast Index API key** + live RSS / enclosure HTTP
  for real feeds and audio.
- **Known gaps/TODOs:** the queue is an in-process/abstract primitive, not a real
  SQS/Cloudflare/Postgres job table (spec calls those out as PoC-acceptable);
  object storage is referenced conceptually but not wired to S3/R2/MinIO —
  `fetch` returns bytes/hashes, persistence to cold storage is left to the
  caller. No retry/backoff policy on real network is implemented beyond what
  `tenacity` (a core dep) would provide if added.

### ASR — `base`, `whisperx_backend`, `diarization`, `mock_backend`
- **Implemented?** ✅ `ASRBackend` protocol + talk-time pruning; real WhisperX
  backend; pyannote diarization + token→turn mapping; deterministic
  `MockASRBackend`.
- **Unit-tested green?** ✅ `tests/asr/` (37) — talk-time pruning, diarization
  mapping, mock determinism. The real whisperx/pyannote backends are **not**
  executed by tests (heavy deps lazy).
- **Needs live infra?** 🔌 **GPU + the `asr` extra** (whisperx, pyannote.audio,
  torch, faster-whisper) and a **Hugging Face token** for pyannote weights.
- **Known gaps/TODOs (honest):** the `whisperx_backend` / `diarization` code
  paths are written against the documented WhisperX/pyannote APIs but have **not
  been run against real audio** — the diarization→confident-misattribution top
  risk (spec §11) is only mitigated in *logic* (talk-time drop, the eval's
  speaker-verified check); it is **not yet validated on the adversarial slice**
  (panel / remote-heavy / ad-saturated shows). That validation is a real,
  outstanding gate that requires GPU + audio.

### Speakers — `identity`, `guests`
- **Implemented?** ✅ Host-anchored voiceprint gallery (embedder injected);
  recurring-guest resolution combining episode metadata + the "my guest today
  is…" intro parse + a Wikidata QID, merged across the batch.
- **Unit-tested green?** ✅ `tests/speakers/` (35).
- **Needs live infra?** 🔌 a real **voiceprint embedder** for the gallery;
  **Wikidata SPARQL** for guest QIDs (httpx mocked in tests).
- **Known gaps/TODOs:** the intro-pattern parser is heuristic (matches a small
  set of "my guest today is…"-style phrasings) and will miss atypical intros;
  guest resolution depends on at least one of the three signals firing — a guest
  with no metadata, no canonical intro, and no Wikidata hit stays a per-episode
  speaker. Voiceprint matching quality is entirely a function of the (not-yet-
  real) embedder.

### Extraction — `chunking`, `predicates`, `extractor`
- **Implemented?** ✅ Overlapping speaker-labelled chunking; controlled-predicate
  helpers over the closed schema enum; async `ClaimExtractor` over a structural
  `AsyncChatClient` (real `AsyncOpenAI` lazy), with source-span-in-window
  validation.
- **Unit-tested green?** ✅ `tests/extraction/` (52), fake async client.
- **Needs live infra?** 🔌 an OpenAI-compatible **open-weight LLM endpoint**
  (vLLM/SGLang on rented GPU; DeepSeek/Gemma/Kimi).
- **Known gaps/TODOs (honest):** the controlled predicate vocabulary is enforced
  by the response **schema** (Pydantic validation), but **JSON-schema-constrained
  decoding** at the endpoint is the deployer's responsibility — against a real
  open model, malformed/over-free output is possible and only caught as a
  validation failure, not prevented. The spec's #1 risk (open-weight structured-
  output reliability) is therefore **not yet measured** — that is exactly what
  the Phase-1 spike is for, and the spike has only been run against fakes.
  Extraction *hallucination* is mitigated by span-in-window checks but not by any
  real-model QA.

### Resolution — `subjects`, `wikidata`
- **Implemented?** ✅ Batch subject-entity embedding clustering →
  `canonical_id` (conservative threshold); lightweight Wikidata linking.
- **Unit-tested green?** ✅ `tests/resolution/` (29), `FakeEmbedder` + httpx
  mocked.
- **Needs live infra?** 🔌 a real **embedding endpoint** (BGE-M3 via the `embed`
  extra or served) for real surface forms; **live Wikidata SPARQL**.
- **Known gaps/TODOs:** the clustering threshold is tuned for the deterministic
  fake embedder and is **untuned on real BGE-M3 vectors** — the merge precision
  vs. recall tradeoff (fragmentation vs. wrong-merge) is unvalidated on real
  data. Clustering is single-pass greedy by similarity; no Louvain/community
  step (deferred to the at-scale cascade, spec §13).

### Graph — `store`, `loader`, `temporal`, `fake_store`
- **Implemented?** ✅ `GraphStore` protocol + records; `GraphitiStore`
  (graphiti-core + neo4j lazy in `connect`); the **Approach-B** `ClaimLoader`
  with `bulk_load(bypass_llm_dedup=True)`; bitemporal helpers (validity windows,
  invalidate-not-delete, current-state); in-memory `FakeGraphStore` that counts
  LLM-dedup invocations so the bypass is *assertable*.
- **Unit-tested green?** ✅ `tests/graph/` (27); the smoke test asserts
  `llm_dedup_invocations == 0` on the bulk path.
- **Needs live infra?** 🔌 **Neo4j** (`docker compose up neo4j`) + the `graph`
  extra.
- **Known gaps/TODOs (honest):** `GraphitiStore` is a **thin adapter written
  against the Graphiti/Neo4j API but never executed against a live Neo4j** in
  tests. Whether Graphiti's bulk path *actually* bypasses per-add LLM dedup as
  wired (vs. the FakeStore which models the intended behavior) is **unverified on
  the real backend** — this is the second-biggest spec risk (§7.6) and is
  exactly what Phase 1 must confirm with live infra. Edge invalidation logic is
  implemented in our `temporal.py`; reconciling it with Graphiti's own
  invalidation on the real store is untested.

### Retrieval — `hybrid`, `consensus`
- **Implemented?** ✅ Hybrid semantic + BM25 + graph-traversal, RRF-fused, with
  event-time validity filters; `consensus_over_time` bucketed by `canonical_id`
  with per-speaker breakdown.
- **Unit-tested green?** ✅ `tests/retrieval/` (30), in-memory store + fake
  embedder.
- **Needs live infra?** 🔌 Neo4j + embedding endpoint for corpus-scale retrieval
  (the in-memory store backs offline runs).
- **Known gaps/TODOs:** BM25 + RRF run over an **in-memory** materialization of
  claims from the store, not over Neo4j's native vector/full-text indexes — at
  corpus scale the real path should push retrieval into Neo4j/Graphiti; the
  in-memory store is a PoC-scale stand-in. No cross-encoder reranker is wired
  (the spec lists it as optional).

### MCP — `server`
- **Implemented?** ✅ Five tools (`search_dialogue`, `who_discussed`,
  `consensus_trend`, `belief_history`, `provenance_lookup`); plain handler
  functions over an injected `RetrievalSurface`; `GraphRetrievalSurface` real
  adapter; `build_server` (mcp lazy).
- **Unit-tested green?** ✅ `tests/mcp/` (15) — handlers + surface wiring, no
  `mcp` import needed.
- **Needs live infra?** 🔌 the **`mcp` extra** to actually serve the tools to a
  client; a loaded graph behind the surface.
- **Known gaps/TODOs:** `build_server` (the lazy-`mcp` registration) is **not
  executed by tests** (it needs the `mcp` package); only the handlers and the
  `GraphRetrievalSurface` are exercised. `belief_history` derives a per-person
  stance direction from the bucket's net sentiment (no per-speaker stance scalar
  is carried in the bucket) — a documented approximation in the code.

### UI — `app`
- **Implemented?** ✅ Pure render functions (four-arm side-by-side Markdown
  panels, citation formatting, graceful per-arm error handling); `build_ui`
  (gradio lazy); `python -m dlogos.ui.app` entry conceptually wired via
  `make ui`.
- **Unit-tested green?** ✅ `tests/ui/` (11) — the pure renderers.
- **Needs live infra?** 🔌 the **`ui` extra** (gradio) to launch; real arms
  behind it.
- **Known gaps/TODOs (honest):** `build_ui` (the gradio shell) is **not executed
  by tests**; only the pure render functions are. The UI's `make ui` /
  `python -m dlogos.ui.app` launch path needs an `arms` list to be wired (the
  module exposes `build_ui(arms)` but a zero-config `__main__` that constructs
  real arms from settings is **not** provided — launching it for real requires a
  small wiring script with a frontier client + a retriever over a loaded graph).

### Spike — `run_comparison`, `score`
- **Implemented?** ✅ Approach A vs B comparison over N fixtures (dedup bypassed
  on the load path); the **throughput / $-per-episode gate** (a good-but-slow or
  good-but-expensive shape fails).
- **Unit-tested green?** ✅ `tests/spike/` (28), fakes + a fake cost meter.
- **Needs live infra?** 🔌 open-weight endpoint + Neo4j to produce **real**
  throughput / $ numbers (the gate logic itself is offline).
- **Known gaps/TODOs (honest):** the spike has only been run against **fakes** —
  it proves the comparison harness and the gate arithmetic, **not** the actual
  A-vs-B outcome or any real economics. The working assumption (Approach B) is
  baked into the loader but **not yet empirically confirmed**. The $/episode
  meter is a stand-in; a real `AsyncOpenAI` exposes no running token total, so
  real cost capture needs deployer-side instrumentation.

### Eval — `golden`, `arms`, `rubric`, `blind`, `agreement`, `runner`
- **Implemented?** ✅ Golden query set (5 archetypes × 8 domains, pre-registered
  `AnswerShape`s); the four arms (all collaborators injected); the reweighted
  rubric **+** speaker-verified citation check; seeded blinding/unblinding;
  Cohen's-κ agreement; `EvalRunner` → JSON + Markdown side-by-side artifact.
- **Unit-tested green?** ✅ `tests/eval/` (45), fake arms + fake rater.
- **Needs live infra?** 🔌 a **strong frontier model** (all four arms; optional
  separate web-search endpoint for arm 2), a **loaded dLogos graph** + a naive
  vector index over the same transcripts, and **human raters** for the blinded
  scores.
- **Known gaps/TODOs (honest):** the eval harness is fully built and deterministic
  but has **produced no real scorecard** — there are no live arm outputs, no
  human ratings, and therefore **no greenlight artifact yet**. The 15 hero
  queries / pre-registered shapes are the harness's *capability*, not a populated
  golden set against the real corpus. The rater is an injected callable; whether
  a real LLM-judge or human produces stable, blind, high-agreement scores is
  unmeasured.

---

## Honest bottom line

- **What is true:** the complete pipeline and the four-arm eval are implemented,
  wired, and **green offline** (382 tests), with every heavy dependency isolated
  behind a lazy import + a fake. The end-to-end smoke test proves the chain holds
  together and that a dLogos answer's citation survives the speaker-verified
  check.
- **What is not yet true:** nothing has been run against **real audio, a real
  open-weight model, a live Neo4j, or a frontier eval model**. The four highest
  spec risks — diarization→misattribution on the adversarial slice, open-weight
  structured-output reliability (the Phase-1 spike), the Graphiti bulk-dedup-
  bypass on the real backend, and subject-clustering precision on real
  embeddings — are mitigated **in logic and tested against fakes**, but are
  **empirically unvalidated**. They are gated behind the live-infra steps listed
  per subsystem above and summarized in the build plan.
- **The deliverable that greenlights the company — the populated four-arm
  scorecard — does not exist yet.** The harness to produce it does.
