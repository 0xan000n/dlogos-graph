# dLogos PoC — Implementation Build Plan (as built)

**Date:** 2026-06-18
**Status:** Reflects what was built in this repository. All 382 unit tests green
on the core dependency group (`uv sync && uv run pytest`), fully offline.
**Design source:** `docs/superpowers/specs/2026-06-18-dlogos-dialogue-graph-poc-design.md`.

This plan is organized by the spec's six-phase de-risking build sequence (§12):
**spike → slice → extract+load → scale → retrieve → eval.** For each phase it
lists the modules built, their tests, the gate that phase must clear, and the
**live-infra steps still required** to actually run that phase on real podcasts.

The whole tree is built to a strict rule: heavy/optional deps
(torch, whisperx, pyannote, graphiti-core, neo4j, gradio, mcp, FlagEmbedding,
sentence-transformers) are imported **lazily inside functions**, and every stage
has a mock/in-memory/fake implementation so the unit tests run on the core deps
alone with no network, no GPU, no Neo4j, and no API keys. `tests/test_e2e_smoke.py`
proves the full chain wires together offline.

---

## Foundation (cross-cutting — owned before the phases)

Shared substrate every phase imports from. Not a numbered phase, but the
prerequisite for all of them.

- **Modules:** `src/dlogos/schema.py` (the reified `ExtractedClaim`, `Entity`,
  `SpeakerRef`, `Transcript`, `Episode`, `SourceSpan`, `BitemporalFact`, and the
  `Stance` / `EntityType` / controlled-`Predicate` enums); `src/dlogos/config.py`
  (`Settings` via pydantic-settings + singleton); `src/dlogos/pipeline.py` (the
  end-to-end orchestrator); project tooling (`pyproject.toml` with core + 5
  optional extras, `Makefile`, `docker-compose.yml` for Neo4j, `.env.example`,
  `tests/conftest.py` with the shared `FakeEmbedder`).
- **Tests:** `tests/test_schema.py`, `tests/test_pipeline.py`,
  `tests/test_e2e_smoke.py` (22 tests across the three).
- **Key decisions baked in:** controlled predicate vocabulary is a *closed enum
  in the schema* (not a post-hoc pass); claims are reified nodes; facts are
  bitemporal (event-time + ingestion-time + validity interval, invalidate-not-
  delete).
- **Live infra:** none — pure Pydantic.

---

## Phase 1 — Spike: resolve the Graphiti × open-weight extraction seam

**Goal (spec §7.6, §12.1):** decide Approach **A** (Graphiti extracts, pointed
at an OpenAI-compatible open-weight endpoint) vs **B** (we extract pre-formed,
resolved triples; Graphiti is store + temporal manager + retrieval). Decide
**before** scaling, on three axes — claim quality, backfill throughput,
$/episode — with Graphiti's per-add LLM node-dedup **disabled** on the bulk path.

- **Modules built:**
  - `src/dlogos/spike/run_comparison.py` — orchestrates A vs B over N fixture
    episodes; both arms share the same resolved-claim input so the comparison is
    apples-to-apples; the load path runs with per-add LLM dedup bypassed.
  - `src/dlogos/spike/score.py` — the **gate** logic: a winner must clear a
    throughput **and** $-per-episode bar, not only a quality bar (a good-but-slow
    or good-but-expensive shape fails).
- **Tests:** `tests/spike/test_run_comparison.py`, `tests/spike/test_score.py`
  (28 tests). Both arms run against fakes; the throughput/$ gate is asserted
  deterministically.
- **Gate:** pick A or B **and** clear the throughput/$ bar. Working assumption
  (and what the loader is built around) is **Approach B** — it keeps open-weight
  extraction first-class and swappable and makes the dedup bypass natural.
- **Live-infra steps to run for real:** an OpenAI-compatible **open-weight LLM
  endpoint** (vLLM/SGLang on rented GPU) to get true claim quality; a live
  **Neo4j** to measure real load throughput; cost instrumentation to produce the
  real $/episode number. (Offline, the spike exercises the comparison + gate
  logic with fakes and a fake token-cost meter.)

---

## Phase 2 — Slice (adversarial): ingestion + ASR + diarization + speaker identity

**Goal (spec §12.2):** ingest + transcribe + diarize + resolve speakers on an
**adversarial** slice (panel show, remote-heavy interview, ad-saturated show)
and eyeball quality — especially the top failure mode, *confident
misattribution*.

- **Modules built — ingestion (`src/dlogos/ingestion/`):**
  - `podcast_index.py` — typed Podcast Index client (feed resolution + GUIDs;
    httpx, mockable).
  - `charts.py` — build the corpus manifest from public category charts (spec §4).
  - `manifest.py` — the corpus manifest record: `feed_url`, `show_id`, `domain`
    tags, `known_hosts`, and the `deep_backfill` flag for the high-velocity
    subset.
  - `fetch.py` — audio enclosure fetch with content-hash dedupe + **GUID
    idempotency** (re-polling never reprocesses).
  - `queue.py` — work-queue abstraction for backfill fan-out + incremental poll.
- **Modules built — ASR (`src/dlogos/asr/`):**
  - `base.py` — the `ASRBackend` protocol + `drop_low_talk_time_speakers`
    (drops sub-threshold diarization labels; SPoRC precedent).
  - `whisperx_backend.py` — real WhisperX (whisper-large-v3, word timestamps);
    whisperx/torch imported lazily.
  - `diarization.py` — pyannote diarization + token→speaker-turn mapping by
    timing; pyannote/torch imported lazily.
  - `mock_backend.py` — deterministic offline ASR backend (the canonical fixture
    transcript everything downstream is tested against).
- **Modules built — speakers (`src/dlogos/speakers/`):**
  - `identity.py` — host-anchored cross-episode identity: a voiceprint gallery
    resolves per-episode `SPEAKER_xx` labels to canonical hosts (embedder
    injected).
  - `guests.py` — **recurring-guest resolution**: episode metadata + parse of
    the host's *"my guest today is…"* intro + a Wikidata QID, combined across the
    batch so a guest merges across shows (the belief-tracking subject).
- **Tests:** `tests/ingestion/` (51), `tests/asr/` (37 — incl. talk-time pruning
  and diarization mapping), `tests/speakers/` (35). 123 tests total.
- **Gate:** ASR + speaker-attribution quality acceptable on the adversarial
  cases (no confident misattribution survives).
- **Live-infra steps to run for real:** **Podcast Index API key** + live RSS /
  enclosure HTTP for ingestion; **GPU + the `asr` extra** (whisperx,
  pyannote.audio, torch) and a **Hugging Face token** for pyannote weights;
  a real voiceprint **embedding** model for the host gallery; **Wikidata SPARQL**
  for guest QIDs. (Offline, `MockASRBackend` + injected embedder + httpx mocks
  drive the whole slice.)

---

## Phase 3 — Extract + resolve + load

**Goal (spec §12.3):** run extraction (controlled predicate enum) + lightweight
resolution (subject-entity clustering + recurring-guest) + the bulk graph load
(dedup bypassed) on the slice; QA claim/stance quality and **speaker-verified**
citation accuracy.

- **Modules built — extraction (`src/dlogos/extraction/`):**
  - `chunking.py` — overlapping, **speaker-labelled** chunking so claims spanning
    chunk boundaries are not lost and attribute to the right person.
  - `predicates.py` — controlled-predicate-vocabulary helpers (the closed enum
    from `schema.py`, surfaced for the extraction response schema).
  - `extractor.py` — the async open-weight `ClaimExtractor` over a structural
    `AsyncChatClient` (a real `AsyncOpenAI` lazily, a fake in tests); emits
    schema-valid `ExtractedClaim`s; validates that each source span lands in the
    chunk window before accepting it.
- **Modules built — resolution (`src/dlogos/resolution/`):**
  - `subjects.py` — **subject-entity embedding clustering** over the whole
    extracted-claim batch → a shared `canonical_id` (so *Apple* / *iPhone* /
    *Apple hardware* aggregate); conservative merge threshold; FlagEmbedding
    lazy, embedder injectable.
  - `wikidata.py` — lightweight Wikidata linking (httpx lazy) for guest/entity
    canonicalization.
- **Modules built — graph (`src/dlogos/graph/`):**
  - `store.py` — the `GraphStore` protocol + graph records (`ClaimNode`,
    `SpeakerNode`, `EntityNode`, `GraphEdge`) and `GraphitiStore` (graphiti-core
    + neo4j imported lazily inside `connect`).
  - `loader.py` — the **Approach-B** `ClaimLoader`: resolved `ExtractedClaim`s →
    reified graph records + bitemporal edges; `bulk_load(..., bypass_llm_dedup=True)`
    is the path the pipeline uses (spec §7.5/§7.6).
  - `temporal.py` — bitemporal helpers: validity windows, invalidate-not-delete,
    current-state filtering.
  - `fake_store.py` — in-memory `GraphStore` (records claims, counts LLM-dedup
    invocations so the bypass is *testable*); no Graphiti/Neo4j/network.
- **Tests:** `tests/extraction/` (52), `tests/resolution/` (29), `tests/graph/`
  (27). 108 tests total. `test_e2e_smoke.py` asserts `claims_loaded >= 1`,
  `store.llm_dedup_invocations == 0`, and that every loaded claim carries both a
  resolved speaker id and a `canonical_id`.
- **Gate:** claims are sourced, stance-correct, and citations pass the
  **speaker-verified** check (the person speaking at the cited timestamp is the
  attributed speaker — `eval/rubric.verify_citation`, exercised in the smoke
  test).
- **Live-infra steps to run for real:** the **open-weight LLM endpoint** for
  extraction; a real **embedding endpoint** (BGE-M3 via the `embed` extra or a
  served endpoint) for clustering; **live Wikidata SPARQL**; **Neo4j** + the
  `graph` extra for `GraphitiStore.connect(...)` (offline uses `FakeGraphStore`).

---

## Phase 4 — Scale (full backfill)

**Goal (spec §12.4):** full backfill — **6-month broad across all ~200 shows +
18–24-month deep for the ~15–25 high-velocity subset** — capturing cost +
throughput.

- **What's built:** the orchestration that makes scale a *quantity* change, not a
  new code path. `pipeline.py` runs each episode through ASR → prune → identity →
  chunk → extract → stamp-speaker, accumulates the whole batch, then runs subject
  resolution **once** over the batch and a **single bulk load** (so the load
  bypasses per-add LLM dedup across the whole corpus, not per episode). Recurring
  guests are pre-resolved across the batch before per-episode processing. The
  `manifest.deep_backfill` flag marks the deep-tier subset; `ingestion/queue.py`
  is the fan-out primitive.
- **Tests:** covered by `tests/test_pipeline.py` (multi-episode batch behavior,
  guest pre-resolution, bulk-load-once, the canonical-id refresh back onto
  per-episode runs) and `tests/ingestion/` (the manifest + queue).
- **Gate:** capture cost + throughput numbers; nothing scales past a failing
  earlier gate.
- **Live-infra steps to run for real:** all of Phase 2 + Phase 3 infra at corpus
  volume — **rented GPU** for batched ASR + extraction, the **embedding
  endpoint**, **Neo4j**, the **Podcast Index** feed set — plus **cost
  instrumentation** to produce the firehose-economics byproduct ($/episode,
  episodes/hour). This is the phase that genuinely cannot run without infra; the
  code path is proven offline on the fixture batch.

---

## Phase 5 — Retrieve

**Goal (spec §12.5):** wire hybrid retrieval + temporal filters + the consensus
helper (bucketed by `canonical_id`) + the MCP server.

- **Modules built — retrieval (`src/dlogos/retrieval/`):**
  - `hybrid.py` — semantic + BM25 + breadth-first graph traversal, **RRF**-fused,
    with **event-time validity filters**; `claims_from_graph_store` materializes
    retrievable claims from the store; an in-memory retrieval store backs tests.
  - `consensus.py` — `consensus_over_time`: buckets stance-tagged claims for a
    subject over time windows, keyed by the resolved `canonical_id` (the
    "how has the consensus moved" primitive, with per-speaker breakdown).
- **Modules built — MCP (`src/dlogos/mcp/server.py`):** five tools —
  `search_dialogue`, `who_discussed`, `consensus_trend`, `belief_history`,
  `provenance_lookup`. Each is a plain **handler function** over an injected
  `RetrievalSurface` (unit-tested with a fake, no `mcp` import); `build_server`
  lazily imports `mcp` and registers thin wrappers. `GraphRetrievalSurface`
  (with `from_graph_store(...)`) is the real adapter that wires the handlers to
  the actual `HybridRetriever` + `consensus_over_time` over a loaded store;
  `PipelineResult.build_retrieval_surface(...)` goes from a pipeline run straight
  to a queryable surface.
- **Tests:** `tests/retrieval/` (30), `tests/mcp/` (15 — handler behavior +
  the surface wiring). 45 tests total.
- **Gate:** retrieval returns attributed, time-filtered, provenance-bearing hits;
  the consensus primitive aggregates by `canonical_id` rather than fragmenting.
- **Live-infra steps to run for real:** **Neo4j** + the embedding endpoint for
  corpus-scale hybrid retrieval; the **`mcp` extra** to actually serve the tools
  to Claude (`build_server(...).run()`). Offline, the in-memory store + fake
  embedder exercise every handler and the real `GraphRetrievalSurface`.

---

## Phase 6 — Eval (the greenlight artifact)

**Goal (spec §12.6, §9):** build the golden set with **pre-registered
good-answer shapes**, run the **four-arm** head-to-head (alone / +web-search /
+naive-vector-RAG / +dLogos), **blind-score** with a **second rater on a subset**
(report agreement), produce the side-by-side artifact. The rubric **elevates
temporal-consensus synthesis** and **demotes recency / couldn't-have-known**.

- **Modules built — eval (`src/dlogos/eval/`):**
  - `golden.py` — `GoldenQuery` + the five archetypes (temporal/consensus,
    per-speaker belief, contradiction, consensus-vs-outlier, provenance), the
    eight domains, and `AnswerShape` (the **pre-registered** expected
    speakers/stance/timeframe/citations, frozen before any arm output).
  - `arms.py` — the four arms as async callables behind one `Answer`(text +
    `Citation`s) interface: `ModelAloneArm`, `ModelWebSearchArm`,
    `ModelNaiveRagArm`, `ModelDLogosArm` (+ `DLogosGraphRetriever`). All
    collaborators injected (frontier `ChatClient`, web-search tool, vector
    retriever, dLogos retriever); nothing heavy at module top level.
  - `rubric.py` — the reweighted scorer **+** the **speaker-verified citation
    check** (`verify_citation`): a citation passes only if the person speaking at
    the cited timestamp is the attributed speaker; attribution precision is
    *capped* by the verified fraction.
  - `blind.py` — deterministic, seeded blinding of arm identity (and unblinding
    of scores) so the rater never sees which answer is dLogos.
  - `agreement.py` — percent agreement + Cohen's κ over ordinal-binned totals for
    the second-rater control.
  - `runner.py` — `EvalRunner` ties it together: run four arms per query → blind
    → score (cite-check on) → unblind → optional second-rater agreement →
    `EvalReport` rendered as **JSON + Markdown** side-by-side (mean total + wins
    per arm, per-query verbatim answers, verified/rejected citation counts, and
    the existence-proof spread line).
- **Tests:** `tests/eval/` — `test_golden`, `test_arms`, `test_rubric`,
  `test_blind`, `test_agreement`, `test_runner` (45). Fake arms + a deterministic
  fake rater make the whole harness reproducible.
- **Gate / framing:** the artifact is an **existence proof across a spread**, not
  a generalization proof (15 queries cannot cover 8 domains × 5 archetypes). The
  win must show on **temporal-consensus synthesis across attributed sources** —
  the dimension that survives the web-search and vector-RAG competitors.
- **Live-infra steps to run for real:** a **strong frontier model** API key for
  all four arms (and optionally a separate **web-search-enabled** endpoint for
  arm 2); a **loaded dLogos graph** (Phases 2–4) for arm 4 and a naive vector
  index over the *same* transcripts for arm 3; **human raters** (the primary +
  the second-rater subset) to produce the blinded scores. Offline, fake arms and
  a fake rater produce a complete, deterministic `EvalReport`.

---

## Cross-phase status summary

| Phase | Modules | Tests (green) | Runs offline now | Needs live infra |
|---|---|---|---|---|
| Foundation | schema, config, pipeline, tooling | 22 | ✅ | — |
| 1 Spike | spike/run_comparison, score | 28 | ✅ (fakes + gate) | open-weight endpoint, Neo4j, cost meter |
| 2 Slice | ingestion/*, asr/*, speakers/* | 123 | ✅ (mock ASR, mocks) | Podcast Index, GPU ASR + HF token, embedder, Wikidata |
| 3 Extract+load | extraction/*, resolution/*, graph/* | 108 | ✅ (fake store/embedder/client) | open-weight LLM, embedding endpoint, Neo4j, Wikidata |
| 4 Scale | pipeline batch + manifest/queue | (in pipeline/ingestion) | ✅ (fixture batch) | all of 2+3 at volume + cost instrumentation |
| 5 Retrieve | retrieval/*, mcp/server | 45 | ✅ (in-memory store) | Neo4j, embedding endpoint, `mcp` extra |
| 6 Eval | eval/* | 45 | ✅ (fake arms/rater) | frontier model, loaded graph, human raters |

**Total: 382 tests, all green on the core dependency group, fully offline.**
Each gate can send the work back a step; nothing scales past a failing gate.

---

## Deferred (documented seams — not built; spec §13)

Full ER cascade at 100M+ entities; ClickHouse analytical mirror; multilingual;
sub-minute real-time; productized public API/CLI; owned steady-state GPU fleet;
and the **legal / rights / ToS / fair-use** review required before any
productized surface ships (the PoC redistributes nothing — corpus, transcripts,
and claims stay internal; the only external artifact is the side-by-side
scorecard).
