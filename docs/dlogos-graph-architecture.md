# dLogos Knowledge Graph — Architecture Spec

**Audience:** a senior architect building this for real.
**Status of this doc:** the design as validated by a working PoC (20 real episodes, end-to-end). What's proven vs. to-build is called out in §12.
**One sentence:** a *resolved, bitemporal, multilingual dialogue graph* — speakers, entities, and time-stamped, stance-tagged claims extracted from conversation at scale — served to products through precomputed views, not as a RAG-over-chunks.

---

## 0. The one rule that determines whether this works

> **This is not a vector index over transcript chunks. It is a graph of *resolved identities and claims over time*, and reads are served from materialized views.**

Every hard decision below follows from that rule. If you let it degrade into "embed every chunk, semantic-search the haystack," you have rebuilt large-RAG and inherited its scale failure (recall collapse / "needle in a haystack" as the corpus grows). The graph's job is to **convert similarity-search-over-a-haystack into identity-lookup-over-a-resolved-structure**, and to do the expensive reasoning on the *write* path so reads are cheap.

---

## 1. Design principles (non-negotiable)

1. **Reify claims as nodes, not edges.** A claim must carry stance/sentiment/confidence/source-span and must itself be contradictable and supersedable. An edge can't.
2. **Canonical identity is the product.** The same person/org/concept/claim becomes **one** node across all episodes. The pipeline (transcription, extraction) commoditizes; the *resolved, deduplicated, cross-episode graph* is the moat. Resolution quality is the single biggest lever on everything downstream.
3. **Bitemporal, append-only, invalidate-never-delete.** Every fact carries *event-time* (when said) and *ingestion-time* (when learned) plus a validity window. You compound knowledge; you never rebuild. Belief-state can only accumulate — that's the durable head start.
4. **CQRS: heavy write path, cheap read path.** Ingest → resolve → graph is where cost lives. Product pages read precomputed materialized views in milliseconds. Live graph/OLAP/vector queries are reserved for the interactive "ask" surface only.
5. **Retrieval must be constrained before it is semantic.** Any semantic search must be pre-filtered by at least one of {entity, speaker, topic, time-window}. Unconstrained vector search across the corpus is forbidden by the API, not just discouraged.
6. **Language-neutral core, source-language surface.** The canonical layer (ids, predicates, stance, QIDs) is language-agnostic; claim text is stored in source language (+ optional normalized form). One global identity namespace spans cultures.

---

## 2. Data model

### Node types
| Node | Key | Carries |
|---|---|---|
| **Speaker** | `speaker_id` (QID-anchored or voiceprint/name hash) | name, aliases, `wikidata_qid`, is_host |
| **Entity** | `canonical_id` (`wd-<QID>` or content hash) | name, `type` ∈ {person, org, concept, work}, aliases, `wikidata_qid`, embedding |
| **Claim** *(reified)* | `claim_id` (deterministic content hash → idempotent loads) | predicate (controlled vocab), stance, sentiment, confidence, object, **source_span** {episode, t_start, t_end}, denormalized speaker_id + subject_canonical_id |
| **Episode** | `episode_id` (feed GUID) | show_id, published_at, language, audio_hash, **`transcript_ref`** (a pointer to object storage — *not* the text) |
| **Segment** *(optional skeleton — §2.1)* | `segment_id` | speaker_id, `t_start`, `t_end` — **no text**; the addressable handle a Claim's provenance edge points at |

### Edge types (all bitemporal)
```
Speaker ──asserts──▶ Claim
Claim   ──about────▶ Entity        (the subject)
Claim   ──mentions─▶ Entity        (secondary refs)
Claim   ──disputes / agrees_with──▶ Claim   (cross-speaker, same subject, opposite/same stance)
Claim   ──supersedes─▶ Claim       (same speaker, same subject, later in event-time)
Speaker ──appears_in─▶ Episode
Claim   ──derived_from─▶ Segment    (optional, fine-grained provenance — §2.1)
```

### Bitemporal fields (every edge)
`event_time`, `ingestion_time`, `valid_from`, `valid_to|null`, `invalidated:bool`.
Contradiction/supersession sets `valid_to` + `invalidated`; rows are **never deleted**. A "current-state" read filters `invalidated=false AND valid_to IS NULL`.

**Ids are deterministic content hashes** so re-ingesting an episode is idempotent (no double-counting on backfill re-runs).

### 2.1 What lives in the graph vs. object storage (the hot/cold rule)

> **Store in the graph what you *traverse*; store in object storage what you *dereference*.**

The graph holds the **small, hot** layer — resolved identities, claims, edges, spans — traversed constantly for cross-episode queries. The **transcript text and raw audio are the largest and coldest data** in the system (read only when a user opens *that one* episode), so they live in **object storage**, addressed by `episode_id` + span. `Episode.transcript_ref` is a pointer, never the text. Product surfaces that display the transcript render the text *from object storage* and overlay the graph's entity/claim annotations, joined by `(episode_id, span)` (see §8).

**Why not inline the transcript in the graph.** It is by far the largest data by volume. At 10⁸ episodes, modelling each utterance as a node is **~30B text-bearing nodes** that would dwarf the ~18B claims and ~10M entities — ballooning graph memory, index, backup, and replication, and paying fast-store prices for cold text that is read rarely. The graph's traversal speed *depends* on staying small and hot; co-locating cold bulk defeats the whole point. ("Not searchable" removes the index cost, but **not** the storage/ops cost.)

**Scale guidance:**
- **Small / medium (≤ low-millions of episodes):** inlining the transcript on the `Episode` node is fine and simpler — don't over-engineer early.
- **At scale:** keep text in object storage. If you want first-class, *traversable* provenance (a claim edged to the exact utterance it came from), add the lightweight **`Segment` skeleton** — `Segment {segment_id, speaker_id, t_start, t_end}` with **no text** — plus `Claim ──derived_from──▶ Segment`. Provenance becomes a graph edge, the transcript is one dereference away, and you never store words in the hot store. (The plain `source_span` pointer on the Claim already covers citations; add `Segment` only when you want provenance as an edge.)

> PoC reference: `src/dlogos/schema.py`, `src/dlogos/graph/store.py`, `src/dlogos/graph/loader.py`.

---

## 3. Construction pipeline (incremental, idempotent, event-driven)

```
new episode (RSS push / poll, or text transcript)
  → fetch + content-hash dedupe + cold-store raw         (object storage)
  → transcript: ASR+diarization+word-timestamps  OR  parse existing public transcript
  → chunk (overlap, speaker labels carried in)
  → EXTRACT: open-weight LLM → stance-tagged Claims with source spans  [dominant cost]
  → GROUND each claim's span to real transcript timing
  → RESOLVE (the heart, §4): entities → canonical_id (incremental, vs the global set)
                              speakers → canonical speaker_id
  → LOAD: idempotent upsert into the graph; derive cross-claim edges
  → EMIT dirty-set {entities, speakers, topics touched} → refresh affected views (§6)
```

Two invariants:
- **Incremental, not batch.** Episode N resolves against the *existing* canonical set; it never recomputes 1..N−1. Work per episode is O(new), not O(all). This is the only thing that makes a firehose tractable.
- **Open-weight extraction.** Self-host/rent open-weight models (DeepSeek/Gemma/Kimi class) for extraction — the dominant marginal cost — so the firehose is amortized GPU-hours, not per-token frontier billing. This is the economic thesis; do not put a frontier per-token API in the extraction hot path.

> PoC reference: `src/dlogos/pipeline.py`. Real-infra run: `scripts/run_kg_slice.py`.

---

## 4. Resolution layer — the moat *and* the hard part

This is where the system is won or lost. Goal: the same real-world thing → exactly one canonical node, across episodes, runs, languages.

### Components
1. **Persistent canonical entity store** (`canonical_id → {name, type, qid, embedding, aliases}`). Append-grows; the *single source of truth* for identity. Backed by a real store at scale (see §6); the `candidates(embedding, type, k)` method is the **only** similarity entry point — the ANN seam.
2. **Blocking.** Never compare all-pairs (quadratic, impossible at scale). Embedding ANN (FAISS/HNSW/ScaNN) returns ~k candidate canonicals per new mention. Resolution is then ~linear in new mentions.
3. **Cascade matcher** (cheap→expensive, short-circuit at first decision):
   - **Tier 1 — rules (free, decisive):** shared Wikidata QID, or exact normalized-name match.
   - **Tier 2 — embedding:** cosine vs top candidate; per-type thresholds (orgs/people tight; concepts stricter band).
   - **Tier 3 — LLM adjudication:** *only* the ambiguous middle escalates to a yes/no model call. No adjudicator → conservative NEW (prefer fragmentation over a wrong merge).
4. **Wikidata QID anchoring** for people/orgs → `canonical_id = wd-<QID>`. This is the **global, cross-lingual namespace** (see §11) and the highest-leverage resolution lever.
5. **Merge by graph clustering, not transitive closure.** Pairwise "A=B, B=C ⇒ A=C" over-merges and silently collapses distinct entities. Use community detection (Louvain) over the match graph.
6. **Speaker identity:** for audio, voiceprint gallery (voice is language-agnostic — a culture-proof signal) + name + QID; for text transcripts, the label *is* the name → canonicalize by name/QID. Same person across shows/languages → one `speaker_id`.

### The honest difficulty
- **Named entities (people/orgs) resolve well** — QID anchor + tight embedding. (PoC: OpenAI→1 node `wd-Q21708200`, all major people→1.)
- **Concepts are the unsolved frontier.** They paraphrase infinitely, lack clean QIDs, and the extractor emits compounds ("AGI and ASI stage", "Google's approach"). (PoC: `AGI→14 nodes`, `AI safety→9`.) Mitigations: constrain the extractor to base-entity + relation (don't let it emit compound subjects); per-type thresholds; a curated concept ontology + human-in-the-loop merge review for the high-value concepts; cross-lingual concept linking (§11). **Assume concept resolution needs continuous investment — it is the long pole.**

> PoC reference: `src/dlogos/resolution/{canonical_store,cascade,incremental,wikidata}.py`, `src/dlogos/speakers/`.

---

## 5. Temporal model

- **Two clocks because the world has two.** *event-time* (episode date) ≠ *ingestion-time* (when processed). They diverge constantly: you backfill a 2019 episode in 2026, a show is re-released, a claim is corrected. Both are needed to answer "**as of date T, what did the corpus believe**" vs "what do we now know was said."
- **Belief-over-time = aggregation, not retrieval.** "How has consensus on X moved" is: take all claims `about` canonical X, bucket by event-time, aggregate stance/sentiment. Because X is *one* canonical subject and speakers are *one* canonical each, the time series is signal, not fragmented noise. This is an **OLAP `GROUP BY`** — it has no needle-in-haystack problem by construction (you *want* all rows).
- **Mind-changes are first-class** via `supersedes` (same speaker reverses on a subject over event-time) and `disputes` (speakers disagree). These are the graph-native "contrasting claims" signal — far more defensible than "verified/false" (you show the dialogue; you are not an arbiter of truth).

---

## 6. Storage & serving architecture (where it actually scales)

The data splits along a natural seam; use four stores, each scaling independently, behind stable interfaces so they swap without touching callers.

| Concern | Store | Why |
|---|---|---|
| Raw audio + transcripts | **Object storage** (S3/R2) | infinite, cheap, cold; stream-through then discard/cold-store audio |
| **Claim facts** (append-only, time-stamped) | **Columnar / OLAP** (ClickHouse) | "consensus over time" is a time-bucket aggregation over billions of rows — exactly the columnar sweet spot; cheap per-TB |
| **Identity + relationships** | **Graph** (Neo4j / FalkorDB / Neptune) — smaller | traversals ("entity → all episodes", "who disagrees") are graph-shaped; the resolved identity layer is far smaller than the fact layer |
| **Blocking + semantic retrieval** | **Vector index** (FAISS/HNSW/ScaNN) | ANN candidate generation for resolution *and* constrained semantic search |

### CQRS: the read/write split (this is the architectural correction that makes the product fast)
- **Write path** (ingest → resolve → load) does all the expensive work and **materializes views**.
- **Read path** (product pages) serves **precomputed materialized views** — milliseconds, no live reasoning, no per-pageview LLM. The four flagship product sections each map to one view:

```
episode_summary_view            (episode_id → generated summary + top claims)
episode_claims_view             (episode_id → attributed, timestamped, citable claims)
episode_entities_view           (episode_id → canonical entities mentioned, with spans)
entity_related_episodes_view    (canonical_id → episodes/speakers discussing it, ranked)
speaker_claims_about_entity_view(speaker_id, canonical_id → claims over time)
```

- **Interactive only** ("ask this/across episodes") hits the live graph/OLAP/vector — and even then, **constrained** (§7).

### View freshness — the genuinely hard part (design it deliberately)
Materialized views are easy to describe, hard to keep correct under an append-only firehose. The day a new episode mentions Sam Altman, his `entity_related_episodes_view` and every `speaker_claims_about_entity_view` touching him go stale. You cannot full-recompute at world scale. Therefore:
- Each ingested episode emits a **dirty set** = {touched canonical entities, speakers, topics}.
- Only the views keyed by those dirty ids are **incrementally rebuilt** (a queue of view-refresh jobs).
- Views carry a `freshness_watermark` (max ingestion-time folded in) so reads can show "as of" and staleness is observable.
This dirty-set propagation is the same incrementality as §3/§4, applied to the read layer. **It is the core engineering of this architecture — not an afterthought.**

> PoC status: ran on in-memory graph + SQLite canonical/speaker stores, no columnar/ANN/materialized-views yet. The interfaces are in place; these are swaps, not rewrites (§12).

---

## 7. Retrieval discipline (the needle-in-haystack defense, made a guardrail)

The "needle in a haystack" / vector-dilution failure is real and well-documented: as a flat vector corpus grows, top-k increasingly returns *semantically similar but wrong* chunks, and ANN recall decays. The defense is **structure before similarity**:

- **The retrieval API refuses unconstrained search.** Every query must carry ≥1 of {entity, speaker, topic, time-window}. Resolve those to canonical ids first; that pre-filters to a small slice; *then* run semantic search **within the slice** and rerank (cross-encoder) the top few.
- **Identity queries don't retrieve at all** — they look up by canonical id (graph/OLAP), no vector step.
- **Global/"what does the field think" queries** use precomputed community/cluster summaries (the Microsoft-GraphRAG move), not a live sweep.
- If a query *would* fan out across the whole corpus, the API degrades (require a filter) rather than silently doing a billion-vector search. **Make large-RAG failure structurally impossible.**

---

## 8. The serving contract (what products build against)

Front-end and graph teams build in parallel against named views/queries:
- **Entity-linked transcript** → `episode_entities_view` (spans → canonical ids → clickable).
- **Related-by-entity** → `entity_related_episodes_view` (the cleanest cross-episode win — *proven*).
- **Claims / key ideas** → `episode_claims_view` (attributed, timestamped, citable; verbatim span).
- **Ask this / across episodes** → constrained retrieval API (§7), filtered by episode/speaker/topic/date.
- **"The dialogue around a claim"** (NOT fact-check) → `disputes` / `agrees_with` / `supersedes` edges. Graph-native, defensible, no truth-oracle.

Everything else on a product page (playback, comments, favorites, stats, sponsors, campaigns, accounts) is the **app database**, not the graph. The graph is the special layer; do not let it become the backend for every component.

---

## 9. Why it scales to **millions of episodes**

Order-of-magnitude (calibrated to the PoC: ~180 claims, ~70 distinct entities per episode):

| Scale | Episodes | Claim rows | Distinct entities | Transcript text | Claim+span store | Embeddings (raw) |
|---|---|---|---|---|---|---|
| Curated | 10⁴ | ~2M | ~10⁵ | ~10 GB | ~2 GB | ~4 GB |
| Broad | 10⁶ | ~180M | ~few×10⁶ | ~1 TB | ~150 GB | ~360 GB |
| Firehose / backlog | 10⁸ | ~18B | ~10⁷ | ~100 TB | ~12 TB | ~36 TB (quantize → ~9 TB) |

The four reasons it holds:
1. **Append-only + incremental** → per-episode work is O(new). No global recompute, ever. Throughput scales by adding extract/resolve workers.
2. **Canonicalization is sub-linear.** Millions of episodes mention the same *finite* set of real entities. The graph grows in **distinct entities/claims, not raw mentions**. Distinct entities stay in the low-tens-of-millions even as claims hit 10¹⁰.
3. **Resolution is blocked, not pairwise.** ANN → ~k candidates per mention; cascade spends the LLM call only on the ambiguous few. ~Linear, not quadratic.
4. **The heavy query is OLAP, not traversal.** "Consensus over time" is a columnar aggregation (10–100B rows is routine for ClickHouse); the graph holds only the smaller identity/relationship layer; reads are materialized. No layer is asked to do something it's bad at.

The constraint to watch is **not** storage (cheap) or compute (parallel) — it's **resolution recall** (do mentions find their canonical node?). That's the real scaling risk, and it lives in the ANN-blocking + concept-resolution quality (§4), not in the graph's size.

---

## 10. Why it scales **across time**

- **Bitemporal append-only** means time is a first-class axis, not a migration problem. New episodes extend the timeline; corrections/re-releases are *invalidations*, not rewrites; history is permanent.
- **The moat compounds with time and cannot be backfilled.** A competitor starting later cannot reconstruct what your graph already recorded about *how* a view moved — belief-state only accumulates forward. Freshness (you ingest continuously) + temporal depth (you started earlier) are the two defenses that strengthen every day.
- **Views are "as-of" queryable.** Because every edge carries event-time + ingestion-time, any view can be reconstructed for a past instant (point-in-time reads), which the temporal product surfaces ("what did the corpus think in 2024 vs now") rely on.

---

## 11. Why it scales **across cultures & languages**

This is a designed-in property, not an afterthought — and a strategic upside (cross-cultural intelligence is a product only this structure can offer).

1. **One global identity namespace = Wikidata QIDs.** Wikidata is inherently multilingual: one QID, labels in 300+ languages. "OpenAI" / "オープンAI" / "OpenAI" (es) all anchor to **Q21708200 → one node**. Cross-lingual entity collapse for the head (named people/orgs) is *free*. This is the single most important multilingual mechanism.
2. **One multilingual embedding space** (BGE-M3 / E5-mistral class, 100+ languages). The long tail (entities/concepts without QIDs) clusters cross-lingually in a shared vector space, and retrieval is cross-lingual by default ("claims about X" returns claims in any language).
3. **Language-agnostic extraction.** Open-weight LLMs extract in the source language; the canonical layer (predicate controlled-vocab, stance enum, ids, QIDs) is language-neutral. Store claim text in source language + an optional normalized/translated form. Every claim/episode carries a `language` tag.
4. **Voiceprints are culture-proof.** A guest on English *and* Spanish podcasts is the same voice regardless of language → one `speaker_id` via the voiceprint gallery even when names transliterate differently.
5. **`language` as a filter *and* an aggregation dimension.** You can serve per-community views ("what the Spanish-language AI sphere thinks") *and* global consensus — and **diff them** ("how does the AI-safety conversation differ between the English and Chinese podcast spheres"). That cross-cultural diff is a unique, defensible product the architecture enables for free.
6. **Cultural concepts are the frontier** (harder than entities). Concepts that exist in one culture but not another, or carry different meaning, lack clean QIDs and don't always cluster across languages. Handle with: per-language concept resolution → a **cross-lingual concept-linking pass** (link clusters that are translations/equivalents) → human-in-the-loop ontology for the high-value cultural concepts. Be honest that this is unsolved at the frontier; design for it, don't pretend it's automatic.

---

## 12. Proven vs. to-build (so you know the starting point)

**Proven on 20 real episodes (end-to-end, real infra):**
- Reified-claim bitemporal model; idempotent loads.
- Incremental, Wikidata-anchored entity resolution — named people/orgs → one node across episodes (OpenAI→`wd-Q21708200`).
- Cross-episode speaker unification (Tristan Harris: 5 appearances/3 shows → 1 node).
- Cross-claim `disputes`/`supersedes` derivation; consensus-over-time aggregation primitive.
- Constrained retrieval surface (MCP) + a 4-arm eval harness.

**Not yet built (the real next work, in priority order):**
1. **Serving layer:** columnar (ClickHouse) for claims, ANN index (FAISS/HNSW) for blocking+retrieval, the graph DB for identity, and the **materialized views + dirty-set freshness** (§6). *This is the biggest gap — the PoC ran live/in-memory.*
2. **Concept resolution** (fragments today) + extractor compound-suppression.
3. **Citation precision** (word-level timestamps; the hosted-ASR path gave coarse spans) and **verbatim quote spans**.
4. **Cross-lingual concept linking + per-language pipelines** (PoC is English-only).
5. **Voiceprint speaker gallery** at scale (PoC used name-based identity).

The interfaces are designed so each of these is a swap behind a stable seam, not a rewrite. (PoC code is the reference implementation of the seams.)

---

## 13. Known hard problems & failure modes (don't get surprised)

- **Resolution recall is the scaling risk**, not graph size. Under-merge → fragmentation (AGI→14); over-merge → distinct things collapse (worse — silent and corrupting). Tune per-type; never transitive-closure; review the ambiguous middle.
- **Extraction emits compounds** ("X and Y", "X's approach") as entities → fragmentation. Fix at the prompt (base entity + relation), not downstream.
- **View staleness** if dirty-set propagation lags ingestion. Make freshness observable (watermarks).
- **"Fact-check" liability.** Never ship "verified/false." Ship "contrasting claims / disputed by / superseded." You show the dialogue; you don't adjudicate truth.
- **The fixture-vs-real gap.** Offline tests prove *units*; every real integration seam (Wikidata redirects/rate-limits, async vs sync clients, transcript-label-is-name) had a hole the fakes couldn't show. Test the live wiring, not just the logic.

---

## 14. Suggested build sequence

1. **Persistent stores behind the seams:** ClickHouse (claims) + a graph DB (identity) + an ANN index (blocking/retrieval), each behind the PoC's existing interfaces.
2. **Incremental resolution at scale:** ANN blocking + the cascade against the persistent canonical store; concept-resolution tuning + extractor compound-suppression.
3. **Materialized views + dirty-set freshness:** the five views (§6), refreshed by dirty-set on ingest. This unlocks the product page.
4. **Constrained retrieval API** (§7) for the interactive "ask" surface, with the no-unconstrained-search guardrail.
5. **Citation precision:** word-level ASR/timestamps + verbatim spans.
6. **Multilingual:** per-language extraction, multilingual embeddings already in place, cross-lingual concept-linking pass; the cross-cultural-diff product on top.
7. **Scale ingest:** worker fleets for extract/resolve; Podcast Index firehose; backfill financed by serving revenue.

---

*Build the resolution layer like it's the company, because it is. Everything else — transcription, extraction, storage, serving — is plumbing with known answers. The resolved, bitemporal, cross-lingual identity of who-said-what-about-whom-and-when is the asset that compounds and can't be cloned.*
