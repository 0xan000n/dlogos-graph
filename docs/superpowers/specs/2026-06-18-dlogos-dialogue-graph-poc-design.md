# dLogos Dialogue Graph — PoC Design Spec

**Date:** 2026-06-18
**Status:** Draft for review (revised 2026-06-19 with corrected decisions — see the change log at the end of §14)
**Scope:** Proof-of-concept for the dLogos dialogue intelligence layer (temporal knowledge graph built from podcast conversation), over a curated corpus of ~200 podcasts in the knowledge-economy domains.

---

## 1. One-liner

A **generalized, temporal, dialogue knowledge graph** built from the top ~200 podcasts — speakers, entities, and time-stamped, stance-tagged claims, queryable for *who said what, when, and how the consensus moved* — proven by a head-to-head demo where a frontier model **plus** dLogos answers temporal/consensus questions that the same frontier model **alone** cannot.

This is the infrastructure layer described in the dLogos positioning work: the live dialogue signal a model calls when a question touches the present. The PoC validates the construction pipeline and the temporal retrieval quality on a tractable, high-signal corpus.

---

## 2. Goal & success criteria

### Primary success criterion (the greenlight artifact)

A **four-arm head-to-head demo** across **12–15 hero queries** spanning the target domains, where each query is asked of:

1. **Model alone** — the frontier model with no tools. The *floor*.
2. **Model + web search** — the frontier model with live web search. This is **the Perplexity bar**: it neutralizes naive "recency" and naive "provenance" advantages (the web also has fresh, citable text), forcing dLogos to win on *structure* rather than on freshness alone.
3. **Model + naive vector-RAG** — the frontier model over a dumb vector index built from the **same ~200-podcast transcripts**. This isolates the question that matters: does the **graph / temporal / stance structure** beat *dumb retrieval over identical data*? If arm 4 only beats arm 3 by a hair, the structure isn't earning its keep.
4. **Model + dLogos** — the frontier model + the dLogos temporal graph via the MCP tool.

Arms 1 and 4 are the headline; arms 2 and 3 are the credibility controls that pre-empt the two obvious "you didn't need a graph for that" rebuttals.

The PoC succeeds if dLogos wins specifically on **temporal-consensus synthesis across attributed sources** — the dimension that survives a web-search and a vector-RAG competitor:

- **Temporal-consensus synthesis** (the elevated dimension) — correctly characterizes *how a position moved over time across multiple attributed speakers*, not a static or single-source summary.
- **Attribution precision** — names the speaker and returns a citation (episode + timestamp) whose **speaker actually checks out at that timestamp** when you listen (see the speaker-verified citation check, §9).
- **Provenance integrity** — every assertion traces to a source span, not a confident hallucination.
- **Recency** and **couldn't-have-known** are **demoted**: a web-search arm can also be recent, so a recency-only win proves little. They remain *recorded* but are weighted down in the rubric (see §9).

### Generalization framing (existence proof, not generalization proof)

The win must hold across a **spread** of domains and query archetypes — but 15 queries cannot cover 8 domains × 5 archetypes (40 cells). The honest claim is therefore an **existence proof across a spread**, not a generalization proof. The corpus and extraction are domain-general by design; the eval samples the space to show the win is not a single cherry-picked topic (see §9).

### Secondary (byproduct, measured not optimized)

- **Cost-per-episode** for ASR + extraction on the open-weight stack, to extrapolate firehose economics.
- **Throughput** of the backfill, as a scaling data point.

### Explicit non-goals for the PoC

Not proving: multilingual coverage, sub-minute real-time freshness, the full entity-resolution cascade at 100M+ entities, the analytical (ClickHouse) serving path, the API/CLI surfaces, or self-hosted GPU economics. All deferred with rationale in §13.

---

## 3. Scope

### In scope

- Curated ~200 English-language podcasts across: **Science, Technology, Philosophy, Engineering, Product, Business, Finance, Politics**.
- **Two-tier backfill window:** a **~6-month broad backfill across all ~200 shows**, **plus** an **~18–24 month deeper backfill for a high-velocity subset (~15–25 shows)** that carries the temporal-shift demos. (The deep subset is where "how did the consensus move" has enough signal; the broad tier proves domain spread.)
- Ingestion → ASR + diarization + cross-episode speaker identity (host-anchored) → open-weight extraction of stance-tagged claims → **lightweight resolution** → bitemporal graph load → hybrid temporal retrieval.
- **Lightweight resolution (in scope, not deferred):** (i) **subject-entity embedding clustering** so "consensus about X" is not fragmented across *Apple* / *iPhone* / *Apple hardware*; (ii) **recurring-guest resolution** via episode metadata + the "my guest today is…" intro + a **Wikidata** match (guests, not just hosts, are belief-tracking subjects); (iii) a **controlled predicate vocabulary enforced at extraction time** (a closed enum in the extraction schema — *not* a separate post-hoc normalization pass). The full ER cascade at scale remains deferred (§13); this is the deliberately small slice that the temporal-consensus claim depends on.
- Two serving surfaces: **MCP server + a thin query/side-by-side UI**.
- **Four-arm** evaluation harness producing the head-to-head artifact (model-alone / +web-search / +naive-vector-RAG / +dLogos).

### Out of scope (deferred — see §13)

- Multilingual / translation.
- Full ER cascade **at scale** (embedding-blocking → rules/ML/LLM matcher → Louvain merge → Wikidata canonicalization at 100M+ entities). The *lightweight* subset above is in scope; only the at-scale cascade is deferred.
- ClickHouse analytical mirror for OLAP "consensus-over-time" aggregation.
- Self-hosted GPU fleet (PoC uses rented inference).
- Public API and CLI surfaces (the thin demo UI is in scope; the productized API/CLI are not).
- Sub-minute real-time; full historical (multi-year) backfill.

---

## 4. Corpus & curation

- **Selection:** ~200 shows drawn from **public charts** — Apple/Spotify category charts plus Podcast Index — ranked by chart position within each of the eight domains, with a light manual pass to drop low-quality/off-domain feeds. Chart-driven (not hand-built) so the corpus is reproducible and defensible as "the top shows," and so coverage spans the domains rather than a personal pick.
- **Backfill window — two tiers:**
  - **(a) Broad tier:** **~6 months across all ~200 shows.** Establishes domain spread and gives the demo a wide, recent base.
  - **(b) Deep tier:** **~18–24 months for a high-velocity subset of ~15–25 shows.** This is where the temporal-shift / consensus-movement demos live — short windows don't contain enough re-visits of a topic by the same speaker to show a position *moving*. The subset is chosen for high publish cadence and topic recurrence within the eight domains.
- **Language:** English-first (the curated tier is predominantly English). Multilingual deferred.
- **Corpus manifest:** a versioned config file, one row per show: `feed_url`, canonical `show_id`, `domain` tag(s), `known_hosts` (names + any available reference audio for voiceprint seeding), and a **`deep_backfill` flag** marking the ~15–25 high-velocity subset. This manifest is an input artifact, reviewed before backfill.

---

## 5. Architecture overview

Event-driven pipeline; same skeleton for backfill (one-time fan-out) and incremental (poll new RSS items).

```
RSS feeds (Podcast Index GUIDs)
      │  fetch enclosure, content-hash dedupe, idempotent on episode GUID
      ▼
Object storage (raw audio, cold)  ──► Work queue
      ▼
ASR + Diarization + Alignment        (WhisperX / whisper-large-v3 + pyannote)
      │  speaker-labeled, word-timestamped transcript
      ▼
Cross-episode speaker identity       (voice embeddings, host-anchored gallery)
      ▼
Extraction (open-weight LLM)         (DeepSeek / Gemma / Kimi, self-run on rented GPU for backfill)
      │  domain-general ontology → stance-tagged Claims with source spans
      │  predicate drawn from a CONTROLLED VOCABULARY (closed enum, enforced here)
      ▼
Lightweight resolution               (our module — batch, BEFORE graph load)
      │  subject-entity embedding clustering (Apple≈iPhone≈Apple-hardware)
      │  recurring-guest resolution (metadata + "my guest today is…" + Wikidata)
      ▼
Temporal graph load (Graphiti)       (bitemporal edges, reified Claim nodes)
      │  BULK ingest BYPASSES Graphiti per-add LLM node-dedup — we resolved already
      ▼
Hybrid retrieval (Graphiti)          (semantic + BM25 + graph traversal + RRF rerank)
      │                               + consensus helper (stance over time buckets)
      ▼
MCP server + thin UI  ──►  Claude (head-to-head demo)
```

**Build-vs-adopt stance:** adopt **Graphiti** as the bitemporal store + multi-backend abstraction + incremental-invalidation logic + MCP server. Build on top: the podcast ingestion front-end, the domain-general dialogue ontology, the **open-weight extraction stage**, and host-anchored speaker identity. The temporal plumbing is a dependency; the resolved, time-stamped, dialogue-specific graph is the asset.

---

## 6. Data model

### Node types (domain-general ontology)

- **Speaker** — a resolved human (host **or recurring guest**). Carries voiceprint reference(s), names/aliases, per-show host flags, and an optional **Wikidata QID** for recurring guests. Guests are first-class belief-tracking subjects, not just hosts — recurring-guest resolution is in scope (§7.3).
- **Entity** — Person / Organization / Concept-or-Topic / Work (book, paper, product, other episode). Free-form within these categories; **no fixed topic list**. Each entity carries a **`canonical_id`** filled by subject-entity resolution (§7.4a) so claims about the same real-world thing cluster under one node instead of fragmenting across surface forms (*Apple* / *iPhone* / *Apple hardware*).
- **Episode** — the source unit. Carries `show_id`, publish date, audio hash, transcript reference. Every derived fact traces back here ("answered in episode N").
- **Claim** — **reified as a node** (not just an edge), so it can carry stance, sentiment, confidence, and a source span, and can be contradicted/superseded later.

### Edge types

- `Speaker —asserts→ Claim`
- `Claim —about→ Entity` (subject of the claim)
- `Claim —mentions→ Entity`
- `Claim —agrees_with / disputes→ Claim` (cross-claim relations)
- `Claim —supersedes→ Claim` (same speaker, updated position)
- `Speaker —appears_in→ Episode`

### Claim record (extraction output schema)

```
Claim {
  speaker_ref        // resolved Speaker id (or per-episode speaker if unresolved)
  predicate          // CONTROLLED VOCABULARY — one of a closed enum of ~15
                     //   normalized predicates (expects, rates_positive,
                     //   rates_negative, predicts, recommends, criticizes,
                     //   compares, explains, attributes, forecasts, endorses,
                     //   rejects, questions, agrees, disagrees). Enforced AT
                     //   EXTRACTION TIME via the response schema — not a
                     //   separate post-hoc normalization pass.
  subject_entity     // resolved Entity (the thing the claim is about); carries
                     //   canonical_id from subject-entity clustering (§7.4a)
  object             // free text / entity / value
  stance             // enum: asserts | disputes | hedges | predicts | retracts
  sentiment          // signed scalar or enum
  confidence         // model confidence in the extraction
  source_span {
    episode_id, t_start, t_end, transcript_offset
  }
}
```

### Temporal model (bitemporal — the core, and the part naive designs get wrong)

- **Two independent time axes per fact:**
  - **event-time** — when it was said (episode publish/recording date; the claim's validity window).
  - **ingestion-time** — when we processed it (so backfilling a 2025 episode in 2026, or a re-release, is represented correctly).
- **Validity intervals on edges**, not snapshots. A fact that stops being true is **invalidated** (`t_invalid` set), never deleted — history is preserved without recomputation. **Do not** snapshot graph state at intervals (the documented anti-pattern that explodes node counts at scale).
- **Contradiction handling = invalidation, not recompute.** New conflicting knowledge updates/invalidates the prior edge using temporal metadata; the firehose can ingest continuously without nightly rebuilds.
- **Current-state fast path:** at PoC scale (≤ low-millions of claims) not required; noted as a scaling concern for later.

---

## 7. Component design

### 7.1 Ingestion

- Resolve feeds via **Podcast Index** for clean feed URLs and episode GUIDs.
- Fetch audio enclosure → content-hash → object storage (S3/R2/MinIO). **Idempotent on episode GUID** so re-polling never reprocesses.
- Model as a queue → workers (SQS / Cloudflare Queues / a Postgres job table is fine at PoC scale). Backfill is a bounded fan-out; incremental is a poller on the same path.

### 7.2 Transcription + diarization + alignment

- **WhisperX** (whisper-large-v3) for ASR with word-level timestamps; **pyannote** for diarization; VAD pre-pass; map tokens → speaker turns by timing. (NVIDIA Parakeet/Canary are alternatives if quality/cost favors them — decide during the 10-show slice.)
- Output: speaker-labeled (`SPEAKER_00…`), word-timestamped transcript + language tag.
- Drop speakers below a small talk-time threshold (corpus-scale precedent: SPoRC dropped < 5%).

### 7.3 Cross-episode speaker identity (hosts + recurring guests)

- Diarization only gives *within-episode* labels. Resolve `SPEAKER_xx` → canonical **Speaker** using **voice embeddings** matched against a **host-anchored gallery** seeded from the corpus manifest's `known_hosts`.
- **Recurring-guest resolution is in scope** (not deferred): guests are belief-tracking subjects too, so resolving a guest who appears across several episodes is what makes "what does *[guest]* believe about X, and has it changed?" answerable. Three cheap signals, combined:
  1. **Episode metadata** — RSS/show-notes author/guest fields where present.
  2. **The intro pattern** — parse the host's *"my guest today is …"* utterance from the diarized transcript to get the spoken guest name, tied to the guest's diarization label for that episode.
  3. **A Wikidata match** — canonicalize the parsed name to a **Wikidata QID** (disambiguated by domain context) so the same person merges across shows and gets a stable id.
- Hosts resolve nearly for free; recurring guests resolve via the above; one-off unknown guests remain per-episode speakers (acceptable for the PoC).
- This is the voiceprint-and-name analogue of subject-entity resolution (§7.4a); both are intentionally kept lightweight at 200-show scale, with the full ER cascade deferred (§13). Diarization error here is the **top correctness risk** — see §7.6 / §11 — because a swapped label produces a *confident misattribution*, the worst failure mode.

### 7.4 Extraction (open-weight — your stage, first-class)

- **Backfill extraction — a self-run open-weight model** (candidates: DeepSeek-V3, Gemma 2/3 27B, Kimi K2): weights you control, served via vLLM/SGLang on **rented GPU** (Modal / RunPod / equivalent) and **batched** for bulk economy. Owning the bulk run beats per-token API at backfill volume. (Rented GPU, not an owned fleet — the standing fleet stays deferred.)
- **Incremental extraction going forward — a cheap economical model** (a smaller open-weight model or a low-cost hosted endpoint), since per-episode volume is low once the backfill is done.
- **Chunk with overlap** to fit context and not lose claims spanning chunk boundaries; **carry speaker labels into the prompt** so claims attribute to the right person.
- Emit the Claim schema (§6) conforming to Graphiti's custom entity/edge types. Stance is mandatory — it is what makes "how did belief shift" queryable.
- **Controlled predicate vocabulary, enforced here.** The `predicate` field is constrained to a **closed enum of ~15 normalized predicates** *in the extraction response schema itself* (structured output / JSON-schema-constrained decoding). This is a deliberate design choice: normalize predicates **at extraction time, not in a separate post-hoc pass**. It keeps the graph's relation vocabulary tight (consensus aggregation depends on it) without a second LLM stage.
- **Embeddings:** an open model (default **BGE-M3** — strong, open, multilingual-ready for later phases) for semantic indexing of claims/utterances **and** as the vector space for subject-entity clustering (§7.4a).

### 7.4a Lightweight resolution (in scope — runs after extraction, before graph load)

This is a small, deliberate slice of entity resolution that the temporal-consensus claim depends on. It runs as **our own batch module** over the full extracted-claim set *before* the graph load (so the graph load can bypass Graphiti's per-add dedup — §7.5/§7.6).

- **(i) Subject-entity embedding clustering.** Embed each claim's `subject_entity` surface form with BGE-M3; cluster near-duplicate surface forms (e.g. *Apple* / *iPhone* / *Apple hardware*) and assign a shared **`canonical_id`**. Without this, "consensus about X" fragments across surface forms and the consensus helper undercounts. Clustering thresholds are tuned on the slice; ambiguous merges are conservative (prefer leaving separate over wrongly merging two distinct entities).
- **(ii) Recurring-guest resolution.** As in §7.3 — metadata + the "my guest today is…" intro + Wikidata QID — producing stable Speaker ids for guests across shows.
- **(iii) Controlled predicate vocabulary.** Enforced upstream at extraction time (§7.4); listed here for completeness as the third resolution lever. There is no separate predicate-normalization pass.

### 7.5 Graph load

- **Graphiti on Neo4j** (native vector + full-text indexes; best-supported backend). Kuzu (embedded, zero-ops) and FalkorDB (sparse-matrix traversal at scale) are fallbacks if ops/latency dictate.
- Load resolved entities + reified Claims + bitemporal edges. Lean on Graphiti's incremental-update and edge-invalidation logic.
- **Bulk backfill BYPASSES Graphiti's per-add LLM node-dedup.** Graphiti's default `add_episode` runs an LLM call per addition to dedupe/resolve nodes — at backfill volume that is both slow and a large hidden $/episode line item, and it would *re-do* resolution we already did better in §7.4a. The backfill loader therefore uses Graphiti's **bulk path** with our pre-resolved `canonical_id`s, **disabling the per-add LLM dedup** (bulk ingest / `add_triplet`-style load with resolution already applied). Incremental (low-volume, going-forward) episodes may still use the per-add path. The loader exposes this as an explicit flag so the spike can measure both.

### 7.6 The Graphiti × open-weight extraction seam (the #1 risk — resolve first)

Graphiti's built-in extraction prompts are tuned for GPT-class models; structured-output reliability wobbles on open weights. Two candidate integration shapes:

- **(a)** Graphiti extracts, pointed at an OpenAI-compatible open-weight endpoint.
- **(b)** Our own extraction stage produces pre-formed triples; Graphiti is used as **store + temporal manager + retrieval** (via custom types / `add_triplet` / bulk ingest), **not** as the extractor.

In both shapes the **bulk backfill must bypass Graphiti's per-add LLM node-dedup** (§7.5): we run batch resolution ourselves in §7.4a *before* loading, so the per-add LLM resolution call is redundant cost. Shape (b) makes this natural (we already hold pre-formed, resolved triples); shape (a) must be configured to disable per-add dedup on the bulk path.

**Decision rule:** a 1–2 day **spike on ~20 episodes** compares (a) vs (b) on **three** axes, not one:

1. **Stance-tagged claim quality** — are claims correctly attributed, stanced, and sourced?
2. **Backfill throughput** — episodes/hour through the load path with per-add LLM dedup disabled.
3. **$/episode** — the all-in cost (extraction + load), the firehose-economics byproduct.

Commit to the winner before scaling — and the spike must clear a **throughput / $ gate**, not only a quality bar: a shape that produces good claims but is too slow or too expensive at the per-add-dedup default does not pass. Working assumption is (b) — it keeps open-weight extraction swappable and first-class and trivially supports the dedup bypass — but the spike decides.

---

## 8. Retrieval & serving surface

- **Hybrid retrieval** (Graphiti): cosine semantic similarity + BM25 full-text + breadth-first graph traversal, fused with **RRF** reranking (optional cross-encoder, e.g. a BGE reranker), with **temporal filters** on validity windows. Semantic finds topically relevant; BM25 catches exact names/terms; traversal pulls the connected neighborhood.
- **Consensus helper:** a query that buckets stance-tagged claims for a subject over time windows — the primitive behind "how has the consensus on X moved." It buckets by the resolved **`canonical_id`** (from §7.4a), so claims about *Apple* / *iPhone* / *Apple hardware* aggregate into one subject rather than fragmenting and undercounting the consensus. At PoC scale this runs fine on Neo4j; at firehose scale it is the workload that motivates the deferred columnar mirror (§13).
- **Surfaces:** **MCP server** (Graphiti ships one), so the corpus is queryable from Claude on day one, **plus a thin query/side-by-side UI** — a minimal web view that runs a query across the **four arms** (model-alone / +web-search / +naive-vector-RAG / +dLogos) and shows the answers next to each other, with dLogos's citations surfaced for inspection. This is the greenlight artifact made interactive; it is *not* the productized public API/CLI (those stay deferred).

---

## 9. Evaluation harness (the actual deliverable)

### Golden query set

**12–15 hero queries**, spread across the eight domains and across **query archetypes**. The deep-tier subset (§4 window-(b)) supplies the temporal-shift queries, since only the ~18–24 month window contains enough re-visits to show a position *moving*. The archetypes:

- **Temporal/consensus:** "Who's been discussing *X* in the last *N* months, and how has the framing shifted?"
- **Per-speaker belief tracking:** "What does *[person — host or recurring guest]* believe about *X*, and has their position changed?"
- **Contradiction:** "Who disagrees with *[claim]*, and what do they argue instead?"
- **Consensus vs. outlier:** "What's the emerging consensus on *X*, and who's the contrarian?"
- **Provenance:** "Where was *X* discussed?" → episode + timestamp.

Each query has a documented **good-answer shape** (expected speakers, stance, timeframe, citations), and these shapes are **pre-registered before any arm output is seen** (below).

**Framing — existence proof, not generalization proof.** 15 queries cannot cover 8 domains × 5 archetypes (40 cells). The artifact is therefore an **existence proof across a spread** — the win shows up across several domains and archetypes — not a statistical generalization claim. We state this explicitly so the result is not over-sold.

### Runner — FOUR arms

For each query, produce four answers from the **same strong frontier model** (the one you run in Claude — deliberately strong; beating a weak baseline proves nothing; distinct from the open-weight *extraction* model of §7.4):

1. **Model alone** — no tools. The floor.
2. **Model + web search** — live web search enabled. **The Perplexity bar:** neutralizes naive recency/provenance, since the web is also fresh and citable. Forces dLogos to win on *structure*.
3. **Model + naive vector-RAG** — a dumb top-k vector index over the **same ~200-podcast transcripts**. **Isolates** whether the graph/temporal/stance *structure* beats *dumb retrieval over identical data*. This is the arm that proves the graph earns its keep.
4. **Model + dLogos** — the temporal graph via the MCP tool.

Capture verbatim outputs and full tool traces for every arm.

### Scoring (rubric reweighted)

Hand-scored (≈ 48–60 answers across four arms — tractable at this size) on a rubric that **elevates temporal-consensus synthesis across attributed sources** and **demotes recency / couldn't-have-known** (because the web-search arm can also be recent, so a recency-only win is not a structural win):

| Dimension | Weight | What it measures |
|---|---|---|
| **Temporal-consensus synthesis** | **Elevated** | Correctly characterized how a position moved over time **across multiple attributed speakers** — the dimension that survives web-search and vector-RAG competitors. |
| **Attribution precision (speaker-verified)** | **High** | Correct speaker, and the citation's **speaker actually checks out at the timestamp** — verified by *who is speaking*, not merely topic presence (see speaker-verified citation check below). |
| Provenance integrity | High | Sourced to a real span, not a confident hallucination. |
| Recency | **Demoted** | Surfaced post-cutoff claims — recorded, but weighted down since the web-search arm also can. |
| Couldn't-have-known | **Demoted** | Structurally unavailable to frozen weights — recorded, weighted down for the same reason. |

#### Speaker-verified citation check

A citation (episode + timestamp) passes **only if the person speaking at that timestamp is the person the answer attributes the claim to** — not merely that the topic appears near that timestamp. This directly targets the §11 top risk: diarization error → *confident misattribution*. The check reads the diarized transcript at the cited `t_start`/`t_end` and confirms the speaker id matches.

### Eval credibility controls (so the artifact is believable, not self-graded)

- **Blind scoring:** the scorer is **blinded to arm identity** — the four answers per query are shuffled and de-labeled so the grader cannot reward "the dLogos one" reflexively.
- **Second independent rater on a subset:** a second rater scores an overlapping subset; **inter-rater agreement is reported** (e.g. Cohen's κ on the shared items).
- **Pre-registered good-answer shapes:** the expected speakers/stance/timeframe/citations for each query are written down and frozen **before** any arm output is generated, so scoring is against a fixed target, not a moving one.
- **Adversarial validation slice:** the slice used to validate ASR/diarization/attribution before scaling is deliberately **adversarial** — it includes a **panel show** (many overlapping speakers), a **remote-heavy interview** (variable audio quality), and an **ad-saturated show** (host-read ads that confuse diarization). If attribution survives these, it survives the corpus.

### Output

The committed **four-arm** side-by-side artifact (queries × four answers × blinded scores × inter-rater agreement) + the cost/throughput byproduct numbers (§7.6 spike). This is what greenlights building the company.

---

## 10. Key technical decisions & rationale

| Decision | Choice | Rationale |
|---|---|---|
| Temporal engine | Adopt **Graphiti** (don't build) | Bitemporal model + multi-backend + incremental invalidation + MCP server already productized. |
| Extraction model | **Self-run open-weight** for backfill → **cheap economical** model for incremental | Own the bulk backfill run for cost control; cheap hosted model for low-volume ongoing. Kills frontier per-token extraction cost — the economic thesis. |
| Extraction ownership | Our stage feeding Graphiti (pending spike) | Keeps open-weight extraction first-class and swappable; avoids fighting GPT-tuned prompts. |
| Graph backend | **Neo4j** (Kuzu/FalkorDB fallback) | Best Graphiti support; native vector + full-text. |
| ASR | **WhisperX + pyannote** | Production-standard; word timestamps + diarization; corpus-scale precedent (SPoRC). |
| Speaker identity | Host-anchored voiceprints | Hosts recur and matter most; cheap high-value "who said what." |
| Ontology | **Domain-general**, Claim reified | Works for any subject in the 8 domains; stance/sentiment/confidence need a node, not an edge. |
| Topic scope | **None (generalized)** | Per decision: prove a general dialogue layer, not pre-chosen spine topics. |
| Lightweight resolution | **In scope** — subject-entity clustering + recurring-guest (incl. Wikidata) + controlled predicate enum | Consensus must not fragment across surface forms; guests are belief subjects; predicate vocab kept tight at extraction time. Full ER cascade still deferred. |
| Predicate vocabulary | **Controlled, enforced at extraction time** | Closed enum in the response schema; normalize once, not in a second post-hoc pass. |
| Backfill window | **6-mo broad (all ~200) + 18–24-mo deep (~15–25 high-velocity)** | Broad tier proves domain spread; deep tier carries the temporal-shift demos. |
| Eval arms | **Four** (alone / +web-search / +naive-vector-RAG / +dLogos) | +web-search neutralizes recency; +vector-RAG isolates structure-vs-dumb-retrieval on identical data. |
| Graph bulk load | **Bypass Graphiti per-add LLM dedup; batch-resolve ourselves first** | Per-add LLM dedup is slow and a hidden $/episode cost; we already resolved in §7.4a. |
| Eval credibility | **Blind scorer + 2nd rater on a subset + pre-registered answer shapes + existence-proof framing** | Self-graded demos aren't believable; 15 queries can't generalize over 40 cells. |
| Head-to-head baseline | **Strong frontier model** (the one you run), on all four arms | A weak baseline proves nothing; the demo must beat a top model — even one with web search or naive RAG. |
| Surface | **MCP + thin query UI** | Query from Claude immediately; UI makes the head-to-head demo interactive. Public API/CLI deferred. |

---

## 11. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Diarization → confident MISATTRIBUTION** | **High (top risk)** | The worst failure: a swapped speaker label yields a *confident, sourced* claim attributed to the wrong person. Mitigations: host-anchored gallery + recurring-guest resolution (§7.3); validate on an **adversarial slice** (panel show, remote-heavy interview, ad-saturated show) before scaling; the eval's **speaker-verified citation check** (§9) fails any citation where the person speaking at the timestamp ≠ the attributed speaker; drop low-talk-time speakers. |
| Graphiti × open-weight extraction reliability | High | Spike (a) vs (b) on 20 episodes **before** scaling; assume (b). Spike clears a throughput/$ gate, not only quality (§7.6). |
| Extraction hallucinates claims / bad citations | High | Mandatory source spans; eval rubric's speaker-verified citation check; spot-QA on the slice. |
| Subject-entity fragmentation (consensus undercount) | Medium | Subject-entity embedding clustering assigns a shared `canonical_id` (§7.4a) so consensus aggregates by entity, not surface form; conservative merge thresholds tuned on the slice. |
| Recurring-guest identity unresolved | Medium (PoC) | In scope now: metadata + "my guest today is…" intro + Wikidata QID (§7.3). One-off unknown guests still fall back to per-episode speakers; full ER cascade deferred. |
| Stance extraction too noisy for "consensus shift" | Medium | Tight stance enum + controlled predicate enum; QA on slice; consensus helper aggregates to smooth noise. |
| Backfill throughput / cost (bulk load) | Medium | Bypass Graphiti per-add LLM dedup; batch-resolve ourselves (§7.5); spike measures episodes/hour **and** $/episode (§7.6) before the full run. |
| Cost overrun on backfill | Low | Rented inference; bounded windows (6-mo broad / 18–24-mo deep subset); instrument $/episode from the slice and extrapolate before full run. |
| Neo4j struggles on consensus aggregation | Low (PoC) / High (later) | Fine at PoC scale; documented seam → ClickHouse mirror at firehose scale (§13). |

---

## 12. Build sequence (de-risk first)

1. **Spike** — resolve the Graphiti × open-weight extraction seam on ~20 episodes, **with Graphiti's per-add LLM dedup disabled** on the bulk path. *Gate: pick (a) or (b) AND clear the throughput / $-per-episode bar (§7.6), not only claim quality.*
2. **Slice (adversarial)** — ingest + ASR + diarization + speaker identity on a slice that **includes a panel show, a remote-heavy interview, and an ad-saturated show**; eyeball transcript/diarization/attribution quality, especially the failure mode of confident misattribution. *Gate: ASR + speaker-attribution quality acceptable on the adversarial cases.*
3. **Extract + resolve + load** — run extraction (controlled predicate enum) + **lightweight resolution** (subject-entity clustering + recurring-guest) + graph load (bulk, dedup bypassed) on the slice; QA claim/stance quality and **speaker-verified** citation accuracy. *Gate: claims are sourced, stance-correct, and citations pass the speaker check.*
4. **Scale** — full backfill: **6-month broad across all ~200 shows + 18–24-month deep for the ~15–25 high-velocity subset**; capture cost + throughput.
5. **Retrieve** — wire hybrid retrieval + temporal filters + consensus helper (bucketed by `canonical_id`) + MCP server.
6. **Eval** — build the golden set with **pre-registered good-answer shapes**, run the **four-arm** head-to-head (alone / +web-search / +naive-vector-RAG / +dLogos), **blind-score** with a **second rater on a subset** (report agreement), produce the artifact.

Each gate can send us back a step; nothing scales past a failing gate.

---

## 13. Future phases / documented seams (not built now)

- **Entity resolution at scale** — embedding blocking → cascade matcher (rules → ML → LLM) → Louvain merge (not transitive closure) → Wikidata canonicalization. Required past ~100M entities and for multilingual variant collapsing. **Note:** the *lightweight* subset (subject-entity clustering, recurring-guest resolution incl. Wikidata, controlled predicate vocab) is **in scope now** (§7.4a); only the full at-scale cascade is deferred here.
- **Analytical read path** — mirror resolved Claim facts into **ClickHouse** for OLAP "consensus over time, by speaker, by topic" at firehose scale. Two read models (graph for retrieval/memory, columnar for analytics) over one resolved write model.
- **Multilingual** — language ID + canonical-form resolution across languages; BGE-M3 embeddings already chosen with this in mind.
- **Real-time** — WebSub/push + low-latency poll so new episodes are queryable within minutes of RSS publish.
- **Additional surfaces** — foundation-model tool/connector, public API, CLI; metering at the query-engine boundary so all surfaces meter uniformly.
- **Inference scaling** — the PoC already self-runs open-weight extraction for the backfill (batched on rented GPU); the forward path moves incremental extraction to a cheap economical model and, as volume grows, to **owned steady-state GPU serving** (vLLM/SGLang) — the standing fleet that's deferred now.
- **Legal / rights / ToS — a named future-phase seam (the PoC redistributes nothing).** Transcribing, storing, and redistributing *derived claims* from copyrighted podcast audio — especially where ASR or any pipeline step runs through a **commercial API** whose terms may restrict derivative use — is a real rights / terms-of-service / fair-use seam. The PoC sidesteps it by **redistributing nothing**: the corpus, transcripts, and extracted claims stay internal to the eval, and the only external artifact is the side-by-side scorecard. Before any productized surface (public API, hosted query) ships, this needs a deliberate rights/ToS/fair-use review — flagged here so it is not discovered late.

---

## 14. Resolved decisions (2026-06-18)

1. **Demo surface:** MCP server **+ a thin query/side-by-side UI**. (§8)
2. **Corpus:** curated from **public charts** (Apple/Spotify category charts + Podcast Index), chart-ranked within each domain, light manual quality pass — reproducible, not hand-built. (§4)
3. **Models — two distinct roles:**
   - *Pipeline extraction:* **self-run open-weight** model for the bulk backfill (weights you control, batched on rented GPU); **a cheap economical model** for incremental extraction going forward. (§7.4)
   - *Head-to-head baseline:* a **strong frontier model** (the one you run in Claude) on **all four arms** of the demo — strong on purpose, since beating a weak model proves nothing. (§9)
4. **Backfill window:** **two tiers — 6-month broad across all ~200 shows + 18–24-month deep for a ~15–25 high-velocity subset.** (§4)
5. **Eval:** **four arms** — model-alone / +web-search / +naive-vector-RAG over the same transcripts / +dLogos — scored on a rubric that **elevates temporal-consensus synthesis** and **demotes recency/couldn't-have-known**, with **blind scoring, a second rater on a subset, pre-registered answer shapes**, framed as an **existence proof**. (§9)
6. **Lightweight resolution is in scope:** subject-entity embedding clustering, recurring-guest resolution (metadata + intro + Wikidata), controlled predicate vocabulary enforced at extraction time. (§7.4a)
7. **Graph bulk load bypasses Graphiti's per-add LLM node-dedup;** we batch-resolve ourselves first, and the spike measures throughput + $/episode, not only quality. (§7.5/§7.6)

### Change log — corrected decisions (2026-06-19)

These supersede any stale text elsewhere; the body sections above have been edited to match.

- **Four eval arms** (added +web-search and +naive-vector-RAG) with a **reweighted rubric** (elevate temporal-consensus synthesis; demote recency / couldn't-have-known). §2, §9.
- **Lightweight resolution promoted from deferred into scope** (subject-entity clustering, recurring-guest incl. Wikidata, controlled predicate enum). §3, §6, §7.3, §7.4/§7.4a, §10, §11, §13.
- **Two-tier backfill window** (6-mo broad + 18–24-mo deep subset). §3, §4, §9, §11, §12.
- **Diarization raised to the top risk**, with an **adversarial validation slice** and a **speaker-verified citation check**. §7.3, §9, §11, §12.
- **Graphiti bulk per-add LLM dedup bypassed**; we resolve ourselves first; **spike gates on throughput + $/episode**. §7.5, §7.6, §11, §12.
- **Eval credibility**: blind scorer, second independent rater on a subset (agreement reported), pre-registered good-answer shapes, existence-proof framing. §9.
- **Legal / rights / ToS seam** named as a future phase (PoC redistributes nothing). §13.

### Still genuinely open (not blocking the plan)

- Exact chart sources and the popularity/quality cutoff for the final ~200, and which ~15–25 shows form the deep-backfill subset (resolved during the corpus-manifest task in build step 1–2).
- Which specific open-weight model wins the extraction spike (resolved by the spike itself).
```
