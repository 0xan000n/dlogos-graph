# Scale Machinery — Proper 20-Episode Knowledge Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the resolution/identity machinery that makes a multi-episode dialogue graph *correct* — the same company, person, and concept become **one** canonical node across episodes, with precise citations — and do it with scale-ready interfaces, validated on a 20-episode slice with a fragmentation metric.

**Architecture:** Three correctness layers added beneath the existing pipeline, each with a simple-now / scale-ready-later split. (1) A **persistent canonical entity store** + **incremental resolver** that anchors head entities to **Wikidata QIDs** and runs a **rules → embedding → LLM-adjudication cascade** against the accumulated canonical set (so episode N resolves against episodes 1..N-1, stably, re-runnably). (2) A **persistent canonical speaker store** so the same host/guest across episodes is one `speaker_id`. (3) **Word-level re-segmentation** so grounded citations snap to ~sentence spans. A **fragmentation report** turns "does it shatter?" into a number. At 20 episodes we use brute-force cosine and SQLite — the `candidates()` call is the ANN seam and the SQLite store is the columnar/graph-DB seam for later.

**Tech Stack:** Python 3.11, pydantic v2, numpy, **sqlite3** (stdlib — the persistent stores, no new dep), the existing DeepInfra OpenAI-compatible client (extraction model doubles as the LLM adjudicator), the existing `WikidataLinker` (`resolution/wikidata.py`), AssemblyAI word-level output. No FAISS / ClickHouse / Neo4j at this scale (YAGNI; seams documented).

**Scope boundaries (read before starting):**
- **20 episodes, not 20k.** Brute-force cosine in `candidates()` is fine for ~hundreds of entities. Do **not** add FAISS/HNSW now — but keep `candidates(embedding, type, k)` as the only similarity entry point so an ANN backend drops in behind it unchanged.
- **Speaker identity is name-driven, not voiceprint-driven, here.** The hosted AssemblyAI path gives diarization *labels* (A/B), not voice embeddings, and labels reset per episode — so cross-episode speaker identity keys on **names** (spoken intros + manifest hosts + guest metadata, canonicalized to QID where possible). Voiceprint-based host matching needs pyannote/WhisperX (GPU) and is the documented scale-path, **out of scope here**. Unnamed speakers stay per-episode (the existing fallback).
- **Back-compat:** every new collaborator is injected and defaulted off, so the existing 567-test suite stays green; the new behavior is opt-in via `PipelineDeps`.

---

## File Structure

**New files**
- `src/dlogos/resolution/canonical_store.py` — `CanonicalEntity` record; `CanonicalEntityStore` Protocol; `InMemoryCanonicalStore`; `SqliteCanonicalStore`. The persistent, ANN-ready entity index.
- `src/dlogos/resolution/cascade.py` — `match_entity(...)`: rules → embedding (per-type thresholds) → optional LLM adjudication. Pure decision logic over a store's candidates.
- `src/dlogos/resolution/incremental.py` — `IncrementalResolver`: resolve a claim batch's subject entities against the persistent store (Wikidata-anchored, cascade-matched), assign `canonical_id`, upsert. Drop-in for `resolve_subjects` (returns `SubjectResolution`). (Secondary/mention entities are out of scope — `ExtractedClaim` carries only `subject_entity` today.)
- `src/dlogos/speakers/speaker_store.py` — `CanonicalSpeakerStore` Protocol + `SqliteSpeakerStore`; `NameSpeakerResolver` that canonicalizes spoken/metadata names (→ QID where possible) to stable cross-episode `speaker_id`s.
- `src/dlogos/asr/word_segmentation.py` — `Word` re-segmentation: rebuild a `Transcript`'s segments from word-level timings (speaker-contiguous, sentence-bounded, duration-capped).
- `src/dlogos/eval/fragmentation.py` — `fragmentation_report(...)`: per-probe canonical-node counts (the resolution-quality metric).
- `scripts/run_kg_slice.py` — the 20-episode driver: ingest/transcribe (cached) → resegment → pipeline with the incremental resolver + persistent stores → export graph + fragmentation report.

**Modified files**
- `src/dlogos/schema.py` — add `qid: str | None` to `Entity`; add `Word` model + `words: list[Word]` to `Transcript`.
- `src/dlogos/graph/store.py` — add `wikidata_qid: str | None` to `EntityNode`.
- `src/dlogos/graph/loader.py` — carry the entity `qid` onto the `EntityNode` it builds (`loader.py:166`).
- `src/dlogos/asr/hosted_backend.py` — capture AssemblyAI `words` onto `Transcript.words`.
- `src/dlogos/resolution/wikidata.py` — add `anchor_entity(entity, linker) -> str | None` (QID for a person/org subject).
- `src/dlogos/pipeline.py` — add injectable `subject_resolver` and `speaker_resolver` hooks to `PipelineDeps`; optional `resegment_words` transform; default-off so existing tests are untouched.

---

## Phase 0 — Schema + word-level citation precision

*Gate: re-running the smoke on the cached transcript shows fine-grained segments and tight grounded spans (no more 3-minute blobs).* 

### Task 0.1: Add `qid` to the entity records

**Files:** Modify `src/dlogos/schema.py` (the `Entity` model), `src/dlogos/graph/store.py` (`EntityNode`); Test: `tests/test_schema.py`.

- [ ] **Step 1 — failing test** (`tests/test_schema.py`): assert `Entity(name="Apple", type=EntityType.organization, qid="Q312").qid == "Q312"` and that `qid` defaults to `None`.
- [ ] **Step 2 — run, expect FAIL** (`uv run pytest tests/test_schema.py -q`): unexpected keyword `qid`.
- [ ] **Step 3 — implement:** add `qid: str | None = None` to `Entity` (schema.py) and `wikidata_qid: str | None = None` to `EntityNode` (store.py). In `loader.py:166` set `EntityNode(..., wikidata_qid=claim.subject_entity.qid)`.
- [ ] **Step 4 — run, expect PASS**; also `uv run pytest -q` (whole suite green — `extra="forbid"` models need the field added, not breaking).
- [ ] **Step 5 — commit:** `feat(schema): qid on Entity/EntityNode for Wikidata-anchored resolution`.

### Task 0.2: Capture AssemblyAI word-level output

**Files:** Modify `src/dlogos/schema.py` (`Word` + `Transcript.words`), `src/dlogos/asr/hosted_backend.py`; Test: `tests/asr/test_hosted_backend.py`.

- [ ] **Step 1 — failing test:** extend the existing canned-AssemblyAI-JSON test so the completed payload includes a `words` array (`[{"text","start","end","speaker"}]`); assert the mapped `Transcript.words` has them with ms→s conversion and speaker preserved.
- [ ] **Step 2 — run, expect FAIL** (`Transcript` has no `words`).
- [ ] **Step 3 — implement:** add `class Word(BaseModel){text:str; t_start:float; t_end:float; speaker:str|None}` and `words: list[Word] = Field(default_factory=list)` on `Transcript` (schema.py). In `hosted_backend.AssemblyAIBackend._result_to_transcript`, map `result.get("words")` → `list[Word]` (reuse `_ms_to_s`), pass into the `Transcript`. Leave `segments` as-is.
- [ ] **Step 4 — run, expect PASS**; `uv run pytest -q` green.
- [ ] **Step 5 — commit:** `feat(asr): capture AssemblyAI word-level timings on Transcript`.

### Task 0.3: Word-level re-segmentation

**Files:** Create `src/dlogos/asr/word_segmentation.py`; Test: `tests/asr/test_word_segmentation.py`.

- [ ] **Step 1 — failing test:** build a `Transcript` whose `words` are: speaker A "Apple is great." then speaker B "I disagree strongly." then speaker A "Inflation is cooling." (with realistic word spans). Assert `resegment_by_words(transcript)` returns **≥3** segments split on speaker change + sentence end, each `TranscriptSegment` carrying the right speaker and a span equal to its words' [first.start, last.end], and `len(result.segments) > len(transcript.segments)` when the input had one coarse segment.
- [ ] **Step 2 — run, expect FAIL** (module missing).
- [ ] **Step 3 — implement** `resegment_by_words(transcript, *, max_seg_seconds=15.0) -> Transcript`:
  ```python
  def resegment_by_words(transcript, *, max_seg_seconds: float = 15.0) -> Transcript:
      if not transcript.words:
          return transcript  # nothing to refine; keep utterance segments
      segs, cur = [], []
      def flush():
          if cur:
              segs.append(TranscriptSegment(
                  speaker=cur[0].speaker or "A",
                  text=" ".join(w.text for w in cur).strip(),
                  t_start=cur[0].t_start, t_end=cur[-1].t_end))
      for w in transcript.words:
          if cur and (
              (w.speaker or "A") != (cur[0].speaker or "A")
              or (w.t_end - cur[0].t_start) > max_seg_seconds
              or cur[-1].text.endswith((".", "!", "?"))
          ):
              flush(); cur = []
          cur.append(w)
      flush()
      return transcript.model_copy(update={"segments": segs})
  ```
  Return the transcript unchanged when `words` is empty (so the WhisperX/mock paths are unaffected). Keep it pure (stdlib only).
- [ ] **Step 4 — run, expect PASS** (`uv run pytest tests/asr/test_word_segmentation.py -q`).
- [ ] **Step 5 — commit:** `feat(asr): word-level transcript re-segmentation for tight spans`.

### Task 0.4: Wire resegmentation into the pipeline (opt-in)

**Files:** Modify `src/dlogos/pipeline.py`; Test: `tests/test_grounding_pipeline.py`.

- [ ] **Step 1 — failing test:** a pipeline run with `PipelineDeps(..., resegment_words=True)` over a transcript carrying `words` produces claims whose grounded spans match the **fine** segments (≤ `max_seg_seconds`), not the coarse input segment.
- [ ] **Step 2 — run, expect FAIL** (`PipelineDeps` has no `resegment_words`).
- [ ] **Step 3 — implement:** add `resegment_words: bool = False` to `PipelineDeps`. In `Pipeline.run`, immediately after the talk-time prune (`pipeline.py:355`) and before speaker resolution, insert: `if self._deps.resegment_words: transcript = resegment_by_words(transcript)`. Import `from dlogos.asr.word_segmentation import resegment_by_words`.
- [ ] **Step 4 — run, expect PASS**; `uv run pytest -q` green.
- [ ] **Step 5 — commit:** `feat(pipeline): optional word-level re-segmentation before grounding`.

---

## Phase 1 — Persistent canonical entity store + incremental, Wikidata-anchored, cascade resolution

*Gate: a two-call test proves the SAME real-world entity gets the SAME `canonical_id` across separate `resolve` calls against a persistent store (the cross-episode fragmentation fix), and a distinct entity does not collide.*

### Task 1.1: Canonical entity store — record, Protocol, in-memory impl

**Files:** Create `src/dlogos/resolution/canonical_store.py`; Test: `tests/resolution/test_canonical_store.py`.

- [ ] **Step 1 — failing test:** upsert two `CanonicalEntity` (org "Apple" emb≈[1,0], org "OpenAI" emb≈[0,1]); `candidates([0.99,0.01], EntityType.organization, k=1)` returns Apple with score ≈1; `by_qid("Q312", organization)` returns it after a qid upsert; a `person` named "Apple" is never returned for an `organization` query (type partition).
- [ ] **Step 2 — run, expect FAIL** (module missing).
- [ ] **Step 3 — implement:**
  ```python
  class CanonicalEntity(BaseModel):
      canonical_id: str
      canonical_name: str
      entity_type: EntityType
      qid: str | None = None
      embedding: list[float] | None = None
      aliases: list[str] = Field(default_factory=list)

  @runtime_checkable
  class CanonicalEntityStore(Protocol):
      def get(self, canonical_id: str) -> CanonicalEntity | None: ...
      def by_qid(self, qid: str, entity_type: EntityType) -> CanonicalEntity | None: ...
      def by_exact_name(self, norm_name: str, entity_type: EntityType) -> CanonicalEntity | None: ...
      def candidates(self, embedding: list[float], entity_type: EntityType, k: int = 5
          ) -> list[tuple[CanonicalEntity, float]]: ...   # the ANN seam (brute-force now)
      def upsert(self, entity: CanonicalEntity) -> None: ...
      def all(self) -> list[CanonicalEntity]: ...
  ```
  `InMemoryCanonicalStore`: dict keyed by `canonical_id`; `candidates` = type-filtered brute-force cosine (reuse `_cosine` from `subjects.py` — import it) sorted desc, top-k; `by_qid`/`by_exact_name` linear scan (fine at scale here). Document at the top: "`candidates()` is the only similarity entry point; swap an ANN index (FAISS/HNSW) behind it for >100k entities — nothing else changes."
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(resolution): canonical entity store (in-memory, ANN-ready)`.

### Task 1.2: SQLite-backed canonical store (persistence across runs)

**Files:** Modify `src/dlogos/resolution/canonical_store.py`; Test: `tests/resolution/test_canonical_store.py`.

- [ ] **Step 1 — failing test:** open `SqliteCanonicalStore(tmp_path/"ent.db")`, upsert Apple; open a **second** `SqliteCanonicalStore` on the same path; assert `by_qid`/`candidates` still find Apple (persisted), and re-upserting the same `canonical_id` merges aliases rather than duplicating.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `SqliteCanonicalStore` via stdlib `sqlite3`: one table `entities(canonical_id PK, canonical_name, entity_type, qid, embedding_json, aliases_json)`; `candidates` loads the type's rows and brute-force cosines in Python (acceptable at this scale; the SQLite table is the seam where a vector extension / external store later lives). `upsert` is `INSERT ... ON CONFLICT(canonical_id) DO UPDATE` merging alias sets. Same Protocol as the in-memory store.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(resolution): sqlite-backed persistent canonical store`.

### Task 1.3: Wikidata subject anchoring

**Files:** Modify `src/dlogos/resolution/wikidata.py`; Test: `tests/resolution/test_wikidata.py`.

- [ ] **Step 1 — failing test:** with a fake Wikidata client returning Q312 for "Apple" (org), `anchor_entity(Entity(name="Apple", type=organization), linker)` returns `"Q312"`; for a `concept` it returns `None` without calling the client (concepts aren't QID-anchored here); an unknown name returns `None`.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `anchor_entity(entity, linker) -> str | None`: only attempt for `EntityType.person`/`organization`; delegate to the existing `WikidataLinker` (read `wikidata.py` for its method name/signature — reuse, don't reimplement); return the QID or `None`. Keep `httpx` lazy via the existing linker.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(resolution): Wikidata QID anchoring for person/org subjects`.

### Task 1.4: The cascade matcher

**Files:** Create `src/dlogos/resolution/cascade.py`; Test: `tests/resolution/test_cascade.py`.

- [ ] **Step 1 — failing tests (one per tier):**
  - *rules/exact:* a store with org "Apple"; matching an entity whose normalized name is "apple" returns that `canonical_id` with no embedding compare.
  - *rules/qid:* matching an entity carrying `qid="Q312"` returns the stored Q312 entity even though its surface form ("the iPhone maker") embeds far away.
  - *embedding-high:* score ≥ `thresholds[org].high` → match.
  - *embedding-low:* score < `thresholds[concept].low` → `NEW`.
  - *ambiguous + adjudicator:* score in the middle, injected `llm_adjudicator(a, b) -> bool` returns `True` → match; with **no** adjudicator, ambiguous → `NEW` (conservative).
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:**
  ```python
  @dataclass(frozen=True)
  class TypeThresholds: high: float; low: float
  DEFAULT_THRESHOLDS = {
      EntityType.organization: TypeThresholds(0.86, 0.62),
      EntityType.person:       TypeThresholds(0.86, 0.62),
      EntityType.concept:      TypeThresholds(0.90, 0.70),  # stricter: concepts paraphrase
      EntityType.work:         TypeThresholds(0.90, 0.70),
  }

  @dataclass
  class MatchDecision: canonical_id: str | None; reason: str   # None => create NEW

  def match_entity(entity, embedding, store, *, qid=None,
                   thresholds=DEFAULT_THRESHOLDS, llm_adjudicator=None) -> MatchDecision:
      # Tier 1 rules
      if qid:
          hit = store.by_qid(qid, entity.type)
          if hit: return MatchDecision(hit.canonical_id, "qid")
      exact = store.by_exact_name(_normalize_surface(entity.name), entity.type)
      if exact: return MatchDecision(exact.canonical_id, "exact-name")
      # Tier 2 embedding
      cands = store.candidates(embedding, entity.type, k=1)
      if not cands: return MatchDecision(None, "new-empty")
      top, score = cands[0]
      t = thresholds[entity.type]
      if score >= t.high: return MatchDecision(top.canonical_id, f"embed-{score:.2f}")
      if score < t.low:   return MatchDecision(None, f"new-{score:.2f}")
      # Tier 3 ambiguous middle -> LLM adjudication (only here; cheap)
      if llm_adjudicator and llm_adjudicator(entity, top):
          return MatchDecision(top.canonical_id, "llm-yes")
      return MatchDecision(None, "ambiguous-conservative")
  ```
  Import `_normalize_surface` from `subjects.py` (DRY). `llm_adjudicator` is an injected callable; the real one (Task 1.5) wraps the DeepInfra client with a yes/no prompt.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(resolution): rules→embedding→LLM cascade matcher`.

### Task 1.5: Incremental resolver (drop-in for `resolve_subjects`)

**Files:** Create `src/dlogos/resolution/incremental.py`; Test: `tests/resolution/test_incremental.py`.

- [ ] **Step 1 — failing tests (the cross-episode property — the whole point):**
  - *convergence:* with a persistent store + fake embedder where "Apple"/"Apple Inc." embed ≈identical, resolve a batch from "episode 1" with subject "Apple", then a **separate** resolve call from "episode 2" with subject "Apple Inc." → **both claims carry the same `canonical_id`** (via exact/embedding/qid), and the store holds **one** Apple canonical with both aliases.
  - *qid convergence:* "Apple" (qid Q312) and "the iPhone maker" (qid Q312, but embeds far) → same `canonical_id` across calls (anchor wins).
  - *no false merge:* "OpenAI" does not collapse into "Apple".
  - *drop-in shape:* returns a `SubjectResolution` (claims with `subject_entity.canonical_id` + `qid` stamped, `clusters`, `surface_to_id`) so the pipeline can call it exactly like `resolve_subjects`.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `IncrementalResolver(store, embedder, *, wikidata_linker=None, llm_adjudicator=None, thresholds=DEFAULT_THRESHOLDS)` with `resolve(claims) -> SubjectResolution`:
  1. Collapse to distinct `(type, normalized-name)`; embed each distinct display form (batch).
  2. For each distinct form: `qid = anchor_entity(entity, wikidata_linker) if wikidata_linker else None`; `decision = match_entity(entity, emb, store, qid=qid, thresholds=..., llm_adjudicator=...)`.
  3. If matched → reuse that `canonical_id`; else mint a new one: **`canonical_id = "wd-"+qid` when qid present, else** the deterministic `_canonical_id(name, type)` from `subjects.py` (stable, content-addressed). `store.upsert(CanonicalEntity(canonical_id, name, type, qid, embedding=emb, aliases=[name]))` (upsert merges aliases for matches).
  4. Build `surface_to_id`; stamp `canonical_id` **and** `qid` onto copies of each claim's `subject_entity`; assemble `clusters` from the store entries touched.
  Reuse `EntityCluster`/`SubjectResolution`/`_normalize_surface`/`_canonical_id`/`_embed_all` from `subjects.py` (import them — DRY).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(resolution): incremental Wikidata-anchored resolver against a persistent store`.

### Task 1.6: Real LLM adjudicator (DeepInfra, lazy)

**Files:** Modify `src/dlogos/resolution/cascade.py` (add a factory); Test: `tests/resolution/test_cascade.py`.

- [ ] **Step 1 — failing test:** `llm_adjudicator_from_client(fake_client)` returns a callable; given a fake client whose completion returns `{"same": true}` for ("Apple","Apple Inc.") the callable returns `True`; `{"same": false}` → `False`; a malformed completion → `False` (conservative).
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `llm_adjudicator_from_client(client, model)`: returns `def adj(entity, candidate) -> bool` that sends a tight yes/no JSON prompt ("Are these the same real-world {type}? A: {name+aliases}. B: {candidate name+aliases}. Answer JSON {\"same\": bool}.") via the injected OpenAI-compatible client (same shape as `extractor._call`), parses `same`, defaults `False` on any parse/HTTP error. Client injected; no network in tests.
- [ ] **Step 4 — run, expect PASS**; `uv run pytest -q` green.
- [ ] **Step 5 — commit:** `feat(resolution): DeepInfra LLM adjudicator for ambiguous entity pairs`.

---

## Phase 2 — Persistent cross-episode speaker identity (name-driven)

*Gate: the same host across two episodes (named in both intros) and a recurring guest both resolve to one stable `speaker_id`; an unnamed speaker stays per-episode.*

### Task 2.1: Canonical speaker store (sqlite)

**Files:** Create `src/dlogos/speakers/speaker_store.py`; Test: `tests/speakers/test_speaker_store.py`.

- [ ] **Step 1 — failing test:** `SqliteSpeakerStore(tmp/"spk.db").canonical_for(name="Darian Woods", qid=None)` mints a stable id; a second store instance on the same path returns the **same** id for the same name; `canonical_for(qid="Q123")` keys on QID regardless of surface name; normalization collapses "darian woods" / "Darian  Woods".
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `SqliteSpeakerStore` + `CanonicalSpeakerStore` Protocol: table `speakers(speaker_id PK, qid, norm_name, display_name)`; `canonical_for(name=None, qid=None) -> CanonicalSpeaker` resolves by QID first, then normalized name, minting `spk-wd-<QID>` or `spk-<sha1(norm_name)[:10]>` on first sight and persisting. Reuse `CanonicalSpeaker` from `speakers/identity.py` (read it for the field names).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(speakers): sqlite persistent canonical speaker store`.

### Task 2.2: Name-driven speaker resolver

**Files:** Create `NameSpeakerResolver` in `src/dlogos/speakers/speaker_store.py`; Test: `tests/speakers/test_speaker_store.py`.

- [ ] **Step 1 — failing test:** given a transcript + a `label->name` map (host "Darian Woods" on label A from intro/manifest; guest "Cardiff Garcia" on label B) and the store, `NameSpeakerResolver(store).resolve(transcript, label_names, qids={"Cardiff Garcia":"Q..."}) -> dict[label, SpeakerResolution]` returns stable ids for A and B; a label with no name → unresolved `SpeakerResolution`. A second episode reusing "Darian Woods" yields the **same** id.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `NameSpeakerResolver.resolve(transcript, label_names, *, qids=None) -> dict[str, SpeakerResolution]`: for each label with a name, `store.canonical_for(name, qid)` → `SpeakerResolution(label, resolved=CanonicalSpeaker(...), score=1.0)`; labels without a name → unresolved (the pipeline's fallback hook then gives them a per-episode id). Pure over the injected store.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(speakers): name-driven cross-episode speaker resolver`.

### Task 2.3: Name extraction from intros + manifest

**Files:** Create `extract_label_names(transcript, *, known_hosts) -> dict[str,str]` in `src/dlogos/speakers/speaker_store.py`; Test: `tests/speakers/test_speaker_store.py`.

- [ ] **Step 1 — failing test:** a transcript with "I'm Darian Woods" (label A) and "my guest today is Cardiff Garcia" (host label) + `known_hosts=["Darian Woods"]` → `{ "A": "Darian Woods" }` (self-intro) and the guest name surfaced for the inferred guest label. A transcript with no names → `{}`.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `extract_label_names`: reuse the guest-intro regexes already in `speakers/guests.py` (read it; DRY — import the patterns or a helper) plus a self-intro pattern (`I'm X` / `I am X` / `this is X`); map a detected name to the diarization label of the segment it occurs in. Prefer manifest `known_hosts` exact matches when a detected name matches a known host. Return `{label: name}`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(speakers): label->name extraction from intros + manifest hosts`.

---

## Phase 3 — Pipeline integration + fragmentation eval + the 20-episode driver

*Gate: `scripts/run_kg_slice.py` over a 20-episode manifest writes a graph where each probe entity ("the Fed", "OpenAI", a recurring guest) is ≤ a small number of canonical nodes, the fragmentation report prints the counts, and `disputes`/`supersedes` edges appear across episodes.*

### Task 3.1: Inject the resolvers into the pipeline

**Files:** Modify `src/dlogos/pipeline.py`; Test: `tests/test_pipeline_resolvers.py` (new).

- [ ] **Step 1 — failing test:** `PipelineDeps(..., subject_resolver=IncrementalResolver(...), speaker_resolver=NameSpeakerResolver(...))`; run two episodes sharing a subject "Apple"/"Apple Inc." and a host; assert (a) both episodes' Apple claims share one `canonical_id`, (b) the host has one `speaker_id` across episodes, (c) with both hooks `None` the behavior is byte-identical to today (back-compat).
- [ ] **Step 2 — run, expect FAIL** (`PipelineDeps` lacks the fields).
- [ ] **Step 3 — implement:**
  - Add `subject_resolver: SubjectResolverLike | None = None` and `speaker_resolver: object | None = None` to `PipelineDeps`, where `SubjectResolverLike` is a `Protocol` with `resolve(claims) -> SubjectResolution`.
  - In `Pipeline.run` step 8 (`pipeline.py:399`): `if self._deps.subject_resolver is not None: subject_resolution = self._deps.subject_resolver.resolve(all_claims) else: subject_resolution = resolve_subjects(all_claims, self._deps.embedder, threshold=self._similarity_threshold)`.
  - In `_resolve_speakers` (`pipeline.py:686`): when `speaker_resolver` is present, first compute `label_names = extract_label_names(transcript, known_hosts=...)` and let `speaker_resolver.resolve(...)` fill labels **before** the existing host-gallery/guest/fallback chain (name-resolved labels win; the rest fall through unchanged).
- [ ] **Step 4 — run, expect PASS**; `uv run pytest -q` green (back-compat test included).
- [ ] **Step 5 — commit:** `feat(pipeline): injectable persistent subject + speaker resolvers`.

### Task 3.2: Fragmentation report

**Files:** Create `src/dlogos/eval/fragmentation.py`; Test: `tests/eval/test_fragmentation.py`.

- [ ] **Step 1 — failing test:** given a list of `EntityNode`s (from a loaded store) where "Apple" appears as 3 separate `canonical_id`s and "OpenAI" as 1, and probes `[Probe("Apple", aliases=["apple","apple inc","the iphone maker"]), Probe("OpenAI", aliases=["openai"])]`, `fragmentation_report(nodes, probes)` reports `apple: 3 nodes`, `openai: 1 node`, and an overall `mean_fragments`/`worst` summary.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `Probe(name, aliases, qid=None)` and `fragmentation_report(entity_nodes, probes) -> FragReport`: for each probe, count distinct `canonical_id`s whose `name`/`aliases` (normalized) intersect the probe's aliases **or** whose `wikidata_qid == probe.qid`; report per-probe counts + `mean_fragments` + `worst`. Pure; reads `EntityNode`s (pull them from the store via `store.all_entities()`/`export_graph` — read `graph/export.py`/`fake_store.py` for the accessor and reuse it).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(eval): entity-fragmentation report (resolution-quality metric)`.

### Task 3.3: The 20-episode driver

**Files:** Create `scripts/run_kg_slice.py`; Test: `tests/test_run_kg_slice_script.py` (offline, fakes).

- [ ] **Step 1 — failing test:** a `run_kg_slice(...)` core (injectable, like `run_smoke`) over **two** fake episodes sharing a subject and a host, using `InMemoryCanonicalStore` + an in-memory speaker store + `FakeGraphStore`, asserts: one canonical Apple node, one host `speaker_id`, a `disputes` or `supersedes` edge present, and a non-empty `FragReport`. No network/heavy deps.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `scripts/run_kg_slice.py` mirroring `scripts/run_smoke_inmemory.py` (reuse `CachingASR` + `CachingExtractor`):
  - Accept a `--manifest` (a small `CorpusManifest` JSON of ~3–5 shows / 20 episodes total) **or** a `--audio-files` list for offline/dev.
  - Build real backends (AssemblyAI + DeepInfra) for the live run; wrap ASR with `resegment_by_words` (Phase 0); construct `SqliteCanonicalStore(out/canonical.db)`, `SqliteSpeakerStore(out/speakers.db)`, `IncrementalResolver(store, embedder, wikidata_linker=WikidataLinker(...), llm_adjudicator=llm_adjudicator_from_client(extraction_client, model))`, `NameSpeakerResolver(speaker_store)`; inject all via `PipelineDeps`.
  - Run all episodes through **one** `Pipeline.run` (cross-episode resolution), export `out/graph.json` (`write_graph_json`), run `fragmentation_report` with a small built-in probe set (configurable), and **print** the per-probe fragment counts + the graph summary. Persistent stores in `out/` so re-runs are incremental.
- [ ] **Step 4 — run, expect PASS** (the offline test); `uv run pytest -q` green.
- [ ] **Step 5 — commit:** `feat(scripts): 20-episode KG slice driver with persistent resolution + fragmentation report`.

---

## Phase 4 — Verify + view

*Gate: full suite green; the viewer renders the multi-episode graph; the fragmentation numbers are legible.*

### Task 4.1: Suite + viewer

- [ ] **Step 1:** `uv sync && uv run pytest -q` → all green (back-compat held). Grep `src/` for top-level heavy imports — none new (`sqlite3` is stdlib).
- [ ] **Step 2:** point the existing viewer at the slice graph: `uv run python -m dlogos.ui.graph_app --graph out/graph.json --port 8765` and confirm GET `/` + `/graph.json` are 200 with the multi-episode node/edge counts.
- [ ] **Step 3 — commit** any fixes: `test: verify scale-machinery suite green + viewer over 20-ep graph`.

---

## What this deliberately does NOT build (scale-path, not now)

- **ANN index (FAISS/HNSW).** Brute-force `candidates()` is fine at 20 episodes; the method is the only seam, so the swap is local later.
- **Voiceprint speaker identity.** Name-driven only here; pyannote/WhisperX embeddings (GPU) are the real cross-episode "who," and are the documented next step the moment audio quality / collapsed diarization (seen in the smoke) demands it.
- **Columnar analytical store (ClickHouse).** The SQLite stores are the persistence seam; the consensus-over-time OLAP mirror is a firehose-scale concern.
- **Concept ontology / human-in-the-loop merge review.** Concepts use a stricter cascade threshold + LLM adjudication; a curated concept ontology and a merge-review UI are later.

## Run order

Phase 0 → 1 → 2 → 3 → 4, each gate must pass before the next. Then the **live 20-episode run** (`scripts/run_kg_slice.py` against a real manifest) is the experiment we execute after the plan lands — it produces the fragmentation number that says how close to "one node per real entity" we actually are, and the graph the viewer shows.
