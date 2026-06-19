# dLogos PoC — Bring-up Runbook

This runbook takes the dLogos PoC from a clean checkout to a greenlight artifact,
one de-risking step at a time. Each step has a **gate**: if it fails, you stop
and fix before spending money on the next. Nothing scales past a failing gate.

The codebase is built so the whole tree runs **offline** on the core dependency
group (no GPU, no Neo4j, no API keys) against mocks/fakes — `uv run pytest`
proves the chain wires together. This runbook is about turning each stage on
against **real infra**, cheaply, in order.

- **Step 1 — one-episode smoke** (this doc, in full). The smallest real signal:
  one real episode → real ASR+diarization → real open-weight extraction → real
  graph → a retrieval answer with a checkable citation. Proves the FOUNDATION.
- **Steps 2–6** (brief, below). Each points at the as-built plan:
  `docs/superpowers/plans/2026-06-18-dlogos-poc-build-plan.md`.

---

## Step 1 — The one-episode smoke run

**What it proves.** That the machine produces *correct, sourced* claims: the
right claim, attributed to the right speaker, at a checkable timestamp. You
verify it **by ear** — open the audio at the printed timestamp and confirm the
named speaker is the one talking, saying what the claim says.

**What it does NOT prove.** One episode cannot show a temporal *shift* (a
position moving over time needs ≥ 2 episodes — that's step 2). The smoke is the
foundation, not the headline.

**Target infra (deliberately the lowest-friction path — hosted, no GPU):**

| Stage | Backend | Why |
|---|---|---|
| ASR + diarization | **AssemblyAI** (hosted) | one API key → speaker labels + word timestamps over HTTPS; no torch/pyannote/CUDA |
| Extraction | **DeepInfra**, `deepseek-ai/DeepSeek-V3` | OpenAI-compatible endpoint; the repo's extractor already speaks this wire protocol |
| Embeddings | **DeepInfra**, `BAAI/bge-m3` | same key; OpenAI-compatible `/embeddings` for subject-entity resolution |
| Graph | **Neo4j** via docker-compose (local) | the **direct** `Neo4jStore` (driver-level Cypher we control), NOT Graphiti — keeps dLogos logic isolated from the Graphiti A/B spike (step 3) |

### 1.1 Prerequisites (accounts / keys / tooling)

1. **AssemblyAI account** → an API key. Sign up at <https://www.assemblyai.com/>.
   Pay-as-you-go; transcription is billed per audio-hour.
2. **DeepInfra account** → an API key. Sign up at <https://deepinfra.com/>. The
   same key serves both the extraction model and the embedding model.
3. **Docker** (Desktop or Engine) running locally, for Neo4j.
4. **uv** (the Python toolchain): <https://docs.astral.sh/uv/>.

### 1.2 Exact commands

```bash
# 0) From the repo root, on branch build/dlogos-poc.

# 1) Start Neo4j locally (Bolt on :7687, browser on :7474).
docker compose up -d neo4j
#    Wait until healthy:
docker compose ps        # STATUS should read "healthy"

# 2) Configure secrets. Copy the template and fill in the four required keys.
cp .env.example .env
#    Edit .env and set:
#      ASSEMBLYAI_API_KEY=<your AssemblyAI key>
#      EXTRACTION_API_KEY=<your DeepInfra key>
#      EMBED_API_KEY=<your DeepInfra key>            # same key is fine
#    The EXTRACTION_/EMBED_ base URLs + models, and NEO4J_*, are pre-filled in
#    .env.example for this exact (DeepInfra + local Neo4j) setup.

# 3) Install dependencies. Core deps + the optional `graph` extra (the neo4j
#    driver). AssemblyAI uses httpx (core) and the extractor/embedder use openai
#    (core), so only `graph` is extra here.
uv sync --extra graph

# 4) Run the smoke on ONE episode. Two ways to point at the episode:

#  A) Direct audio URL (simplest — no Podcast Index key needed):
uv run python scripts/smoke_one_episode.py \
    --audio-url "https://example.com/path/to/episode.mp3" \
    --title "Episode title" --show-id my-show \
    --question "What does the guest claim about <entity>, and who says it?"

#  B) From an RSS feed via the Podcast Index (needs PODCAST_INDEX_KEY/SECRET):
uv run python scripts/smoke_one_episode.py \
    --feed-url "https://feeds.example.com/show.xml" --episode-index 0 \
    --question "What does the host claim about <entity>?"
```

Useful flags: `--language en` (skip auto-detect), `--speakers-expected 2` (hint
the diarizer), `--top-k 8` (how many claims to cite), `--frontier` (optional
head-to-head preview — see 1.5).

**Pick a short episode for the first run** (15–30 min): faster, cheaper, and a
two-speaker interview is the easiest thing to verify by ear.

### 1.3 What the script prints

```
==============================================================================
dLogos ONE-EPISODE SMOKE — attributed, cited answer
==============================================================================
Episode id    : ...
Audio         : 1840s, language=en
Diarized turns: 312
Claims loaded : 47 (extracted+resolved 47)
Canon. subjects: 19

QUESTION: What does the guest claim about <entity>, and who says it?
------------------------------------------------------------------------------
ANSWER:
<consensus/attribution synthesis over the retrieved claims>
------------------------------------------------------------------------------
CITED CLAIMS (verify each by ear at the printed timestamp):

  [1] speaker=spk-<episode>-b
      episode=...  span=[742.0s, 749.5s]  diarized_label=B
      claim text : <the claim>
      transcript : "<the exact words spoken at that span>"
      >> open the audio at 12:22 and confirm spk-<episode>-b is the one
         speaking, and is saying this.
  [2] ...
==============================================================================
```

The **transcript snippet** is the diarized text at the cited `[t_start, t_end]`,
pulled from the same transcript the pipeline built — so you can read it, then
scrub the audio to that timestamp and confirm it by ear.

### 1.4 Pass / fail criteria (the gate)

**PASS** — all four hold:

1. **Claims are real.** `Claims loaded ≥ 1`, and the cited claims are things
   actually said in the episode (not invented). Spot-check 3–5.
2. **Attribution is correct.** For each cited claim, the **named speaker is the
   one actually talking** at that timestamp. Open the audio at the printed
   `mm:ss` and listen. (The same who-is-speaking check the eval's
   speaker-verified rubric automates — here you are the rubric.)
3. **Citations check out.** The `[t_start, t_end]` span lands on the words the
   claim summarizes — the timestamp points at the right moment, not a nearby
   topic mention.
4. **The answer is grounded.** The synthesized answer reflects the cited claims
   and does not assert things with no citation behind them.

**FAIL** signals and where to look:

- *No claims loaded* → extraction returned nothing usable. Check `EXTRACTION_*`
  config and that the model id is served by DeepInfra; re-run with a more
  claim-dense episode.
- *Claims loaded but no citations printed* (script exits non-zero) → retrieval
  found nothing for your `--question`, or claims lack resolved speakers. Try a
  broader question naming the entity; check the diarization wasn't pruned to
  nothing.
- *Wrong speaker at a timestamp* → diarization mislabeled the turn (the §11 top
  risk: confident misattribution). One bad label on a short episode is a data
  point, not a stop — note it; the adversarial slice (step 4) stress-tests this.

### 1.5 Optional: head-to-head preview (`--frontier`)

Set `FRONTIER_BASE_URL` / `FRONTIER_API_KEY` / `FRONTIER_MODEL` in `.env`, then
add `--frontier`. The script runs the SAME question through the **model-alone**
arm (no tools) and the **dLogos** arm (graph-grounded) and prints them side by
side. The dLogos arm carries speaker-verified citations; the model-alone arm
cannot anchor a claim to a real episode timestamp. This is a preview of the
step-6 scorecard, on one episode.

### 1.6 Expected cost & runtime

For a ~30-minute, two-speaker episode (order-of-magnitude, not a quote):

| Item | Rough cost |
|---|---|
| AssemblyAI transcription (~0.5 audio-hr) | a few US cents to ~$0.20 |
| DeepInfra extraction (open-weight, a few dozen chunk calls) | cents |
| DeepInfra embeddings (BGE-M3, dozens of short strings) | sub-cent |
| Neo4j (local docker) | $0 |

**Total: roughly a few cents to well under a dollar per episode.**

**Runtime:** AssemblyAI transcription dominates and is roughly a fraction of the
audio length (typically minutes for a 30-minute episode, depending on queue);
the script polls until it completes. Extraction + resolution + load + the one
query then add seconds to a couple of minutes. Budget **~5–15 minutes** end to
end for a first short episode.

### 1.7 Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `CONFIG ERROR: Missing required configuration: ...` | a required key is blank in `.env` | the message lists exactly which vars; fill them |
| Neo4j connection refused / auth failure | container not up, or `NEO4J_PASSWORD` ≠ `NEO4J_AUTH` | `docker compose ps`; ensure both halves of `NEO4J_AUTH=neo4j/<pw>` match `NEO4J_USER`/`NEO4J_PASSWORD` |
| `ImportError: Neo4jStore requires the optional 'graph' extra` | ran `uv sync` without `--extra graph` | `uv sync --extra graph` |
| AssemblyAI polling times out | a very long episode, or a transient queue | re-run; raise the backend's `poll_timeout_s` if needed; start with a shorter episode |
| AssemblyAI `error` status | a dead/redirecting audio URL | confirm the enclosure URL plays in a browser; some feeds gate enclosures behind redirects |
| HTTP 401/403 from DeepInfra | wrong key or model id | check `EXTRACTION_API_KEY`/`EMBED_API_KEY` and that the model ids are exactly `deepseek-ai/DeepSeek-V3` / `BAAI/bge-m3` |
| Lots of claims, all one speaker | diarization collapsed speakers, or talk-time pruning dropped a quiet speaker | try `--speakers-expected N`; a monologue episode legitimately has one speaker |

> **Honesty note.** The real-infra components (`AssemblyAIBackend`,
> `OpenAICompatibleEmbedder`, the DeepInfra extractor, `Neo4jStore`) cannot be
> exercised in CI — there are no keys or Neo4j there. Their **first real run is
> this smoke**. What CI covers is their pure logic (request shaping, response
> mapping, Cypher/param building) against fakes, plus that importing them leaks
> no heavy dependency. Treat a green suite as "the wiring is correct", and this
> smoke as "the wiring works against real infra".

---

## Step 2 — Two-to-three-episode temporal-shift check

**Goal.** Show the headline dLogos move that one episode cannot: a *position
moving over time*. Run the smoke pipeline over **2–3 episodes of the same show**
(or the same recurring guest) published weeks/months apart, then ask a
`consensus_trend` / `belief_history` question and confirm the answer reports a
**shift** across attributed sources with the right direction.

**How.** Load each episode through the same path (the `Pipeline` already loads a
batch and joins each claim to its episode's publish date — the event-time anchor
for `consensus_over_time`). Query via the `consensus_trend` / `belief_history`
MCP tools or `GraphRetrievalSurface.consensus(...)`.

**Gate.** The trend's `direction` + `sentiment_delta` match what a human hears
across the episodes, bucketed by the resolved `canonical_id` (not fragmented
across surface variants). See plan **Phase 5 — Retrieve**.

---

## Step 3 — The Graphiti A/B spike gate

**Goal (spec §7.6, §12.1).** Decide the integration shape **before** scaling:
Approach **A** (Graphiti owns extraction, pointed at the open-weight endpoint)
vs Approach **B** (our extractor → resolution → bulk-load; Graphiti/Neo4j is
store + temporal manager + retrieval), with Graphiti's per-add LLM node-dedup
**disabled** on the bulk path.

**How.** `src/dlogos/spike/run_comparison.py` runs both arms over the same N
fixtures and captures claim quality, backfill throughput, and **$/episode**;
`src/dlogos/spike/score.py` is the gate.

**Gate.** Pick A or B **and** clear a throughput **and** a $-per-episode bar — a
good-but-slow or good-but-expensive shape fails. The smoke (step 1) already
isolates dLogos logic on the direct `Neo4jStore`, so this spike is a clean A/B on
the *integration* question, not the dLogos logic. See plan **Phase 1 — Spike**.

---

## Step 4 — The 10-show adversarial-slice gate

**Goal (spec §12.2).** Stress ASR + diarization + cross-episode speaker identity
on a deliberately **adversarial** slice — panel shows, remote-heavy interviews,
ad-saturated shows — where confident misattribution is most likely.

**How.** Run ~10 shows through ingestion → ASR → talk-time pruning → host-anchored
gallery + recurring-guest resolution (the `Pipeline.from_manifest` path), then
audit attribution.

**Gate.** ASR + speaker-attribution quality acceptable on the hard slice; the
speaker-verified citation check (the §11 teeth) passes at an acceptable rate. A
failure here sends you back to diarization/identity tuning, not forward to scale.
See plan **Phase 2 — Slice (adversarial)**.

---

## Step 5 — Scale to ~200 shows (full backfill)

**Goal (spec §12.4).** Full backfill — a broad sweep across all ~200 shows plus a
deeper ~18–24-month window on the subset that carries temporal shifts — at corpus
scale.

**How.** All of step-3 and step-4 infra at volume: the open-weight endpoint for
extraction, the embedding endpoint for resolution, Neo4j for the load, with the
dedup-bypass bulk path. Capture cost + throughput numbers as you go.

**Gate.** Cost + throughput land within budget; nothing scales past a failing
earlier gate. See plan **Phase 4 — Scale**.

---

## Step 6 — The four-arm eval scorecard (the greenlight artifact)

**Goal (spec §9, §12.6).** Against the **pre-registered** golden good-answer
shapes, run the **four-arm** head-to-head — model **alone** / model **+ web
search** / model **+ naive vector RAG** (independent index over the same
transcripts) / model **+ dLogos graph** — scored with the reweighted rubric that
**elevates temporal-consensus synthesis across attributed sources** and
**demotes recency / couldn't-have-known**.

**How.** `src/dlogos/eval/` (`golden.py`, `arms.py`, `rubric.py`, `blind.py`,
`agreement.py`, `runner.py`). Answer shapes are frozen before any arm output is
seen; the speaker-verified citation check caps attribution credit.

**Gate / framing.** The artifact is an **existence proof across a spread**, not a
generalization proof — and any dLogos win must show on **temporal-consensus
synthesis across attributed sources**, not on recency (which web search also
has). See plan **Phase 6 — Eval**.

---

### Where the gates can send you

Each gate can send the work back a step; nothing scales past a failing gate.
Step 1 is the cheapest possible real signal — run it first, on a short episode,
before wiring anything else.
