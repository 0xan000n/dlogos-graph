# dLogos PoC — Temporal Dialogue Knowledge Graph

A generalized, temporal, dialogue knowledge graph built from ~200 podcasts:
speakers, entities, and time-stamped, stance-tagged claims, queryable for
*who said what, when, and how the consensus moved*.

Full design: `docs/superpowers/specs/2026-06-18-dlogos-dialogue-graph-poc-design.md`.

## Pipeline

```
RSS ingest -> ASR + diarization -> cross-episode speaker identity
  -> open-weight LLM extraction of stance-tagged Claims (controlled predicate vocab)
  -> lightweight resolution (subject-entity clustering + recurring-guest resolution)
  -> Graphiti/Neo4j bitemporal graph (bulk load bypasses per-add LLM dedup)
  -> hybrid + temporal retrieval -> MCP server + thin UI
  -> 4-arm head-to-head eval
```

## Development

```bash
make sync      # uv sync (CORE deps only; heavy extras are NOT installed)
make test      # uv run pytest
make neo4j-up  # docker compose up neo4j
make ui        # launch the thin side-by-side UI
make lint      # ruff (optional)
```

Heavy/optional deps (torch, whisperx, pyannote, graphiti-core, neo4j, gradio,
mcp, FlagEmbedding, sentence-transformers) are imported **lazily** inside
functions. Importing any module for tests requires only the core dependency
group; mock/in-memory implementations back the unit tests.
