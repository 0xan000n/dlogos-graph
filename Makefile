.PHONY: sync test neo4j-up ui lint

# Install CORE deps only. Heavy optional extras (asr/graph/embed/ui/mcp) are NOT
# synced here — they are large and may be unreachable in CI/dev.
sync:
	uv sync

test:
	uv run pytest -q

neo4j-up:
	docker compose up -d neo4j

ui:
	uv run --extra ui python -m dlogos.ui.app

lint:
	uv run ruff check src tests
