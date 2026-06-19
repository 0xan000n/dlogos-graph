"""Hosted, OpenAI-compatible embedder (DeepInfra / BGE-M3) for the smoke run.

The subject-entity clustering (:mod:`dlogos.resolution.subjects`) and the
retrieval surface depend only on a tiny structural :class:`Embedder` protocol —
``embed(text) -> list[float]`` plus an optional ``embed_batch``. The offline
default (:func:`dlogos.resolution.subjects.default_embedder`) builds a *local*
BGE-M3 through ``FlagEmbedding`` (a heavy GPU-ish extra). The one-episode smoke
run deliberately targets the *lowest-friction* path instead: a **hosted**
OpenAI-compatible ``/embeddings`` endpoint (DeepInfra serving ``BAAI/bge-m3``),
which needs only an API key over HTTPS — no torch, no FlagEmbedding, no GPU.

This adapter is the embedding twin of
:meth:`dlogos.extraction.extractor.ClaimExtractor.from_settings`: it speaks the
same OpenAI wire protocol (``client.embeddings.create(model=..., input=[...])``)
and reads ``EMBED_BASE_URL`` / ``EMBED_API_KEY`` / ``EMBED_MODEL`` from
:class:`~dlogos.config.Settings`.

HARD CONSTRAINT — the ``openai`` SDK is imported **lazily** inside
:meth:`from_settings`, never at module top level, so importing this module costs
nothing and pulls in no client. The client itself is injectable, so the adapter
stays importable (and unit-testable) without any key: tests inject a fake client
exposing ``embeddings.create`` and assert the pure request/response mapping.

HONESTY: like the hosted ASR + extractor, this adapter's first REAL call is the
smoke run itself (no key/network here). The unit tests cover the request shape
and the response→vector mapping against a fake client returning canned data.
"""

from __future__ import annotations

from typing import Any, Protocol


class _Embeddings(Protocol):
    def create(self, *, model: str, input: list[str], **kwargs: Any) -> Any: ...


class _EmbeddingsClient(Protocol):
    """Structural type for the slice of a sync ``OpenAI`` client we use.

    A real ``openai.OpenAI`` satisfies this; tests inject a fake exposing
    ``embeddings.create(model=..., input=[...]) -> response``. Keeping the
    dependency structural means no heavy import is needed to type or test the
    adapter.
    """

    embeddings: _Embeddings


class OpenAICompatibleEmbedder:
    """An :class:`~dlogos.resolution.subjects.Embedder` over a hosted endpoint.

    Satisfies the structural ``Embedder`` protocols in
    :mod:`dlogos.resolution.subjects` and :mod:`dlogos.retrieval.hybrid`
    (``embed`` required, ``embed_batch`` used when present), so it drops into the
    pipeline's subject-entity resolution and the retrieval surface unchanged.

    Parameters
    ----------
    client:
        An object satisfying :class:`_EmbeddingsClient` (a real ``openai.OpenAI``
        in production, a fake in tests). Construct a real one via
        :meth:`from_settings`.
    model:
        The embedding model id (e.g. ``BAAI/bge-m3``). Defaults to the configured
        ``embed_model`` resolved lazily on first use when ``None``.
    batch_size:
        Max inputs per ``/embeddings`` request. The endpoint caps batch size, so
        :meth:`embed_batch` chunks larger inputs into multiple requests.

    Nothing in ``__init__`` imports ``openai`` or reads settings — both are
    deferred so the adapter is importable without the SDK or any key.
    """

    def __init__(
        self,
        client: _EmbeddingsClient,
        *,
        model: str | None = None,
        batch_size: int = 100,
    ) -> None:
        self._client = client
        self._model = model
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size!r}")
        self._batch_size = batch_size

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_settings(cls, settings: Any | None = None) -> "OpenAICompatibleEmbedder":
        """Build an embedder backed by a real ``openai.OpenAI`` client.

        The ``openai`` import happens here, lazily, so importing this module
        never constructs a network client. Reads ``EMBED_BASE_URL`` /
        ``EMBED_API_KEY`` / ``EMBED_MODEL`` from :class:`~dlogos.config.Settings`.
        """

        if settings is None:
            from dlogos.config import settings as default_settings

            settings = default_settings
        from openai import OpenAI  # lazy: avoid import at module load

        client = OpenAI(
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
        )
        return cls(client, model=settings.embed_model)

    # ------------------------------------------------------------------ #
    # Lazy model resolution
    # ------------------------------------------------------------------ #
    def _resolve_model(self) -> str:
        if self._model:
            return self._model
        from dlogos.config import settings

        model = (settings.embed_model or "").strip()
        if not model:
            raise ValueError(
                "No embedding model configured: set EMBED_MODEL or pass model=."
            )
        self._model = model
        return model

    # ------------------------------------------------------------------ #
    # Embedder protocol
    # ------------------------------------------------------------------ #
    def embed(self, text: str) -> list[float]:
        """Embed a single string into a dense vector."""

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings, chunking into ``batch_size``-sized requests.

        The endpoint preserves input order within a response, so we concatenate
        per-batch vectors in request order to keep the output aligned 1:1 with
        ``texts`` (the clustering and retrieval code index positionally).
        """

        model = self._resolve_model()
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            if not batch:
                continue
            resp = self._client.embeddings.create(model=model, input=batch)
            out.extend(_vectors_from_response(resp, expected=len(batch)))
        return out


def _vectors_from_response(resp: Any, *, expected: int) -> list[list[float]]:
    """Pull embedding vectors out of an OpenAI-compatible response, in order.

    Tolerates both object-style (``resp.data[i].embedding``, the real SDK) and
    dict-style (``resp["data"][i]["embedding"]``, lightweight fakes) responses.
    Sorts by each item's ``index`` when present so order survives any
    out-of-order delivery, then validates the count matches the request.
    """

    data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
    if not data:
        raise ValueError("embeddings response had no 'data'")

    indexed: list[tuple[int, list[float]]] = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            vec = item.get("embedding")
            idx = item.get("index", i)
        else:
            vec = getattr(item, "embedding", None)
            idx = getattr(item, "index", i)
        if vec is None:
            raise ValueError("embeddings response item missing 'embedding'")
        indexed.append((int(idx), [float(x) for x in vec]))

    indexed.sort(key=lambda pair: pair[0])
    vectors = [vec for _, vec in indexed]
    if len(vectors) != expected:
        raise ValueError(
            f"embeddings response returned {len(vectors)} vectors for "
            f"{expected} inputs"
        )
    return vectors
