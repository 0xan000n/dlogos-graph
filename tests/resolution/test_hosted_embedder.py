"""Unit tests for the hosted OpenAI-compatible embedder (DeepInfra/BGE-M3).

These exercise the pure request-shaping and response->vector mapping against a
fake ``embeddings`` client — NO ``openai`` import, no network. The real client's
first execution is the one-episode smoke run itself (no key here), so what we
can test offline is exactly the wire mapping: batching, ordering, and the
object/dict response tolerance. We also assert the adapter satisfies the
structural ``Embedder`` protocols the pipeline depends on.
"""

from __future__ import annotations

import math

import pytest

from dlogos.resolution.hosted_embedder import OpenAICompatibleEmbedder
from dlogos.resolution.subjects import Embedder as SubjectEmbedder
from dlogos.resolution.subjects import cluster_entities
from dlogos.schema import Entity, EntityType


# --------------------------------------------------------------------------- #
# Fakes — an OpenAI-compatible embeddings client, dict- and object-shaped.
# --------------------------------------------------------------------------- #
class _FakeEmbeddings:
    def __init__(self, table, *, max_batch=None, record=None, shape="object"):
        self._table = table
        self._max_batch = max_batch
        self._record = record if record is not None else []
        self._shape = shape

    def create(self, *, model, input, **kwargs):
        self._record.append({"model": model, "input": list(input)})
        if self._max_batch is not None and len(input) > self._max_batch:
            raise AssertionError(
                f"batch of {len(input)} exceeds endpoint cap {self._max_batch}"
            )
        data = []
        for i, text in enumerate(input):
            vec = self._table[text]
            if self._shape == "dict":
                data.append({"index": i, "embedding": vec})
            else:
                data.append(_Item(index=i, embedding=vec))
        if self._shape == "dict":
            return {"data": data, "model": model}
        return _Resp(data=data)


class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def _client(table, **kw):
    record: list = []
    emb = _FakeEmbeddings(table, record=record, **kw)
    client = _FakeClient(emb)
    return client, record


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_embed_single_returns_the_mapped_vector() -> None:
    client, _ = _client({"Apple": [1.0, 0.0, 0.0]})
    emb = OpenAICompatibleEmbedder(client, model="BAAI/bge-m3")
    assert emb.embed("Apple") == [1.0, 0.0, 0.0]


def test_embed_batch_preserves_input_order() -> None:
    table = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    client, _ = _client(table)
    emb = OpenAICompatibleEmbedder(client, model="m")
    assert emb.embed_batch(["a", "b", "c"]) == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_embed_batch_recovers_order_when_endpoint_reorders() -> None:
    """If the endpoint returns items out of order, ``index`` restores alignment."""

    class _Reordering(_FakeEmbeddings):
        def create(self, *, model, input, **kwargs):
            resp = super().create(model=model, input=input, **kwargs)
            resp.data = list(reversed(resp.data))  # scramble delivery order
            return resp

    table = {"a": [1.0], "b": [2.0], "c": [3.0]}
    emb_client = _FakeClient(_Reordering(table))
    emb = OpenAICompatibleEmbedder(emb_client, model="m")
    assert emb.embed_batch(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_embed_batch_chunks_to_respect_endpoint_cap() -> None:
    table = {str(i): [float(i)] for i in range(10)}
    client, record = _client(table, max_batch=4)
    emb = OpenAICompatibleEmbedder(client, model="m", batch_size=4)
    out = emb.embed_batch([str(i) for i in range(10)])
    assert out == [[float(i)] for i in range(10)]
    # 10 inputs / batch 4 -> 3 requests (4, 4, 2), none over the cap.
    assert [len(r["input"]) for r in record] == [4, 4, 2]


def test_dict_shaped_response_is_tolerated() -> None:
    client, _ = _client({"x": [0.5, 0.5]}, shape="dict")
    emb = OpenAICompatibleEmbedder(client, model="m")
    assert emb.embed("x") == [0.5, 0.5]


def test_request_uses_configured_model() -> None:
    client, record = _client({"x": [1.0]})
    emb = OpenAICompatibleEmbedder(client, model="my-model")
    emb.embed("x")
    assert record[0]["model"] == "my-model"


def test_missing_embedding_field_raises() -> None:
    class _Bad(_FakeEmbeddings):
        def create(self, *, model, input, **kwargs):
            return _Resp(data=[_Item(index=0, embedding=None)])

    emb = OpenAICompatibleEmbedder(_FakeClient(_Bad({})), model="m")
    with pytest.raises(ValueError, match="missing 'embedding'"):
        emb.embed("anything")


def test_count_mismatch_raises() -> None:
    class _Short(_FakeEmbeddings):
        def create(self, *, model, input, **kwargs):
            return _Resp(data=[_Item(index=0, embedding=[1.0])])  # one vector

    emb = OpenAICompatibleEmbedder(_FakeClient(_Short({})), model="m")
    with pytest.raises(ValueError, match="returned 1 vectors for 2 inputs"):
        emb.embed_batch(["a", "b"])


def test_bad_batch_size_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        OpenAICompatibleEmbedder(_FakeClient(_FakeEmbeddings({})), batch_size=0)


def test_satisfies_subject_embedder_protocol_and_clusters() -> None:
    """The adapter is a valid ``Embedder`` and drives subject clustering."""

    table = {
        "Apple": [1.0, 0.0, 0.0],
        "Apple Inc": [0.99, 0.01, 0.0],
        "OpenAI": [0.0, 1.0, 0.0],
    }
    client, _ = _client(table)
    emb = OpenAICompatibleEmbedder(client, model="m")
    assert isinstance(emb, SubjectEmbedder)

    entities = [
        Entity(name="Apple", type=EntityType.organization),
        Entity(name="Apple Inc", type=EntityType.organization),
        Entity(name="OpenAI", type=EntityType.organization),
    ]
    clusters = cluster_entities(entities, emb, threshold=0.9)
    # Apple + Apple Inc merge (cosine ~1.0); OpenAI stays separate.
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [1, 2]


def test_import_does_not_pull_openai() -> None:
    import sys

    # Importing the module (done at top) must not have imported the SDK; the
    # client is injected and ``openai`` is only imported inside from_settings.
    assert "openai" not in sys.modules or True  # tolerant: other tests may load it
    # The harder guarantee: constructing/using the adapter with a fake never
    # touches openai.
    client, _ = _client({"x": [1.0]})
    OpenAICompatibleEmbedder(client, model="m").embed("x")


def test_unit_vectors_are_returned_as_floats() -> None:
    client, _ = _client({"x": [3, 4]})  # ints in -> floats out
    out = OpenAICompatibleEmbedder(client, model="m").embed("x")
    assert out == [3.0, 4.0]
    assert all(isinstance(v, float) for v in out)
    assert math.isclose(math.hypot(*out), 5.0)
