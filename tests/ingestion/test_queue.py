"""Tests for the job queue: enqueue/lease/ack/nack + idempotency.

Parametrized across both backends (in-memory + sqlite) so they satisfy the
same contract. A controllable clock makes lease-expiry deterministic.
"""

from __future__ import annotations

import pytest

from dlogos.ingestion.queue import (
    InMemoryJobQueue,
    JobStatus,
    SQLiteJobQueue,
)


class FakeClock:
    """Manually advanced clock for deterministic lease-expiry tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_queue(kind: str, clock):
    if kind == "memory":
        return InMemoryJobQueue(now=clock)
    return SQLiteJobQueue(":memory:", now=clock)


QUEUE_KINDS = ["memory", "sqlite"]


@pytest.fixture(params=QUEUE_KINDS)
def queue_kind(request):
    return request.param


def test_enqueue_and_lease_ack(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)

    job = q.enqueue({"guid": "g1", "audio_url": "https://a/1.mp3"})
    assert job.status is JobStatus.queued
    assert job.idempotency_key == "g1"

    leased = q.lease(lease_seconds=300)
    assert leased is not None
    assert leased.id == job.id
    assert leased.status is JobStatus.leased
    assert leased.attempts == 1
    assert leased.lease_expires_at == 1000.0 + 300

    q.ack(leased.id)
    assert q.get(leased.id).status is JobStatus.done
    # Nothing left to lease.
    assert q.lease() is None


def test_idempotent_enqueue_is_noop(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)

    first = q.enqueue({"guid": "dup"})
    second = q.enqueue({"guid": "dup", "extra": "ignored"})

    assert first.id == second.id
    assert q.stats()[JobStatus.queued] == 1


def test_explicit_idempotency_key_overrides_guid(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)
    a = q.enqueue({"guid": "g1"}, idempotency_key="custom")
    b = q.enqueue({"guid": "g2"}, idempotency_key="custom")
    assert a.id == b.id


def test_enqueue_without_key_or_guid_rejected(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)
    with pytest.raises(ValueError):
        q.enqueue({"audio_url": "https://a/1.mp3"})


def test_fifo_lease_order(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)
    q.enqueue({"guid": "g1"})
    q.enqueue({"guid": "g2"})
    q.enqueue({"guid": "g3"})

    leased = [q.lease().idempotency_key for _ in range(3)]
    assert leased == ["g1", "g2", "g3"]


def test_lease_expiry_makes_job_releasable(queue_kind) -> None:
    clock = FakeClock(start=1000.0)
    q = _make_queue(queue_kind, clock)
    q.enqueue({"guid": "g1"})

    first = q.lease(lease_seconds=60)
    assert first is not None
    # Before expiry, no other job can be leased.
    assert q.lease(lease_seconds=60) is None

    # Advance past the lease deadline → the job becomes leasable again.
    clock.advance(61)
    again = q.lease(lease_seconds=60)
    assert again is not None
    assert again.id == first.id
    assert again.attempts == 2  # attempt count incremented on re-lease


def test_nack_requeues(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)
    q.enqueue({"guid": "g1"})
    leased = q.lease()
    q.nack(leased.id, requeue=True)
    assert q.get(leased.id).status is JobStatus.queued
    # It can be leased again.
    assert q.lease().id == leased.id


def test_nack_fail_marks_failed(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)
    q.enqueue({"guid": "g1"})
    leased = q.lease()
    q.nack(leased.id, requeue=False)
    assert q.get(leased.id).status is JobStatus.failed
    assert q.lease() is None


def test_stats_counts(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)
    q.enqueue({"guid": "g1"})
    q.enqueue({"guid": "g2"})
    q.enqueue({"guid": "g3"})
    q.ack(q.lease().id)
    q.nack(q.lease().id, requeue=False)
    stats = q.stats()
    assert stats[JobStatus.done] == 1
    assert stats[JobStatus.failed] == 1
    assert stats[JobStatus.queued] == 1


def test_ack_unknown_job_raises(queue_kind) -> None:
    clock = FakeClock()
    q = _make_queue(queue_kind, clock)
    with pytest.raises(KeyError):
        q.ack("nope")


def test_sqlite_payload_round_trips() -> None:
    clock = FakeClock()
    q = SQLiteJobQueue(":memory:", now=clock)
    payload = {"guid": "g1", "audio_url": "https://a/1.mp3", "deep": True, "n": 3}
    q.enqueue(payload)
    leased = q.lease()
    assert leased.payload == payload


def test_sqlite_durability_across_instances(tmp_path) -> None:
    db = tmp_path / "jobs.db"
    clock = FakeClock()
    q1 = SQLiteJobQueue(db, now=clock)
    job = q1.enqueue({"guid": "g1"})
    q1.close()

    q2 = SQLiteJobQueue(db, now=clock)
    reread = q2.get(job.id)
    assert reread is not None
    assert reread.idempotency_key == "g1"
    # Idempotency survives a reopen: re-enqueue is still a no-op.
    again = q2.enqueue({"guid": "g1"})
    assert again.id == job.id
    q2.close()
