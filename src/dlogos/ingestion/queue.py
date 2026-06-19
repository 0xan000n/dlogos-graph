"""A small work-queue abstraction for backfill fan-out + incremental polling.

Spec §7.1: "Model as a queue → workers ... a Postgres job table is fine at PoC
scale. Backfill is a bounded fan-out; incremental is a poller on the same
path." We keep the abstraction deliberately tiny — ``enqueue`` / ``lease`` /
``ack`` / ``nack`` — with two interchangeable backends:

- :class:`InMemoryJobQueue` — for tests and single-process runs.
- :class:`SQLiteJobQueue` — a durable Postgres-job-table stand-in at PoC scale,
  using only the stdlib ``sqlite3`` (no heavy deps).

**Idempotency.** ``enqueue`` takes an idempotency key (default: the payload's
episode GUID). Re-enqueuing the same key is a no-op that returns the existing
job, so re-polling an RSS feed never creates duplicate work. ``lease`` hands a
queued job to a worker with a lease deadline; ``ack`` marks it done; ``nack``
returns it to the queue (with an attempt count) or marks it failed.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    """Lifecycle states for a queued job."""

    queued = "queued"
    leased = "leased"
    done = "done"
    failed = "failed"


@dataclass
class Job:
    """A unit of work flowing through the queue."""

    id: str
    idempotency_key: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.queued
    attempts: int = 0
    lease_expires_at: float | None = None


def _new_job_id() -> str:
    return uuid.uuid4().hex


class JobQueue(ABC):
    """Abstract queue: enqueue / lease / ack / nack with idempotency."""

    @abstractmethod
    def enqueue(
        self, payload: dict[str, Any], *, idempotency_key: str | None = None
    ) -> Job:
        """Add a job. Re-enqueuing a known idempotency key is a no-op.

        If ``idempotency_key`` is omitted it falls back to ``payload["guid"]``;
        a payload with neither is rejected so dedupe can never silently break.
        """

    @abstractmethod
    def lease(self, *, lease_seconds: float = 300.0) -> Job | None:
        """Atomically claim the oldest queued (or lease-expired) job, or None."""

    @abstractmethod
    def ack(self, job_id: str) -> None:
        """Mark a leased job as successfully done."""

    @abstractmethod
    def nack(self, job_id: str, *, requeue: bool = True) -> None:
        """Release a leased job: requeue it, or mark it failed."""

    @abstractmethod
    def get(self, job_id: str) -> Job | None:
        """Fetch a job by id (None if absent)."""

    @abstractmethod
    def stats(self) -> dict[JobStatus, int]:
        """Count of jobs per status — for backfill progress reporting."""


def _resolve_key(payload: dict[str, Any], idempotency_key: str | None) -> str:
    if idempotency_key is not None:
        if not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        return idempotency_key
    guid = payload.get("guid")
    if not guid:
        raise ValueError(
            "enqueue requires an idempotency_key or a payload['guid']"
        )
    return str(guid)


class InMemoryJobQueue(JobQueue):
    """Single-process, dict-backed queue. Deterministic given a clock.

    ``now`` is injectable for deterministic lease-expiry tests.
    """

    def __init__(self, *, now: Any = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        # Insertion-ordered queue of job ids; FIFO lease order.
        self._order: list[str] = []
        self._clock = now if now is not None else time.time

    def _time(self) -> float:
        return float(self._clock())

    def enqueue(
        self, payload: dict[str, Any], *, idempotency_key: str | None = None
    ) -> Job:
        key = _resolve_key(payload, idempotency_key)
        existing_id = self._by_key.get(key)
        if existing_id is not None:
            return self._jobs[existing_id]
        job = Job(id=_new_job_id(), idempotency_key=key, payload=dict(payload))
        self._jobs[job.id] = job
        self._by_key[key] = job.id
        self._order.append(job.id)
        return job

    def lease(self, *, lease_seconds: float = 300.0) -> Job | None:
        now = self._time()
        for job_id in self._order:
            job = self._jobs[job_id]
            available = job.status == JobStatus.queued or (
                job.status == JobStatus.leased
                and job.lease_expires_at is not None
                and job.lease_expires_at <= now
            )
            if available:
                job.status = JobStatus.leased
                job.attempts += 1
                job.lease_expires_at = now + lease_seconds
                return job
        return None

    def ack(self, job_id: str) -> None:
        job = self._require(job_id)
        job.status = JobStatus.done
        job.lease_expires_at = None

    def nack(self, job_id: str, *, requeue: bool = True) -> None:
        job = self._require(job_id)
        job.lease_expires_at = None
        job.status = JobStatus.queued if requeue else JobStatus.failed

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def stats(self) -> dict[JobStatus, int]:
        counts = {s: 0 for s in JobStatus}
        for job in self._jobs.values():
            counts[job.status] += 1
        return counts

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job id: {job_id}")
        return job


class SQLiteJobQueue(JobQueue):
    """Durable queue backed by stdlib ``sqlite3`` (Postgres-job-table stand-in).

    Pass ``path=":memory:"`` (the default) for an ephemeral DB in tests, or a
    file path for durability across processes. Idempotency is enforced by a
    UNIQUE constraint on ``idempotency_key``, so concurrent enqueues of the same
    key collapse to one row.
    """

    def __init__(self, path: str | Path = ":memory:", *, now: Any = None) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._clock = now if now is not None else time.time
        self._init_schema()

    def _time(self) -> float:
        return float(self._clock())

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id               TEXT PRIMARY KEY,
                idempotency_key  TEXT NOT NULL UNIQUE,
                payload          TEXT NOT NULL,
                status           TEXT NOT NULL,
                attempts         INTEGER NOT NULL DEFAULT 0,
                lease_expires_at REAL,
                seq              INTEGER
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status_seq ON jobs(status, seq)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteJobQueue":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def enqueue(
        self, payload: dict[str, Any], *, idempotency_key: str | None = None
    ) -> Job:
        key = _resolve_key(payload, idempotency_key)
        existing = self._conn.execute(
            "SELECT * FROM jobs WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if existing is not None:
            return _row_to_job(existing)
        job_id = _new_job_id()
        seq = self._next_seq()
        self._conn.execute(
            """
            INSERT INTO jobs
                (id, idempotency_key, payload, status, attempts, lease_expires_at, seq)
            VALUES (?, ?, ?, ?, 0, NULL, ?)
            """,
            (job_id, key, json.dumps(payload), JobStatus.queued.value, seq),
        )
        self._conn.commit()
        return Job(
            id=job_id,
            idempotency_key=key,
            payload=dict(payload),
            status=JobStatus.queued,
            attempts=0,
            lease_expires_at=None,
        )

    def _next_seq(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM jobs").fetchone()
        return int(row["n"])

    def lease(self, *, lease_seconds: float = 300.0) -> Job | None:
        now = self._time()
        # Oldest job that is queued, or leased-but-expired. BEGIN IMMEDIATE
        # serializes concurrent leases on the same SQLite file.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                   OR (status = ? AND lease_expires_at IS NOT NULL
                       AND lease_expires_at <= ?)
                ORDER BY seq ASC
                LIMIT 1
                """,
                (JobStatus.queued.value, JobStatus.leased.value, now),
            ).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            new_attempts = int(row["attempts"]) + 1
            expires = now + lease_seconds
            self._conn.execute(
                "UPDATE jobs SET status = ?, attempts = ?, lease_expires_at = ? WHERE id = ?",
                (JobStatus.leased.value, new_attempts, expires, row["id"]),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        job = _row_to_job(row)
        job.status = JobStatus.leased
        job.attempts = new_attempts
        job.lease_expires_at = expires
        return job

    def ack(self, job_id: str) -> None:
        self._require(job_id)
        self._conn.execute(
            "UPDATE jobs SET status = ?, lease_expires_at = NULL WHERE id = ?",
            (JobStatus.done.value, job_id),
        )
        self._conn.commit()

    def nack(self, job_id: str, *, requeue: bool = True) -> None:
        self._require(job_id)
        new_status = JobStatus.queued if requeue else JobStatus.failed
        self._conn.execute(
            "UPDATE jobs SET status = ?, lease_expires_at = NULL WHERE id = ?",
            (new_status.value, job_id),
        )
        self._conn.commit()

    def get(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return _row_to_job(row) if row is not None else None

    def stats(self) -> dict[JobStatus, int]:
        counts = {s: 0 for s in JobStatus}
        for row in self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ):
            counts[JobStatus(row["status"])] = int(row["n"])
        return counts

    def _require(self, job_id: str) -> None:
        row = self._conn.execute(
            "SELECT id FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown job id: {job_id}")


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        idempotency_key=row["idempotency_key"],
        payload=json.loads(row["payload"]),
        status=JobStatus(row["status"]),
        attempts=int(row["attempts"]),
        lease_expires_at=row["lease_expires_at"],
    )
