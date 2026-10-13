# ADR 0004: SQLite (WAL) on the data volume, not a database server

**Status:** accepted · **Date:** 2026-10-12

## Context
One box serves up to 4 cameras and a few video jobs a day. Writes: one row per counted bag (≈1 per second per
camera at most), status updates every 5 s, job progress every second.

## Decision
SQLite in WAL mode on the shared data volume, short-lived connections, every state transition a single guarded
UPDATE (`claim_job`, `recover_interrupted`). The API and the worker are separate processes on the same host.

## Consequences
- Nothing extra to install, back up or secure at the factory; a backup is one file.
- Not for several hosts: SQLite over a network filesystem is unsafe. Scaling past one box means Postgres
  (the `Store` class is the only module that knows SQL) and an object store for outputs.
