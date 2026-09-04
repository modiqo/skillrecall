---
name: sql-migrations
description: Write, review, and sequence database schema migrations safely, with rollback scripts and zero-downtime patterns. Use when asked to add a column, change a table, or plan a schema change on Postgres or MySQL.
---

# SQL migrations

## When to use

- Add a nullable column to the users table without downtime
- Plan a schema change for the orders table
- Write a rollback script for this migration
- Review this Postgres migration for locking problems

## Procedure

1. Classify the change as additive, destructive, or rewriting.
2. Choose the zero-downtime pattern.
3. Emit forward and rollback scripts.
