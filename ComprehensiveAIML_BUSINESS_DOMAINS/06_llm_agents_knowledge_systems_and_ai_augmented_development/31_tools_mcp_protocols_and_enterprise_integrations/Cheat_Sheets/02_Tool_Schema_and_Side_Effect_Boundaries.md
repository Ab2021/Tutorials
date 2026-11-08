---
collection: "tools-mcp-protocols-and-enterprise-integrations"
file_id: "02"
topic: "Tool schema and side-effect boundaries"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Tool Schema and Side-Effect Boundaries

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) is the consulted inventory. This is a new course-topic draft; source bodies were unavailable.

## 1. A tool is a capability boundary

A tool definition states input schema, output/error schema, identity, authorization, resource limits, side-effect class, idempotency key and audit record. A valid function call is not a permitted or successful business operation.

## 2. Schema design

| Element | Contract | Avoid |
|---|---|---|
| Name | specific intent and action | overloaded `run`/`execute` |
| Input | typed required fields, bounds, examples | free-form command strings |
| Output | stable result/next-action fields | prose-only success |
| Errors | typed retryability and safe detail | leaking secrets or ambiguous failures |
| Version | compatibility and deprecation path | silent semantic reuse |

Validate syntax, authorization and domain semantics separately. `amount=100` can be type-valid but invalid for currency, account, approval or limit.

## 3. Side-effect taxonomy

Read-only tools may still expose sensitive data. Reversible writes need authoritative state and compensation conditions. Irreversible/external writes require explicit scope, approval binding, idempotency/reconciliation and audit. Never infer commit from client timeout.

## 4. Idempotency boundary

For operation key `k`, repeated accepted requests with the same semantic payload must resolve to one authoritative effect or a defined conflict. Bind key to caller, action and normalized payload; reuse of `k` with a different payload is an error. Cache/retry alone does not create exactly-once effects across an external system.

## 5. Invocation flow

1. Authenticate caller and resolve tenant/context.
2. Authorize capability and resource scope.
3. Validate schema and business preconditions.
4. Apply approval/idempotency/limit controls before side effect.
5. Execute, persist authoritative status and return typed outcome.
6. Emit redacted audit event; reconcile ambiguous outcomes.

## 6. Failure modes

| Failure | Signal | Containment |
|---|---|---|
| Overbroad tool | unrelated destructive actions exposed | split capabilities and least privilege |
| Timeout ambiguity | caller retries write | query authoritative status/reconcile |
| Schema-valid bad request | downstream business rejection | semantic validation and typed errors |
| Key collision/reuse | payload differs for same key | bind and reject conflict |
| Sensitive output leak | logs/model context contains secrets | minimize/redact and scope outputs |

## 7. Interview checks

1. **Why separate capability from execution?** Permission to call does not prove an effect occurred.
2. **What does idempotency require?** Stable key, semantic binding and authoritative result handling.
3. **Why typed errors?** The caller can decide retry, repair, escalation or stop.
4. **What is unsafe after timeout?** Blindly retrying a non-idempotent external write.
5. **What does a schema not prove?** Authorization, business validity or commit.

## 8. Quick Reference Card

For every tool: name one intent; type/bound inputs; state identity and scope; classify side effects; define approval, idempotency, errors, limits, audit and reconciliation; test authorized and denied paths plus crash/retry windows.

**Limitations:** No deployed protocol, tool server or source execution was inspected.
