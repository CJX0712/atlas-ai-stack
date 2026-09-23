# ADR-001: Layered architecture with Protocol-only cross-layer contracts

## Status
Accepted (2026-09-24)

## Background

The system has five externally visible capabilities: retrieval, reranking,
tool-augmented reasoning, grounding verification and evaluation. A conventional
layout puts these in one service module per feature. That layout makes two things
impossible to do well:

1. **Test an algorithm without its dependencies.** If the reranking module
   constructs its own model handle, testing the guard requires the model.
2. **Swap a backend without touching callers.** If a route handler imports the
   vector index directly, replacing it means editing handlers.

Both constraints are hard requirements here, because the verification chain must
run offline while the production path must be able to load real models.

## Decision

Six layers, with a strict downward-only import rule:

```
L0 contracts / settings / telemetry
L1 providers    (implements L0 ports)
L2 engines      (pure algorithms; imports nothing from L1 or L3)
L3 pipeline     (composition root)
L4 api          (transport)
L5 cli          (entry point)
```

- Every external capability is a `typing.Protocol` in `atlas/contracts.py`.
  Implementations do not inherit from the protocols; they satisfy them
  structurally.
- Implementations are injected. `AtlasContainer` is the only module that names a
  concrete backend.
- `engines/` imports only `contracts` and other `engines`. It performs no I/O.

## Consequences

**Positive**

- Every algorithm is unit-testable with a three-line fake. The guard test uses a
  lambda, not a model.
- The offline path and the production path are the same code with different
  injections. There is no "test mode" branch anywhere in the tree.
- Adding a backend is additive: one file plus one registry line.

**Negative**

- More indirection than a flat service. Reading the code requires following the
  container to learn which implementation is live.
- `Protocol` gives no runtime check. A provider that satisfies only part of a
  port fails at call time unless the module's own tests cover it. Mitigated by
  `scripts/verify.py` importing every module and by the integration tests
  exercising each port through the container.

## Related
ADR-002 (provider fallback), ADR-003 (zero-dependency defaults)
