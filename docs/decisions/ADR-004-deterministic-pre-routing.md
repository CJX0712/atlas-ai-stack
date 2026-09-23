# ADR-004: Intercept computable queries before the model; never ask a small model to compute

## Status
Accepted (2026-09-24)

## Background

A ReAct agent backed by a 0.5B quantised model answered `12*(3+4)` with `72`.
No hedging, no uncertainty marker — a confident wrong answer.

This is not a prompt-engineering problem. Exact arithmetic is outside the
capability envelope of models in this size class. Prompting harder does not move
a capability boundary; it only changes how the failure is phrased.

## Decision

Insert a deterministic classifier ahead of the reasoning loop.

```
question
   |
   v
router.route(question)  -- arithmetic | unit | date --> tool --> answer (steps = 0)
   |
   +-- otherwise --> ReAct loop (model involved)
```

Rules:

1. The classifier is regex-based over a normalised query. It must be cheap,
   total and explainable. It returns a structured `Route` carrying kind, tool,
   payload and a human-readable reason, so every decision is auditable.
2. Arithmetic is only routed when the whole expression consists of digits,
   operators, parentheses and whitespace, and parentheses balance. A question
   that merely contains digits must not route.
3. Full-width punctuation is stripped from the ends before the character test.
   An unstripped `？` made an otherwise valid expression fail the gate — a real
   defect found by the evaluation set, not by inspection.
4. The arithmetic tool walks a whitelisted AST. `eval` is never called. The
   exponent is bounded so a crafted input cannot exhaust memory.

## Consequences

**Positive**

- Arithmetic, unit conversion and date differences are exact by construction.
- Cost drops: a tool-routed query makes zero model calls. This is asserted
  (`steps == 0`, `llm.calls == 0`), so a regression that starts calling the model
  fails the build.
- Routing decisions are logged and returned in the answer trace.

**Negative**

- The regex set is a maintenance surface. It is deliberately conservative: a
  missed route degrades to the model, which is the previous behaviour, while a
  false route would answer the wrong question. The tests assert both directions,
  including the negative cases (`"2024 年的营收是多少"` must not route).
- Non-arithmetic exact reasoning (algebra, currency conversion at a rate, time
  zones) is still delegated to the model. Out of scope for v1.0.

## Related
`atlas/engines/router.py`, `atlas/engines/tools.py`, `tests/test_router_tools.py`
