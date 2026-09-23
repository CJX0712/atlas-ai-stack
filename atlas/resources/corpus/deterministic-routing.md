# Deterministic pre-routing: do not ask a small model to do arithmetic

Small quantised models, in the half-billion-parameter class, are unreliable at
exact computation. They are good at fluency and paraphrase. Asking one to
evaluate twelve times the sum of three and four returned seventy-two in a
direct measurement, with no hedging.

This is not a training-data problem that a better prompt fixes. It is a
capability boundary. The correct engineering response is to stop asking.

## Architecture

Insert a deterministic classifier in front of the reasoning loop. It inspects
the raw query and decides whether the query is computable by a tool.

- If yes: invoke the tool directly. The language model is never called. The
  answer is exact by construction.
- If no: hand the query to the reasoning loop as usual.

The classifier must be cheap, total and explainable. Regular expressions over a
normalised query satisfy all three. The classifier returns a structured route
object carrying the kind, the tool name, the payload and a human-readable
reason, so every routing decision is auditable after the fact.

## What to route

Arithmetic expressions over numbers and the operators plus, minus, times,
divide, modulo, power and parentheses. Validate that the expression parses as a
pure arithmetic string before routing it, otherwise a natural-language question
containing digits will be misrouted.

Unit conversions between lengths, masses and temperatures.

Date differences between two explicit calendar dates, when an interval keyword
such as "how many days" is also present.

## What not to route

Anything requiring world knowledge, judgement, or reading a document. Do not
route a query merely because it contains a number. In particular, do not route
"what did the report say about the 2024 revenue" - that is a retrieval question
that happens to contain a digit.

## Safety of the arithmetic tool

Never call the built-in evaluator. Walk a whitelisted abstract syntax tree
instead: numeric constants, binary arithmetic operators, unary sign operators.
Reject every other node type. Bound the exponent so a crafted input cannot
allocate unbounded memory. Replace the caret with a double asterisk for user
convenience, then parse.

## Verification

The routing table is covered by a parameterised test asserting that every
expression which parses as arithmetic routes to the calculator, and that the
reasoning loop records zero model calls for such queries. Both properties are
checked offline.
