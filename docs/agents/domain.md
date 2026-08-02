# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

This is a **single-context repository**.

Domain vocabulary belongs in `CONTEXT.md` at the repository root. System-wide architecture decisions belong under `docs/adr/`.

## Before exploring, read these

- **`CONTEXT.md`** at the repository root.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If either location does not exist, proceed silently. Do not flag its absence or suggest creating it upfront. Domain-modeling skills create these files lazily when terminology or architectural decisions are actually resolved.

## File structure

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
│       ├── 0001-example-decision.md
│       └── 0002-another-decision.md
└── src/
```

## Use the glossary's vocabulary

When output names a domain concept—such as in an issue title, refactor proposal, hypothesis, or test name—use the term defined in `CONTEXT.md`. Do not drift to synonyms the glossary explicitly avoids.

If a required concept is absent from the glossary, reconsider whether the language belongs to the project or note the gap for domain modeling.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly rather than silently overriding it:

> _Contradicts ADR-0007 (event-sourced orders)—but worth reopening because…_
