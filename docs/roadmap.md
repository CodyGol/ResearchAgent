# ResearchAgentv2 — Roadmap

This document separates **implemented** capabilities from **future** work. Only items under *Implemented* exist in the codebase today.

---

## Implemented: Research Intelligence

| Phase | Capability | Status |
|-------|------------|--------|
| **1** | LangGraph agent, Planner / Researcher / Critic / Writer loop | ✅ |
| **1** | FastAPI NDJSON streaming, Next.js UI, LangSmith tracing | ✅ |
| **2A** | Source normalization, evidence extraction, integrity validation | ✅ |
| **2B** | Claim extraction, materiality, claim–evidence relations | ✅ |
| **2B.5+** | Query router (SIMPLE_FACT / STANDARD / DEEP), research budgets | ✅ |
| **2B.6–2B.7** | SIMPLE_FACT fast path (frozen) | ✅ |
| **2C** | Cross-source claim verification | ✅ |
| **2D** | Deterministic Knowledge State (final Critic exit only) | ✅ |

---

## Implemented: Decision Intelligence

| Phase | Capability | Status |
|-------|------------|--------|
| **3A** | Decision Framing — `DecisionFrame` from query + research context | ✅ |
| **3B** | Option Evaluation — evidence-grounded option×criterion matrix | ✅ |
| **3C** | Decision Synthesis — recommendation status, constraints, change conditions | ✅ |

The core Decision Intelligence architecture through Phase 3 is **frozen**. See [architecture.md](architecture.md) for graph, schemas, and invariants.

---

## Future / Productization (Not Implemented)

These are potential directions, not committed phases:

- **User-facing Decision Brief** — Writer integration so the final report presents structured recommendations
- **Persistent monitored decisions** — decision workspace across runs
- **Change detection** — what changed since last evaluation
- **Re-evaluation** — automatic refresh when new evidence arrives
- **Decision history** — audit trail of framing, evaluation, and synthesis over time
- **Action / operator layer** — automated follow-up research, notifications, execution

Also not implemented:

- Monitoring and alerting on research runs
- Planner / research guided by `DecisionFrame`
- Numerical utility, weights, or scoring systems
- Full evidence/claim caching system (plan cache only today)
- Fast-path Knowledge State compatibility
- UI redesign for knowledge buckets or decision artifacts

---

## Explicitly Out of Scope (Today)

- Graph database for claim networks
- Pseudo-options when `DecisionFrame` has no concrete options (3B skips evaluation instead)
- Upgrading `tentative_recommendation` → `recommend` via validators (validators may preserve or downgrade only)
