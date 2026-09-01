# ResearchAgentv2 — Roadmap

This document separates **implemented** capabilities from **planned** work. Only items under *Implemented* exist in the codebase today.

---

## Implemented: Phase 1–2D Research Foundation

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

See [architecture.md](architecture.md) for the current graph and data flow.

---

## Next: Phase 3 — Decision Intelligence

**Not implemented.** Potential sequence:

1. **Decision Framing** — structure the decision context from research outputs
2. **Option Evaluation** — compare alternatives against verified claims and knowledge state
3. **Recommendation / what would change the recommendation** — explicit sensitivity to new evidence

Prerequisites likely include Writer integration of verification and Knowledge State (not yet done).

---

## Future (Not Planned in Detail)

- Monitoring and alerting on research runs
- Change detection across runs (“what changed since last time”)
- Decision re-evaluation when new evidence arrives
- Action / operator layer (automated follow-up research, notifications)

---

## Explicitly Out of Scope (Today)

- Decision Engine
- Assumptions / risks / option analysis as first-class objects
- Graph database for claim networks
- Full evidence/claim caching system
- Fast-path Knowledge State compatibility
- UI redesign for knowledge buckets
