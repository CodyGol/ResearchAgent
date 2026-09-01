# ResearchAgentv2 — System Architecture (Phase 3C)

This document describes the **implemented** architecture through Phase 3C (Decision Synthesis). Items in [docs/roadmap.md](roadmap.md) under *Future* are not implemented unless explicitly marked here.

## High-Level System

```mermaid
graph TB
    subgraph Frontend
        UI[Next.js Deep Research Console]
        APIRoute[/api/research Edge proxy]
    end

    subgraph Backend
        FastAPI[api.py — NDJSON streaming]
        Graph[LangGraph StateGraph — graph.py]
    end

    subgraph External
        Claude[Anthropic Claude]
        Tavily[Tavily Search]
        LangSmith[LangSmith Tracing]
        Supabase[Supabase PostgreSQL]
    end

    UI --> APIRoute --> FastAPI --> Graph
    Graph --> Claude
    Graph --> Tavily
    Graph --> LangSmith
    Graph --> Supabase
```

## Agent Graph (Implemented)

```mermaid
flowchart TD
    START([User Query]) --> ROUTER[Router]

    ROUTER -->|SIMPLE_FACT| FAST[Fast Path]
    ROUTER -->|STANDARD / DEEP| FRAMER[Decision Framer]

    FAST -->|success| END1([END])
    FAST -->|escalate| FRAMER

    FRAMER --> PLANNER[Planner]
    PLANNER --> RESEARCHER[Researcher]
    RESEARCHER --> EVIDENCE[Evidence Extractor]
    EVIDENCE --> CLAIMS[Claim Extractor]
    CLAIMS --> VERIFIER[Claim Verifier]
    VERIFIER --> CRITIC[Critic]

    CRITIC -->|insufficient and under iteration budget| RESEARCHER
    CRITIC -->|sufficient or max iterations| KS[Knowledge State]

    KS -->|no decision_frame| WRITER1[Writer]
    KS -->|decision_frame present| OE[Option Evaluator]

    OE -->|options empty| WRITER2[Writer]
    OE -->|options present| DS[Decision Synthesizer]

    DS --> WRITER3[Writer]
    WRITER1 --> END2([END])
    WRITER2 --> END2
    WRITER3 --> END2
```

### Route: SIMPLE_FACT (frozen)

Narrow factual questions bypass the full research pipeline:

```
Question → Fact Target → Targeted Search → Decisive Evidence
       → Structured Fact Value → Validation → Canonical Claim → Concise Answer
```

Implemented in `services/fast_path.py`, `services/fact_target.py`, `services/fact_value.py`, `nodes/fast_path.py`. On failure, escalates to the STANDARD/DEEP path at **Decision Framer** (not Planner directly).

### Route: STANDARD / DEEP

Full evidence-grounded pipeline with configurable research budgets (`services/query_router.py`). **Decision orientation** (whether a `DecisionFrame` is produced) is separate from routing complexity (SIMPLE_FACT / STANDARD / DEEP).

## Trusted Research Chain

```
SOURCE
  → VALIDATED EVIDENCE
  → DIRECT ATOMIC CLAIM
  → MATERIAL CLAIM
  → CROSS-SOURCE VERIFICATION
  → KNOWLEDGE STATE
```

Every cited fact in the Writer report should trace: **Answer → Evidence → Source → URL**. Claims and verifications provide structured provenance; Knowledge State summarizes epistemic status per material claim.

## Trusted Decision Chain

```
DECISION FRAME + KNOWLEDGE STATE
  → OPTION EVALUATION
  → DECISION SYNTHESIS
```

End-to-end (decision-oriented runs):

```
SOURCE → EVIDENCE → CLAIM → VERIFICATION → KNOWLEDGE STATE
  → DECISION FRAME → OPTION EVALUATION → DECISION SYNTHESIS
```

The Writer produces the user-facing report but **does not yet consume** `DecisionSynthesis`.

## Query Routing

`services/query_router.py` classifies each query:

| Route | Typical use | Budget highlights |
|-------|-------------|-------------------|
| `simple_fact` | Capital, winner, revenue, CEO, date | 1 search, ~3 sources, 0 critic iterations |
| `standard` | Summaries, multi-part facts | 3 searches, ~8 sources, 1 iteration |
| `deep` | Comparisons, strategy, causal | 5 searches, more sources, 2 iterations |

Output: `QueryClassification` stored in `state.query_classification` with a `ResearchBudget` (search caps, claim depth, evidence caps, `max_iterations`).

## Node Responsibilities

| Node | File | Role |
|------|------|------|
| Router | `nodes/router.py` | Classify query; set route and budget |
| Fast Path | `nodes/fast_path.py` | SIMPLE_FACT answer or escalation |
| Decision Framer | `nodes/decision_framer.py` | Structure decision context (Phase 3A); fail-open |
| Planner | `nodes/planner.py` | Sub-queries, search terms, domain filters |
| Researcher | `nodes/researcher.py` | Tavily search, spam filter, source normalization |
| Evidence Extractor | `nodes/evidence_extractor.py` | Verbatim passages + integrity validation |
| Claim Extractor | `nodes/claim_extractor.py` | Atomic claims, materiality, origin SUPPORTS links |
| Claim Verifier | `nodes/claim_verifier.py` | Cross-source SUPPORTS / CONTRADICTS / QUALIFIES |
| Critic | `nodes/critic.py` | Evidence quality; refinement loop guard |
| Knowledge State | `nodes/knowledge_state.py` | Deterministic epistemic buckets (Phase 2D) |
| Option Evaluator | `nodes/option_evaluator.py` | Option×criterion assessments (Phase 3B) |
| Decision Synthesizer | `nodes/decision_synthesizer.py` | Recommendation synthesis (Phase 3C) |
| Writer | `nodes/writer.py` | Evidence-grounded report with `[E#]` citations |

## Evidence Layer (Phase 2A)

- **Source normalization**: `services/source_normalizer.py` — Tavily hits → `Source` entities
- **Extraction**: `services/evidence_pipeline.py`, `services/evidence_extractor.py`
- **Validation**: `services/evidence_validator.py` — span integrity (exact / normalized / fuzzy match)
- **Output**: `validated_evidence[]` with provenance (`source_id`, locator, match type)

## Claim Layer (Phase 2B)

- **Pipeline**: `services/claim_pipeline.py` — LLM candidates → deterministic support check → relevance → batch validation → dedup
- **Materiality**: `services/claim_relevance.py` — filters to `material_claims`
- **Origin links**: `claim_evidence_relations` with `SUPPORTS` only at extraction time

### Claim–evidence relationships

| Relationship | Meaning |
|--------------|---------|
| `supports` | Evidence substantiates the claim |
| `contradicts` | Evidence conflicts with the claim |
| `qualifies` | Evidence narrows or conditions the claim |
| `contextualizes` | Reserved; not used in current verification aggregation |

## Cross-Source Verification (Phase 2C)

`services/claim_verification.py` — **material claims only**.

1. Preserves origin `SUPPORTS` from claim extraction
2. Selects cross-source evidence (excludes origin evidence IDs and origin publisher domains)
3. Classifies additional pairs deterministically, then batched LLM for ambiguous cases
4. Aggregates to `VerificationResult` per claim

### Verification statuses

| Status | Typical meaning |
|--------|-----------------|
| `supported` | 2+ independent publisher domains support; no contradicts |
| `partially_supported` | Single-source or qualified support |
| `uncertain` | Credible support and contradict coexist |
| `contradicted` | Contradict without support |
| `insufficient_evidence` | No meaningful relevant evidence links |
| `unverifiable` | Claim type not verifiable from documentary evidence |

**Publisher-domain independence** uses `source.metadata["domain"]`. Same domain on different URLs counts as one publisher — an approximation, not true source lineage.

## Knowledge State (Phase 2D)

`services/knowledge_state.py` — **derived layer, no LLM**.

Runs only on **final Critic exit** (sufficient or max iterations), before Writer. **Not** derived during researcher refinement loops.

### Inputs

- `material_claims`
- `verification_results`
- `claim_evidence_relations`
- `critique.unsupported_areas` (information gaps only)

### Buckets

| Bucket | Derivation |
|--------|------------|
| `known` | `supported` + `high` confidence → persisted `knowledge_category = known` |
| `likely` | `partially_supported` → `likely` |
| `disputed` | `uncertain` → `disputed` |
| `unknown` | `insufficient_evidence` → `unknown` |
| `contradicted` | `contradicted` status; `knowledge_category` left `null` |
| `unverifiable` | `unverifiable` status; `knowledge_category` left `null` |
| `information_gaps` | Critic `unsupported_areas` as gap hints (`source = critic_unsupported_area`) |

Entries reference `claim_id`, `verification_id`, `relation_ids`, `evidence_ids` — **no claim text duplication**.

Orphan material claims (no matching `VerificationResult`) are **not** classified as UNKNOWN; they increment `orphan_material_claims` in metrics.

### Fast-path limitation

Successful SIMPLE_FACT runs bypass Claim Verifier and **do not** receive Knowledge State or decision artifacts in the current MVP.

## Decision Framing (Phase 3A)

`services/decision_framing.py`, `nodes/decision_framer.py` — structures decision context from query + research artifacts.

**Runs on:** STANDARD / DEEP only. SIMPLE_FACT does not run Decision Framing. Framing is **fail-open** (pipeline continues if framing fails).

### DecisionFrame fields

| Field | Notes |
|-------|-------|
| `decision` | Core decision question |
| `decision_type` | Classification of decision shape |
| `options` | Each with label + provenance (`explicit` / `implied`) |
| `criteria` | Each with label + provenance (`explicit` / `inferred`) + priority (`primary` / `standard`) |
| `constraints` | Explicit-only hard constraints |
| `time_horizon` | Decision time scope |
| `missing_decision_context` | Gaps in framing |
| `explicit_assumptions` | Explicit-only assumptions |

### Semantics

- Decision orientation is **separate** from SIMPLE / STANDARD / DEEP complexity
- **Inferred criteria cannot be `primary`** (sanitized deterministically)
- Constraints and assumptions are **explicit-only**
- **No scoring or recommendation** in 3A

## Option Evaluation (Phase 3B)

`services/option_evaluation.py`, `nodes/option_evaluator.py`

**Input:** `DecisionFrame` + `KnowledgeState`

**Output:** `OptionEvaluation` — full option×criterion matrix

### Assessment vocabulary

`favorable` · `unfavorable` · `mixed` · `neutral` · `uncertain` · `insufficient_information`

### Knowledge coverage

`grounded` · `partial` · `insufficient`

Each criterion evaluation preserves: option label + provenance, criterion label + provenance, criterion priority, assessment, knowledge coverage, claim IDs, verification IDs, knowledge categories, brief reason.

### Invariants

| Rule | Behavior |
|------|----------|
| Claim lineage | Every substantive evaluation has claim lineage |
| No external knowledge | No model knowledge beyond trusted Knowledge State |
| No recommendation / scoring | Assessments only |
| Contradicted-only knowledge | → `insufficient_information` |
| Disputed knowledge | Cannot produce confident directional support |
| Unknown knowledge | Cannot become directional evidence |
| Matrix completeness | `expected pairs = options × criteria` |
| Empty options | No pseudo-option; **Option Evaluation skipped** when `DecisionFrame` has no concrete options |

## Decision Synthesis (Phase 3C)

`services/decision_synthesis.py`, `nodes/decision_synthesizer.py`

**Input:** `DecisionFrame` + `OptionEvaluation` + trusted `KnowledgeState` / material claims (for hard-constraint evaluation)

**Output:** `DecisionSynthesis`

### Recommendation statuses

`recommend` · `tentative_recommendation` · `insufficient_basis`

### DecisionSynthesis fields

- `recommendation_status`, `recommended_option`, `rationale`
- `supporting_criteria`, `limiting_criteria`
- `constraint_assessments` (every option×constraint pair)
- `key_uncertainties`, `decision_limitations`, `critical_missing_context`
- `assumptions_relied_on`
- `what_would_change` / change conditions

### Decision hierarchy (qualitative — no numeric weights)

1. Hard constraints
2. Explicit primary criteria
3. Explicit standard criteria
4. Inferred criteria

### Hard-constraint semantics

All `DecisionFrame` constraints are **hard constraints**. Every option×constraint pair must be assessed.

| Compliance | Meaning |
|------------|---------|
| `satisfied` | Evidence supports compliance |
| `violated` | Evidence supports violation |
| `not_established` | Insufficient evidence either way |

| Rule | Behavior |
|------|----------|
| Violated constraint | Prevents recommendation of that option |
| Not-established constraint | Prevents full `recommend` |
| Missing matrix pairs | Forces `insufficient_basis` |
| Constraint evidence | May use full trusted KnowledgeState / material-claim catalog (not limited to 3B claim IDs) |
| Absence of evidence | Is **not** violation |

### Matrix completeness

- **Option Evaluation:** `expected pairs = options × criteria`
- **Decision Synthesis constraints:** `expected pairs = options × constraints`

Incomplete matrices must **never** silently produce full recommendations.

### What would change the recommendation

Change conditions may reference existing options, criteria, constraints, explicit assumptions, missing decision context, and claim IDs. No invented thresholds or new factual claims.

## Epistemic Safety

Knowledge State buckets and their downstream decision behavior:

| Bucket | Decision behavior |
|--------|-------------------|
| `known` | Usable grounded support |
| `likely` | Usable with reduced confidence |
| `disputed` | Cannot support confident directional assessments (≠ contradicted) |
| `unknown` | Cannot become directional evidence |
| `contradicted` | Not usable support; contradicted-only → `insufficient_information` in 3B |
| `unverifiable` | Not usable as directional support |

**Distinctions:**

- `disputed` ≠ `contradicted`
- `uncertain` (assessment) ≠ `insufficient_information` (assessment)
- Insufficient evidence is a legitimate decision outcome (`insufficient_basis`)
- Validators may preserve or **downgrade** recommendation strength
- Validators must **never** upgrade `tentative_recommendation` → `recommend`

## AgentState

Defined in `state.py` as a LangGraph `TypedDict`. Key fields:

| Area | Fields |
|------|--------|
| Routing | `query_classification`, `fast_path_metrics`, `escalate_to_standard`, `escalated_from_fast_path` |
| Artifacts | `normalized_sources`, `validated_evidence`, `validated_claims`, `material_claims`, `claim_evidence_relations`, `verification_results`, `knowledge_state` |
| Decision | `decision_frame`, `decision_frame_metrics`, `option_evaluation`, `option_evaluation_metrics`, `decision_synthesis`, `decision_synthesis_metrics` |
| Control | `critique`, `iteration_count`, `current_node`, `research_sufficient` |
| Output | `final_report` |
| Metrics | `evidence_metrics`, `claim_metrics`, `verification_metrics`, `cost_metrics`, `report_metrics` |

Domain entities live in `domain/models.py` (`Source`, `Evidence`, `Claim`, `ClaimEvidenceRelation`, `VerificationResult`, etc.).

## Persistence (Supabase)

Optional — controlled by `SUPABASE_URL` / `SUPABASE_KEY`. When disabled, in-memory negative IDs are used.

### Tables

| Table | Purpose |
|-------|---------|
| `research_runs` | Run lifecycle, metadata snapshot (includes `knowledge_state`) |
| `sources` | Normalized search hits |
| `evidence` | Validated passages |
| `claims` | Atomic claims |
| `claim_evidence` | Claim ↔ evidence relationships |
| `verifications` | Per-claim verification + `knowledge_category` (known/likely/disputed/unknown) |
| `research_reports` | Final reports (legacy + optional `research_run_id` link) |

Legacy cache tables (`research_plans`, etc.) remain in `db/schema.sql`. Evidence foundation tables are in `db/migrations/001_evidence_foundation.sql`.

Repositories: `db/evidence_repositories.py`, `db/repository.py`.

**Note:** There is no full caching system for evidence or claims. Plan caching (`ENABLE_CACHING`) applies to research plans only.

## Observability

- **LangSmith**: `utils/observability.py` — `trace_llm_call` spans per node/operation
- **Structured logging**: JSON logs with PII redaction (`utils/pii_redaction.py`)
- **Per-stage metrics**: `*_metrics` dicts on state; `cost_metrics` aggregates run-level counters including knowledge-state counts

## Streaming API

`api.py` uses a queue-based NDJSON event generator:

- LangGraph runs in a background task
- Events: `log`, `result`, `error`, `done`
- Client disconnects do not kill the graph run (LangSmith traces finalize)

See [README.md](../README.md) for deployment and entry points.

## Known Limitations (Current)

- **Writer does not consume `DecisionSynthesis`** — user-facing report does not present structured recommendation
- Decision artifacts inspectable via state / metadata / validation scripts only
- **Writer** does not consume `verification_results` or `knowledge_state`
- **Critic** does not consume verification results
- **Planner / research** not guided by `DecisionFrame`
- Successful fast-path runs have no Knowledge State or decision artifacts
- No persistent decision workspace, monitoring, change detection, or automatic re-evaluation
- No actions / execution layer; no numerical utility or weighting
- Constraint mapping relies on constrained LLM semantic judgment against trusted claims
- Full KnowledgeState claim catalog used (no relevance-aware truncation)
- Publisher-domain independence is approximate
- Cross-domain `supported` / `known` depends on retrieval yielding diverse sources

## Testing

```bash
uv run pytest   # 244 tests, no live LLM for core logic
```

Phase validation scripts (isolated live LLM — manual inspection):

| Script | Phase |
|--------|-------|
| `scripts/validate_phase_2a5_e2e.py` | Evidence extraction |
| `scripts/validate_phase_2b_e2e.py` | Claim extraction |
| `scripts/validate_phase_2b7_e2e.py` | SIMPLE_FACT fast path |
| `scripts/validate_phase_2c_e2e.py` | Cross-source verification |
| `scripts/validate_phase_2d_e2e.py` | Knowledge State buckets |
| `scripts/validate_phase_3a_e2e.py` | Decision Framing |
| `scripts/validate_phase_3b_live.py` | Option Evaluation |
| `scripts/validate_phase_3c_live.py` | Decision Synthesis |

Isolated live validation has been completed for Decision Framing, Option Evaluation, and Decision Synthesis. Full end-to-end production evaluation of decision artifacts in the Writer/UI is **not** implemented.
