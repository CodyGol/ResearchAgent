"""Per-run cost and latency observability."""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTiming:
    """Timing for a single pipeline stage."""

    stage: str
    duration_ms: float = 0.0
    model_calls: int = 0


@dataclass
class CostMetrics:
    """Aggregated cost/latency observability for a research run."""

    complexity_class: str = ""
    search_queries_executed: int = 0
    raw_sources: int = 0
    canonical_sources: int = 0
    evidence_items: int = 0
    candidate_claims: int = 0
    material_claims: int = 0
    deterministic_rejects: int = 0
    relevance_rejects: int = 0
    llm_validation_calls: int = 0
    validation_batches: int = 0
    model_calls_by_stage: dict[str, int] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    short_circuited: bool = False
    short_circuit_reason: str = ""

    _stage_starts: dict[str, float] = field(default_factory=dict, repr=False)

    def start_stage(self, stage: str) -> None:
        self._stage_starts[stage] = time.monotonic()

    def end_stage(self, stage: str, *, model_calls: int = 0) -> None:
        start = self._stage_starts.pop(stage, None)
        if start is not None:
            elapsed = (time.monotonic() - start) * 1000
            self.stage_timings_ms[stage] = (
                self.stage_timings_ms.get(stage, 0.0) + elapsed
            )
        if model_calls:
            self.model_calls_by_stage[stage] = (
                self.model_calls_by_stage.get(stage, 0) + model_calls
            )

    def record_model_call(self, stage: str) -> None:
        self.model_calls_by_stage[stage] = self.model_calls_by_stage.get(stage, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        total_model_calls = sum(self.model_calls_by_stage.values())
        return {
            "complexity_class": self.complexity_class,
            "search_queries_executed": self.search_queries_executed,
            "raw_sources": self.raw_sources,
            "canonical_sources": self.canonical_sources,
            "evidence_items": self.evidence_items,
            "candidate_claims": self.candidate_claims,
            "material_claims": self.material_claims,
            "deterministic_rejects": self.deterministic_rejects,
            "relevance_rejects": self.relevance_rejects,
            "llm_validation_calls": self.llm_validation_calls,
            "validation_batches": self.validation_batches,
            "total_model_calls": total_model_calls,
            "model_calls_by_stage": dict(self.model_calls_by_stage),
            "stage_timings_ms": dict(self.stage_timings_ms),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "short_circuited": self.short_circuited,
            "short_circuit_reason": self.short_circuit_reason,
        }
