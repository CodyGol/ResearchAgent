/** Decision Brief presentation types and helpers (no LLM). */

export type RecommendationStatus =
  | "recommend"
  | "tentative_recommendation"
  | "insufficient_basis";

export type CriterionAssessment =
  | "favorable"
  | "unfavorable"
  | "mixed"
  | "neutral"
  | "uncertain"
  | "insufficient_information";

export type KnowledgeCoverage = "grounded" | "partial" | "insufficient";

export type ConstraintCompliance = "satisfied" | "violated" | "not_established";

export interface ClaimEvidenceLineage {
  display_id: string | null;
  snippet: string | null;
  source_title: string | null;
  source_url: string | null;
}

export interface ClaimLineageEntry {
  text: string;
  knowledge_category: string | null;
  verification_status: string | null;
  evidence: ClaimEvidenceLineage[];
}

export interface DecisionFramePayload {
  decision: string;
  decision_type?: string;
  options?: Array<{ label: string; origin: string }>;
  criteria?: Array<{
    label: string;
    origin: string;
    priority: "primary" | "standard";
  }>;
  constraints?: string[];
  missing_decision_context?: string[];
  explicit_assumptions?: string[];
}

export interface CriterionEvaluationPayload {
  criterion_label: string;
  criterion_origin: "explicit" | "inferred";
  criterion_priority: "primary" | "standard";
  assessment: CriterionAssessment;
  knowledge_coverage: KnowledgeCoverage;
  claim_ids: number[];
  reason: string;
}

export interface OptionEvaluationEntryPayload {
  option_label: string;
  option_origin: string;
  criteria_evaluations: CriterionEvaluationPayload[];
}

export interface OptionEvaluationPayload {
  decision: string;
  option_evaluations: OptionEvaluationEntryPayload[];
  decision_limitations?: string[];
  constraints?: string[];
}

export interface CriterionReferencePayload {
  option_label: string;
  criterion_label: string;
  criterion_origin: string;
  criterion_priority: string;
  assessment: CriterionAssessment;
  knowledge_coverage: KnowledgeCoverage;
  claim_ids: number[];
}

export interface ConstraintAssessmentPayload {
  option_label: string;
  constraint: string;
  compliance: ConstraintCompliance;
  claim_ids: number[];
  reason: string;
}

export interface ChangeConditionPayload {
  description: string;
  change_type: string;
  related_option_label?: string | null;
  related_criterion_label?: string | null;
  related_constraint?: string | null;
  related_assumption?: string | null;
  related_missing_context?: string | null;
  related_claim_ids?: number[];
}

export interface DecisionSynthesisPayload {
  decision: string;
  recommendation_status: RecommendationStatus;
  recommended_option: string | null;
  rationale: string;
  supporting_criteria: CriterionReferencePayload[];
  limiting_criteria: CriterionReferencePayload[];
  constraint_assessments: ConstraintAssessmentPayload[];
  key_uncertainties: string[];
  decision_limitations: string[];
  critical_missing_context: string[];
  assumptions_relied_on: string[];
  change_conditions: ChangeConditionPayload[];
}

export interface DecisionBriefPayload {
  decision_frame: DecisionFramePayload | null;
  option_evaluation: OptionEvaluationPayload | null;
  decision_synthesis: DecisionSynthesisPayload;
  claim_lineage: Record<string, ClaimLineageEntry>;
}

const VALID_STATUSES: RecommendationStatus[] = [
  "recommend",
  "tentative_recommendation",
  "insufficient_basis",
];

export function shouldShowDecisionBrief(
  brief: DecisionBriefPayload | null | undefined
): brief is DecisionBriefPayload {
  if (!brief?.decision_synthesis) return false;
  const status = brief.decision_synthesis.recommendation_status;
  return VALID_STATUSES.includes(status);
}

export function formatRecommendationStatus(status: RecommendationStatus): string {
  switch (status) {
    case "recommend":
      return "Recommended";
    case "tentative_recommendation":
      return "Tentative recommendation";
    case "insufficient_basis":
      return "Insufficient basis";
    default:
      return status;
  }
}

export function formatAssessment(assessment: CriterionAssessment): string {
  const labels: Record<CriterionAssessment, string> = {
    favorable: "Favorable",
    unfavorable: "Unfavorable",
    mixed: "Mixed",
    neutral: "Neutral",
    uncertain: "Uncertain",
    insufficient_information: "Insufficient information",
  };
  return labels[assessment] ?? assessment;
}

export function formatCoverage(coverage: KnowledgeCoverage): string {
  const labels: Record<KnowledgeCoverage, string> = {
    grounded: "Grounded",
    partial: "Partial",
    insufficient: "Insufficient",
  };
  return labels[coverage] ?? coverage;
}

export function formatCompliance(compliance: ConstraintCompliance): string {
  const labels: Record<ConstraintCompliance, string> = {
    satisfied: "Satisfied",
    violated: "Violated",
    not_established: "Not established",
  };
  return labels[compliance] ?? compliance;
}

export type CriterionRowKind = "explicit_primary" | "explicit_standard" | "inferred";

export interface ComparisonMatrixRow {
  criterionLabel: string;
  criterionOrigin: "explicit" | "inferred";
  criterionPriority: "primary" | "standard";
  rowKind: CriterionRowKind;
  cells: Record<string, CriterionEvaluationPayload | null>;
}

export interface ComparisonMatrix {
  optionLabels: string[];
  rows: ComparisonMatrixRow[];
}

function criterionSortKey(
  label: string,
  frame: DecisionFramePayload | null
): { order: number; origin: "explicit" | "inferred"; priority: "primary" | "standard" } {
  const criteria = frame?.criteria ?? [];
  const match = criteria.find((c) => c.label === label);
  const origin: "explicit" | "inferred" =
    match?.origin === "inferred" ? "inferred" : "explicit";
  const priority: "primary" | "standard" =
    match?.priority === "primary" ? "primary" : "standard";
  let order = 2;
  if (origin === "explicit" && priority === "primary") order = 0;
  else if (origin === "explicit") order = 1;
  return { order, origin, priority };
}

export function buildComparisonMatrix(
  frame: DecisionFramePayload | null,
  optionEvaluation: OptionEvaluationPayload | null
): ComparisonMatrix | null {
  if (!optionEvaluation?.option_evaluations?.length) return null;

  const optionLabels = optionEvaluation.option_evaluations.map((o) => o.option_label);
  const criterionLabels = new Set<string>();

  for (const opt of optionEvaluation.option_evaluations) {
    for (const ce of opt.criteria_evaluations) {
      criterionLabels.add(ce.criterion_label);
    }
  }

  const sortedLabels = [...criterionLabels].sort((a, b) => {
    const ka = criterionSortKey(a, frame);
    const kb = criterionSortKey(b, frame);
    if (ka.order !== kb.order) return ka.order - kb.order;
    return a.localeCompare(b);
  });

  const rows: ComparisonMatrixRow[] = sortedLabels.map((label) => {
    const key = criterionSortKey(label, frame);
    const rowKind: CriterionRowKind =
      key.origin === "inferred"
        ? "inferred"
        : key.priority === "primary"
          ? "explicit_primary"
          : "explicit_standard";

    const cells: Record<string, CriterionEvaluationPayload | null> = {};
    for (const opt of optionEvaluation.option_evaluations) {
      cells[opt.option_label] =
        opt.criteria_evaluations.find((ce) => ce.criterion_label === label) ?? null;
    }

    return {
      criterionLabel: label,
      criterionOrigin: key.origin,
      criterionPriority: key.priority,
      rowKind,
      cells,
    };
  });

  return { optionLabels, rows };
}

export interface ConstraintMatrixRow {
  constraint: string;
  cells: Record<string, ConstraintAssessmentPayload | null>;
}

export interface ConstraintMatrix {
  optionLabels: string[];
  rows: ConstraintMatrixRow[];
}

export function buildConstraintMatrix(
  synthesis: DecisionSynthesisPayload
): ConstraintMatrix | null {
  const assessments = synthesis.constraint_assessments ?? [];
  if (!assessments.length) return null;

  const optionLabels = [...new Set(assessments.map((a) => a.option_label))].sort();
  const constraints = [...new Set(assessments.map((a) => a.constraint))].sort();

  const rows: ConstraintMatrixRow[] = constraints.map((constraint) => {
    const cells: Record<string, ConstraintAssessmentPayload | null> = {};
    for (const opt of optionLabels) {
      cells[opt] =
        assessments.find(
          (a) => a.constraint === constraint && a.option_label === opt
        ) ?? null;
    }
    return { constraint, cells };
  });

  return { optionLabels, rows };
}

export function dedupeExactStrings(items: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of items) {
    const trimmed = item.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    result.push(trimmed);
  }
  return result;
}

export function buildUncertaintyList(synthesis: DecisionSynthesisPayload): string[] {
  return dedupeExactStrings([
    ...(synthesis.key_uncertainties ?? []),
    ...(synthesis.critical_missing_context ?? []),
  ]);
}

export function buildLimitationsList(synthesis: DecisionSynthesisPayload): string[] {
  const uncertainties = new Set(buildUncertaintyList(synthesis));
  return dedupeExactStrings(synthesis.decision_limitations ?? []).filter(
    (item) => !uncertainties.has(item)
  );
}

export function findCriterionReason(
  optionEvaluation: OptionEvaluationPayload | null,
  optionLabel: string,
  criterionLabel: string
): string | null {
  if (!optionEvaluation) return null;
  const opt = optionEvaluation.option_evaluations.find(
    (o) => o.option_label === optionLabel
  );
  const ce = opt?.criteria_evaluations.find(
    (c) => c.criterion_label === criterionLabel
  );
  return ce?.reason ?? null;
}

export function getClaimsForIds(
  claimLineage: Record<string, ClaimLineageEntry>,
  claimIds: number[]
): ClaimLineageEntry[] {
  const result: ClaimLineageEntry[] = [];
  const seen = new Set<string>();
  for (const id of claimIds) {
    const key = String(id);
    if (seen.has(key)) continue;
    const entry = claimLineage[key];
    if (entry) {
      seen.add(key);
      result.push(entry);
    }
  }
  return result;
}
