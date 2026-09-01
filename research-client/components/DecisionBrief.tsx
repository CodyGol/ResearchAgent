"use client";

import { useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  HelpCircle,
  X,
} from "lucide-react";
import {
  buildComparisonMatrix,
  buildConstraintMatrix,
  buildLimitationsList,
  buildUncertaintyList,
  type ClaimLineageEntry,
  type ConstraintCompliance,
  type CriterionEvaluationPayload,
  type DecisionBriefPayload,
  type DecisionSynthesisPayload,
  formatAssessment,
  formatCompliance,
  formatCoverage,
  formatRecommendationStatus,
  getClaimsForIds,
  findCriterionReason,
} from "../lib/decision-brief";

interface DecisionBriefProps {
  brief: DecisionBriefPayload;
}

function ComplianceIcon({ compliance }: { compliance: ConstraintCompliance }) {
  if (compliance === "satisfied") {
    return <Check className="inline w-3.5 h-3.5 mr-1 shrink-0" aria-hidden />;
  }
  if (compliance === "violated") {
    return <X className="inline w-3.5 h-3.5 mr-1 shrink-0" aria-hidden />;
  }
  return <HelpCircle className="inline w-3.5 h-3.5 mr-1 shrink-0" aria-hidden />;
}

function complianceClass(compliance: ConstraintCompliance): string {
  if (compliance === "violated") return "text-red-400 border-red-700";
  if (compliance === "not_established") return "text-yellow-400 border-yellow-800";
  return "text-green-400 border-green-800";
}

function ClaimEvidenceList({
  claims,
}: {
  claims: ClaimLineageEntry[];
}) {
  if (!claims.length) return null;
  return (
    <ul className="mt-2 space-y-2 text-sm text-green-600 list-none pl-0">
      {claims.map((claim, idx) => (
        <li key={idx} className="border-l-2 border-green-800 pl-3">
          <div className="text-green-400">{claim.text}</div>
          {(claim.knowledge_category || claim.verification_status) && (
            <div className="text-xs text-green-700 mt-0.5">
              {[claim.knowledge_category, claim.verification_status]
                .filter(Boolean)
                .join(" · ")}
            </div>
          )}
          {claim.evidence?.map((ev, evIdx) => (
            <div key={evIdx} className="mt-1 text-xs">
              {ev.snippet && (
                <div className="italic text-green-700">&ldquo;{ev.snippet}&rdquo;</div>
              )}
              {ev.source_url ? (
                <a
                  href={ev.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-green-400"
                >
                  {ev.source_title || ev.source_url}
                </a>
              ) : (
                ev.source_title && <span>{ev.source_title}</span>
              )}
            </div>
          ))}
        </li>
      ))}
    </ul>
  );
}

function ExpandableCriterionRow({
  rowLabel,
  rowMeta,
  cells,
  optionLabels,
  claimLineage,
  onToggle,
  expanded,
}: {
  rowLabel: string;
  rowMeta: string;
  cells: Record<string, { assessment: string; knowledge_coverage: string; claim_ids: number[]; reason: string } | null>;
  optionLabels: string[];
  claimLineage: Record<string, ClaimLineageEntry>;
  expanded: boolean;
  onToggle: () => void;
}) {
  const allClaimIds = [...new Set(optionLabels.flatMap((opt) => cells[opt]?.claim_ids ?? []))];

  return (
    <>
      <tr className="border-b border-green-900">
        <td className="py-2 pr-4 align-top">
          <button
            type="button"
            onClick={onToggle}
            className="flex items-start gap-1 text-left hover:text-green-300"
            aria-expanded={expanded}
          >
            {expanded ? (
              <ChevronDown className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            )}
            <span>
              {rowLabel}
              {rowMeta && (
                <span className="text-green-700 text-xs ml-1">{rowMeta}</span>
              )}
            </span>
          </button>
        </td>
        {optionLabels.map((opt) => {
          const cell = cells[opt];
          if (!cell) {
            return (
              <td key={opt} className="py-2 px-2 align-top text-green-700">
                —
              </td>
            );
          }
          return (
            <td key={opt} className="py-2 px-2 align-top">
              <div>{formatAssessment(cell.assessment as never)}</div>
              <div className="text-xs text-green-700">
                {formatCoverage(cell.knowledge_coverage as never)}
              </div>
            </td>
          );
        })}
      </tr>
      {expanded && (
        <tr className="border-b border-green-900 bg-green-950/30">
          <td colSpan={optionLabels.length + 1} className="py-3 px-2">
            {optionLabels.map((opt) => {
              const cell = cells[opt];
              if (!cell) return null;
              return (
                <div key={opt} className="mb-3 last:mb-0">
                  <div className="text-green-500 text-xs font-semibold mb-1">
                    {opt}
                  </div>
                  {cell.reason && (
                    <div className="text-sm text-green-600 mb-1">{cell.reason}</div>
                  )}
                  <ClaimEvidenceList
                    claims={getClaimsForIds(claimLineage, cell.claim_ids)}
                  />
                </div>
              );
            })}
            {!allClaimIds.length && (
              <div className="text-sm text-green-700">No linked claims.</div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function RecommendationHeader({ synthesis }: { synthesis: DecisionSynthesisPayload }) {
  const status = synthesis.recommendation_status;
  const isInsufficient = status === "insufficient_basis";

  return (
    <section className="border border-green-700 p-4">
      <div className="text-green-500 font-semibold mb-3 tracking-wide">
        DECISION BRIEF
      </div>
      <div className="text-xs text-green-700 mb-2 uppercase tracking-wide">
        Recommendation
      </div>
      {isInsufficient ? (
        <>
          <div className="text-lg text-green-300 font-semibold">
            {formatRecommendationStatus(status)}
          </div>
          <p className="mt-2 text-green-400">
            There isn&apos;t enough evidence to make a responsible recommendation yet.
          </p>
        </>
      ) : (
        <>
          {synthesis.recommended_option && (
            <div className="text-xl text-green-300 font-semibold">
              {synthesis.recommended_option}
            </div>
          )}
          <div className="text-sm text-green-500 mt-1">
            {formatRecommendationStatus(status)}
          </div>
        </>
      )}
      {synthesis.decision && (
        <p className="mt-3 text-sm text-green-600">{synthesis.decision}</p>
      )}
    </section>
  );
}

function WhySection({
  synthesis,
  optionEvaluation,
  claimLineage,
}: {
  synthesis: DecisionSynthesisPayload;
  optionEvaluation: DecisionBriefPayload["option_evaluation"];
  claimLineage: Record<string, ClaimLineageEntry>;
}) {
  const hasCriteria =
    synthesis.supporting_criteria.length > 0 ||
    synthesis.limiting_criteria.length > 0;

  if (!synthesis.rationale && !hasCriteria) return null;

  return (
    <section>
      <h3 className="text-green-500 font-semibold mb-2">WHY</h3>
      {synthesis.rationale && (
        <p className="text-green-400 text-sm mb-4 leading-relaxed">
          {synthesis.rationale}
        </p>
      )}
      {synthesis.supporting_criteria.length > 0 && (
        <div className="mb-3">
          <div className="text-xs text-green-600 mb-1 uppercase">Supporting</div>
          <ul className="space-y-2 text-sm">
            {synthesis.supporting_criteria.map((ref, i) => (
              <li key={i} className="border-l-2 border-green-700 pl-3">
                <div className="text-green-400">
                  {ref.criterion_label}
                  {ref.criterion_priority === "primary" && (
                    <span className="text-green-600 text-xs ml-1">· Primary</span>
                  )}
                  <span className="text-green-700 text-xs ml-1">
                    ({ref.option_label})
                  </span>
                </div>
                <div className="text-green-600 text-xs">
                  {formatAssessment(ref.assessment)} ·{" "}
                  {formatCoverage(ref.knowledge_coverage)}
                </div>
                {findCriterionReason(
                  optionEvaluation,
                  ref.option_label,
                  ref.criterion_label
                ) && (
                  <div className="text-green-700 text-xs mt-0.5">
                    {findCriterionReason(
                      optionEvaluation,
                      ref.option_label,
                      ref.criterion_label
                    )}
                  </div>
                )}
                <ClaimEvidenceList
                  claims={getClaimsForIds(claimLineage, ref.claim_ids)}
                />
              </li>
            ))}
          </ul>
        </div>
      )}
      {synthesis.limiting_criteria.length > 0 && (
        <div>
          <div className="text-xs text-green-600 mb-1 uppercase">Limiting</div>
          <ul className="space-y-2 text-sm">
            {synthesis.limiting_criteria.map((ref, i) => (
              <li key={i} className="border-l-2 border-green-800 pl-3">
                <div className="text-green-400">
                  {ref.criterion_label}
                  {ref.criterion_priority === "primary" && (
                    <span className="text-green-600 text-xs ml-1">· Primary</span>
                  )}
                  <span className="text-green-700 text-xs ml-1">
                    ({ref.option_label})
                  </span>
                </div>
                <div className="text-green-600 text-xs">
                  {formatAssessment(ref.assessment)} ·{" "}
                  {formatCoverage(ref.knowledge_coverage)}
                </div>
                {findCriterionReason(
                  optionEvaluation,
                  ref.option_label,
                  ref.criterion_label
                ) && (
                  <div className="text-green-700 text-xs mt-0.5">
                    {findCriterionReason(
                      optionEvaluation,
                      ref.option_label,
                      ref.criterion_label
                    )}
                  </div>
                )}
                <ClaimEvidenceList
                  claims={getClaimsForIds(claimLineage, ref.claim_ids)}
                />
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

export function DecisionBrief({ brief }: DecisionBriefProps) {
  const { decision_frame, option_evaluation, decision_synthesis, claim_lineage } =
    brief;
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  const matrix = buildComparisonMatrix(decision_frame, option_evaluation);
  const constraintMatrix = buildConstraintMatrix(decision_synthesis);
  const uncertainties = buildUncertaintyList(decision_synthesis);
  const limitations = buildLimitationsList(decision_synthesis);
  const assumptions = decision_synthesis.assumptions_relied_on ?? [];

  const toggleRow = (label: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  };

  return (
    <div className="space-y-6 border border-green-800 p-4 bg-black/40">
      <RecommendationHeader synthesis={decision_synthesis} />

      <WhySection
        synthesis={decision_synthesis}
        optionEvaluation={option_evaluation}
        claimLineage={claim_lineage}
      />

      {matrix && matrix.rows.length > 0 && (
        <section>
          <h3 className="text-green-500 font-semibold mb-2">OPTION COMPARISON</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[480px]">
              <thead>
                <tr className="border-b border-green-700 text-green-600 text-left">
                  <th className="py-2 pr-4 font-normal">Criterion</th>
                  {matrix.optionLabels.map((opt) => (
                    <th key={opt} className="py-2 px-2 font-normal">
                      {opt}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.rows.map((row) => {
                  const meta =
                    row.rowKind === "explicit_primary"
                      ? "· Primary"
                      : row.rowKind === "inferred"
                        ? "· Inferred"
                        : "";
                  const cells: Record<string, CriterionEvaluationPayload | null> = {};
                  for (const [opt, cell] of Object.entries(row.cells)) {
                    cells[opt] = cell;
                  }
                  return (
                    <ExpandableCriterionRow
                      key={row.criterionLabel}
                      rowLabel={row.criterionLabel}
                      rowMeta={meta}
                      cells={cells}
                      optionLabels={matrix.optionLabels}
                      claimLineage={claim_lineage}
                      expanded={expandedRows.has(row.criterionLabel)}
                      onToggle={() => toggleRow(row.criterionLabel)}
                    />
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {constraintMatrix && constraintMatrix.rows.length > 0 && (
        <section>
          <h3 className="text-green-500 font-semibold mb-2">HARD CONSTRAINTS</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[480px]">
              <thead>
                <tr className="border-b border-green-700 text-green-600 text-left">
                  <th className="py-2 pr-4 font-normal">Constraint</th>
                  {constraintMatrix.optionLabels.map((opt) => (
                    <th key={opt} className="py-2 px-2 font-normal">
                      {opt}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {constraintMatrix.rows.map((row) => (
                  <tr key={row.constraint} className="border-b border-green-900">
                    <td className="py-2 pr-4 align-top">{row.constraint}</td>
                    {constraintMatrix.optionLabels.map((opt) => {
                      const cell = row.cells[opt];
                      if (!cell) {
                        return (
                          <td key={opt} className="py-2 px-2 text-green-700">
                            —
                          </td>
                        );
                      }
                      return (
                        <td
                          key={opt}
                          className={`py-2 px-2 align-top border-l border-green-900 ${complianceClass(cell.compliance)}`}
                        >
                          <div className="flex items-start">
                            <ComplianceIcon compliance={cell.compliance} />
                            <span>{formatCompliance(cell.compliance)}</span>
                          </div>
                          {cell.reason && (
                            <div className="text-xs text-green-700 mt-1 ml-5">
                              {cell.reason}
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {(uncertainties.length > 0 || limitations.length > 0) && (
        <section>
          <h3 className="text-green-500 font-semibold mb-2 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" aria-hidden />
            KEY UNCERTAINTIES
          </h3>
          <ul className="list-disc list-inside text-sm text-green-600 space-y-1">
            {uncertainties.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
          {limitations.length > 0 && (
            <div className="mt-3">
              <div className="text-xs text-green-700 uppercase mb-1">
                Limitations
              </div>
              <ul className="list-disc list-inside text-sm text-green-700 space-y-1">
                {limitations.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {assumptions.length > 0 && (
        <section>
          <h3 className="text-green-500 font-semibold mb-2">ASSUMPTIONS</h3>
          <ul className="list-disc list-inside text-sm text-green-600 space-y-1">
            {assumptions.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>
      )}

      {decision_synthesis.change_conditions.length > 0 && (
        <section>
          <h3 className="text-green-500 font-semibold mb-2">
            WHAT WOULD CHANGE THIS RECOMMENDATION
          </h3>
          <ul className="space-y-3 text-sm">
            {decision_synthesis.change_conditions.map((cc, i) => {
              const meta = [
                cc.related_option_label && `Option: ${cc.related_option_label}`,
                cc.related_criterion_label &&
                  `Criterion: ${cc.related_criterion_label}`,
                cc.related_constraint && `Constraint: ${cc.related_constraint}`,
                cc.related_assumption && `Assumption: ${cc.related_assumption}`,
                cc.related_missing_context &&
                  `Missing: ${cc.related_missing_context}`,
              ].filter(Boolean);
              return (
                <li key={i} className="border-l-2 border-green-800 pl-3 text-green-400">
                  <div>{cc.description}</div>
                  {meta.length > 0 && (
                    <div className="text-xs text-green-700 mt-1">
                      {meta.join(" · ")}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
