"""Writer node: Evidence-grounded report synthesis."""

import logging

from langchain_anthropic import ChatAnthropic

from config import settings
from services.answer_confidence import compute_confidence_assessment
from services.evidence_context import (
    assign_evidence_ids,
    evidence_ids_to_urls,
    extract_cited_evidence_ids,
    format_evidence_for_prompt,
)
from services.query_router import QueryComplexity
from services.report_consistency import run_consistency_checks
from services.writer_schemas import EvidenceGroundedWriterOutput
from state import AgentState, FinalReport
from utils.observability import trace_llm_call

logger = logging.getLogger(__name__)


_WRITER_SYSTEM_PROMPT = """You are an evidence-grounded research report writer.

VALIDATED EVIDENCE is your ONLY authoritative factual substrate.

RULES:
1. Every material factual statement MUST be supportable by the supplied evidence items.
2. Cite evidence using [E#] references inline (e.g. "Max Verstappen won the championship [E3][E7].").
3. Do NOT introduce facts, statistics, dates, or counts not grounded in the evidence.
4. Do NOT use raw search snippets — only validated evidence is provided.
5. Distinguish clearly:
   - Evidence-backed facts (cite with [E#])
   - Synthesis (combining multiple cited facts)
   - Interpretation (label as "Analysis:" or "Interpretation:" — never present as established fact)
6. Match report depth to question complexity:
   - Simple factual questions → short, direct answer (may be 1-3 paragraphs)
   - Complex research questions → longer structured report
7. Do NOT pad with unsupported detail to appear comprehensive.
8. If evidence is insufficient for a detail, omit it or state the gap explicitly.
9. List all evidence IDs you cite in evidence_ids_used.

You will NOT assign confidence — that is computed separately."""


async def writer_node(state: AgentState) -> AgentState:
    """
    Synthesize report from validated evidence only.

    Raw search results are NOT passed to the LLM.
    """
    plan = state.get("research_plan")
    critique = state.get("critique")
    evidence_list = assign_evidence_ids(state.get("validated_evidence") or [])
    sources = state.get("normalized_sources") or []
    evidence_metrics = state.get("evidence_metrics") or {}

    if not plan:
        state["error"] = "Missing research plan"
        state["current_node"] = "end"
        return state

    if not evidence_list:
        state["error"] = "No validated evidence available for report generation"
        state["current_node"] = "end"
        return state

    state["validated_evidence"] = evidence_list

    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.3,
    )

    evidence_block = format_evidence_for_prompt(evidence_list, sources)

    critique_block = ""
    if critique:
        critique_block = f"""
Critic assessment:
- Quality score: {critique.quality_score}
- Coverage: {critique.coverage}
- Source quality: {critique.source_quality}
- Source diversity: {critique.source_diversity}
- Temporal alignment: {critique.temporal_alignment}
- Potential conflicts: {', '.join(critique.potential_conflicts) or 'None identified'}
- Unsupported areas: {', '.join(critique.unsupported_areas) or 'None identified'}
- Issues: {', '.join(critique.issues) or 'None'}
"""

    with trace_llm_call("writer", "synthesize_evidence_grounded_report") as span:
        try:
            user_prompt = f"""Research question: {plan.query}

{critique_block}

VALIDATED EVIDENCE (your only factual source):
{evidence_block}

Write an evidence-grounded report. Use [E#] citations for every material fact.
Keep length appropriate to the question — do not over-generate."""

            span.set_input({
                "query": plan.query,
                "evidence_count": len(evidence_list),
                "quality_score": critique.quality_score if critique else None,
            })

            writer_output: EvidenceGroundedWriterOutput
            try:
                structured_llm = llm.with_structured_output(EvidenceGroundedWriterOutput)
                writer_output = await structured_llm.ainvoke([
                    {"role": "system", "content": _WRITER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ])
            except Exception:
                response = await llm.ainvoke([
                    {"role": "system", "content": _WRITER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ])
                content = response.content if hasattr(response, "content") else str(response)
                cited = extract_cited_evidence_ids(content)
                writer_output = EvidenceGroundedWriterOutput(
                    content=content,
                    evidence_ids_used=[f"E{i}" for i in cited] if cited else [],
                    factual_summary="",
                )

            # Resolve citations from evidence IDs
            cited_ids = writer_output.evidence_ids_used or extract_cited_evidence_ids(
                writer_output.content
            )
            # Normalize to numeric strings
            cited_numeric = []
            for eid in cited_ids:
                eid_clean = eid.replace("E", "").strip()
                if eid_clean.isdigit():
                    cited_numeric.append(eid_clean)
            if not cited_numeric:
                cited_numeric = extract_cited_evidence_ids(writer_output.content)

            citation_urls = evidence_ids_to_urls(cited_numeric, evidence_list, sources)

            # Consistency check
            evidence_texts = [ev.exact_text for ev in evidence_list]
            consistency = run_consistency_checks(writer_output.content, evidence_texts)

            report_content = writer_output.content
            if consistency.issues:
                logger.warning(
                    "Report consistency issues detected: %s", consistency.issues
                )
                # Append note about detected issues (do not silently ignore)
                issues_note = "\n\n---\n*Note: The following consistency concerns were detected in this report: "
                issues_note += "; ".join(consistency.issues)
                issues_note += ". Details should be verified against cited evidence.*"
                report_content += issues_note

            # Separated confidence: answer support vs research completeness
            classification = state.get("query_classification") or {}
            complexity_str = classification.get("complexity", "standard")
            try:
                complexity = QueryComplexity(complexity_str)
            except ValueError:
                complexity = QueryComplexity.STANDARD

            confidence_assessment = compute_confidence_assessment(
                plan.query,
                evidence_list,
                sources,
                complexity=complexity,
                potential_conflicts=critique.potential_conflicts if critique else None,
                consistency_issues=consistency.issues,
            )

            # Legacy confidence field uses answer confidence
            confidence_level = confidence_assessment.answer_confidence
            confidence_numeric = confidence_assessment.answer_confidence_numeric
            confidence_reasoning = confidence_assessment.answer_reasoning

            all_evidence_ids = [ev.metadata.get("display_id", "") for ev in evidence_list]
            used_set = {f"E{i}" for i in cited_numeric}
            unused = [eid for eid in all_evidence_ids if eid and eid not in used_set]

            report_metrics = {
                "validated_evidence_count": len(evidence_list),
                "evidence_used_count": len(cited_numeric),
                "evidence_unused_count": len(unused),
                "citation_count": len(citation_urls),
                "consistency_issues": consistency.issues,
                "consistency_warnings": consistency.warnings,
                "confidence_level": confidence_level.value,
                "confidence_reasoning": confidence_reasoning,
                "answer_confidence_level": confidence_assessment.answer_confidence.value,
                "answer_confidence_reasoning": confidence_assessment.answer_reasoning,
                "research_completeness_level": confidence_assessment.research_completeness.value,
                "research_completeness_reasoning": confidence_assessment.completeness_reasoning,
                "source_dedup_metrics": state.get("source_dedup_metrics"),
                "evidence_metrics": evidence_metrics,
                "claim_metrics": state.get("claim_metrics"),
                "cost_metrics": state.get("cost_metrics"),
            }

            report_result = FinalReport(
                content=report_content,
                sources=citation_urls,
                confidence=confidence_numeric,
                confidence_level=confidence_level.value,
                confidence_reasoning=confidence_reasoning,
                answer_confidence_level=confidence_assessment.answer_confidence.value,
                answer_confidence_reasoning=confidence_assessment.answer_reasoning,
                research_completeness_level=confidence_assessment.research_completeness.value,
                research_completeness_reasoning=confidence_assessment.completeness_reasoning,
                evidence_ids_used=[f"E{i}" for i in cited_numeric],
                report_metrics=report_metrics,
            )

            span.set_output({
                "report_length": len(report_result.content),
                "citation_count": len(citation_urls),
                "confidence_level": confidence_level.value,
                "consistency_issues": len(consistency.issues),
            })

            state["report_metrics"] = report_metrics

            if settings.supabase_url and settings.supabase_key:
                try:
                    from db.repository import _get_report_repo

                    report_repo = _get_report_repo()
                    report_id = await report_repo.save_report(
                        query=plan.query,
                        report=report_result,
                        quality_score=critique.quality_score if critique else None,
                        iteration_count=state.get("iteration_count", 0),
                        research_run_id=state.get("research_run_id"),
                        metadata={
                            "sub_queries": plan.sub_queries,
                            "confidence_level": confidence_level.value,
                            "confidence_reasoning": confidence_reasoning,
                            "evidence_ids_used": report_result.evidence_ids_used,
                            "report_metrics": report_metrics,
                        },
                    )
                    existing_output = span.output_data or {}
                    span.set_output({**existing_output, "report_id": report_id})
                    print(f"✅ Report saved to Supabase (ID: {report_id})")
                except Exception as e:
                    import traceback
                    print(f"❌ Failed to save report to database: {e}")
                    print(f"   Error details: {traceback.format_exc()}")
            else:
                print("ℹ️  Supabase not configured - report not saved to database")

            state["final_report"] = report_result
            state["current_node"] = "end"
            return state

        except Exception as e:
            span.set_error(e)
            state["error"] = f"Writer failed: {str(e)}"
            state["current_node"] = "end"
            return state
