"""Format validated evidence for downstream Critic and Writer nodes."""

from domain.models import Evidence, Source


def assign_evidence_ids(evidence_list: list[Evidence]) -> list[Evidence]:
    """
    Assign stable display IDs (E1, E2, ...) to evidence items in metadata.

    Uses database ID in metadata when available; display ID is always E{n}.
    """
    updated = []
    for i, ev in enumerate(evidence_list, start=1):
        display_id = f"E{i}"
        metadata = {**ev.metadata, "display_id": display_id}
        if ev.id is not None:
            metadata["db_id"] = ev.id
        updated.append(ev.model_copy(update={"metadata": metadata}))
    return updated


def build_source_lookup(sources: list[Source]) -> dict[int, Source]:
    """Map source_id → Source for evidence enrichment."""
    lookup: dict[int, Source] = {}
    for source in sources:
        if source.id is not None:
            lookup[source.id] = source
    return lookup


def format_evidence_for_prompt(
    evidence_list: list[Evidence],
    sources: list[Source],
) -> str:
    """
    Format validated evidence as the factual substrate for Critic/Writer prompts.

    Does NOT include raw search snippets.
    """
    if not evidence_list:
        return "(No validated evidence available.)"

    source_lookup = build_source_lookup(sources)
    blocks: list[str] = []

    for ev in evidence_list:
        display_id = ev.metadata.get("display_id", f"E{ev.id or '?'}")
        source = source_lookup.get(ev.source_id)
        source_title = source.title if source else ev.metadata.get("source_title", "Unknown")
        source_url = source.url if source else ev.metadata.get("source_url", "Unknown")
        source_type = source.source_type.value if source else "unknown"
        source_quality = source.source_quality.value if source else "unknown"
        match_type = ev.match_type.value if ev.match_type else "unknown"
        content_scope = ev.metadata.get("content_scope", "search_snippet")
        relevance = ev.metadata.get("relevance", "")

        block = f"""[{display_id}]
Exact text: {ev.exact_text}
Source title: {source_title}
Source URL: {source_url}
Source type: {source_type}
Source quality: {source_quality}
Content scope: {content_scope}
Validation: {match_type}
Relevance: {relevance}"""
        blocks.append(block)

    return "\n\n".join(blocks)


def extract_cited_evidence_ids(content: str) -> list[str]:
    """Extract [E1], [E2] style references from report content."""
    import re

    return sorted(set(re.findall(r"\[E(\d+)\]", content)))


def evidence_ids_to_urls(
    cited_ids: list[str],
    evidence_list: list[Evidence],
    sources: list[Source],
) -> list[str]:
    """Resolve cited evidence IDs to unique source URLs in citation order."""
    source_lookup = build_source_lookup(sources)
    urls: list[str] = []
    seen: set[str] = set()

    for cited in cited_ids:
        try:
            idx = int(cited) - 1
        except ValueError:
            continue
        if idx < 0 or idx >= len(evidence_list):
            continue
        ev = evidence_list[idx]
        source = source_lookup.get(ev.source_id)
        url = source.url if source else ev.metadata.get("source_url")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    return urls
