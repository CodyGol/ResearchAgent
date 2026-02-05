# The Oracle - Architecture Diagram

```mermaid
graph TD
    Start([User Query]) --> Planner[Planner Node]
    Planner --> |Research Plan| Researcher[Researcher Node]
    Researcher --> |Search Results| Critic[Critic Node]
    Critic --> |Quality Check| Decision{Quality<br/>Sufficient?}
    Decision -->|No - Retry| Researcher
    Decision -->|Yes| Writer[Writer Node]
    Writer --> |Final Report| End([Output])
    
    style Planner fill:#e1f5ff
    style Researcher fill:#fff4e1
    style Critic fill:#ffe1f5
    style Writer fill:#e1ffe1
    style Decision fill:#f0f0f0
```

## State Flow

1. **Planner**: Analyzes user query, generates structured research plan (sub-queries, search terms)
2. **Researcher**: Executes searches via Tavily API (with retries), aggregates results
3. **Critic**: Evaluates result quality (freshness, bias, completeness), decides if refinement needed
4. **Writer**: Synthesizes final report from approved research results

## Recursive Loop Guard

- Maximum iterations: 3 cycles (Planner → Researcher → Critic → Researcher)
- Quality threshold: Critic must score >= 0.7 to proceed
- Exponential backoff: If quality fails, wait before retry
