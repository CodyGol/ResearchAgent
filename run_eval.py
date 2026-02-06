"""High-Performance Evaluation System for The Oracle using LLM-as-a-Judge."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_anthropic import ChatAnthropic
from tabulate import tabulate

from config import settings
from graph import create_graph, create_run_config
from state import AgentState


async def evaluate_answer(query: str, actual: str, expected: str) -> int:
    """
    LLM-as-a-Judge: Grade the actual answer against expected.
    
    Args:
        query: The original query
        actual: The agent's actual answer
        expected: The expected answer
        
    Returns:
        1 if correct, 0 if incorrect
    """
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,  # Deterministic grading
    )
    
    prompt = f"""You are a harsh technical grader. Compare the ACTUAL answer to the EXPECTED answer.

QUERY: {query}

EXPECTED ANSWER: {expected}

ACTUAL ANSWER: {actual}

If the ACTUAL answer contains the core truth of the EXPECTED answer, grade it 1.
If it misses the point or is factually wrong, grade it 0.
Return ONLY the number (0 or 1), no explanation."""

    try:
        response = await llm.ainvoke(prompt)
        grade_str = response.content.strip()
        # Extract first digit (in case LLM adds extra text)
        grade = int(grade_str[0]) if grade_str and grade_str[0].isdigit() else 0
        return min(max(grade, 0), 1)  # Clamp to 0 or 1
    except Exception as e:
        print(f"⚠️  Judge error for query '{query[:50]}...': {e}")
        return 0  # Fail conservatively


async def run_agent_query(
    query: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, float]:
    """
    Run a single query through The Oracle agent.
    
    Args:
        query: The research query
        semaphore: Concurrency limiter
        
    Returns:
        Tuple of (answer, latency_seconds)
    """
    async with semaphore:  # Limit concurrency
        start_time = time.time()
        
        try:
            graph = create_graph()
            app = graph.compile()
            run_config = create_run_config()
            
            initial_state: AgentState = {
                "user_query": query,
                "research_plan": None,
                "research_results": None,
                "critique": None,
                "final_report": None,
                "current_node": "planner",
                "iteration_count": 0,
                "error": None,
            }
            
            final_state = await app.ainvoke(initial_state, config=run_config)
            
            latency = time.time() - start_time
            
            if final_state.get("error"):
                return f"ERROR: {final_state['error']}", latency
            
            report = final_state.get("final_report")
            if not report:
                return "ERROR: No report generated", latency
            
            return report.content, latency
            
        except Exception as e:
            latency = time.time() - start_time
            return f"ERROR: {str(e)}", latency


async def evaluate_single_case(
    case: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Evaluate a single test case: run agent + judge.
    
    Args:
        case: Test case dict with 'query', 'expected_answer', 'category'
        semaphore: Concurrency limiter
        
    Returns:
        Result dict with all metrics
    """
    query = case["query"]
    expected = case["expected_answer"]
    category = case.get("category", "unknown")
    
    # Run agent
    actual_answer, latency = await run_agent_query(query, semaphore)
    
    # Judge the answer (only if not an error)
    if actual_answer.startswith("ERROR"):
        grade = 0
    else:
        grade = await evaluate_answer(query, actual_answer, expected)
    
    return {
        "category": category,
        "query": query,
        "expected": expected,
        "actual": actual_answer[:100] + "..." if len(actual_answer) > 100 else actual_answer,
        "grade": grade,
        "latency": round(latency, 2),
    }


async def main():
    """Main evaluation loop: parallel execution + reporting."""
    # Load golden dataset
    dataset_path = Path(__file__).parent / "tests" / "golden_dataset.json"
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    with open(dataset_path, "r") as f:
        test_cases = json.load(f)
    
    print(f"🚀 Starting Evaluation: {len(test_cases)} test cases")
    print(f"⚡ Running in parallel (max 5 concurrent)...\n")
    
    # Create semaphore for concurrency control (limit to 5)
    semaphore = asyncio.Semaphore(5)
    
    # Run all evaluations in parallel
    start_time = time.time()
    results = await asyncio.gather(
        *[evaluate_single_case(case, semaphore) for case in test_cases]
    )
    total_time = time.time() - start_time
    
    # Create DataFrame for reporting
    df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = (df["grade"].sum() / len(df)) * 100
    avg_latency = df["latency"].mean()
    
    # Print detailed results table
    print("\n" + "=" * 100)
    print("📊 EVALUATION RESULTS")
    print("=" * 100 + "\n")
    
    # Format table (exclude full answers for readability)
    display_df = df[["category", "query", "grade", "latency"]].copy()
    display_df.columns = ["Category", "Query", "Grade", "Latency (s)"]
    display_df["Grade"] = display_df["Grade"].map({1: "✅", 0: "❌"})
    
    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False))
    
    # Print summary
    print("\n" + "=" * 100)
    print("📈 SUMMARY")
    print("=" * 100)
    print(f"✅ Accuracy: {accuracy:.1f}% ({df['grade'].sum()}/{len(df)})")
    print(f"⏱️  Average Latency: {avg_latency:.2f}s")
    print(f"🚀 Total Time: {total_time:.2f}s")
    print(f"📦 Test Cases: {len(df)}")
    
    # Category breakdown
    if "category" in df.columns:
        print("\n📊 By Category:")
        category_stats = df.groupby("category").agg({
            "grade": ["sum", "count"],
            "latency": "mean"
        }).round(2)
        category_stats.columns = ["Correct", "Total", "Avg Latency (s)"]
        category_stats["Accuracy %"] = (category_stats["Correct"] / category_stats["Total"] * 100).round(1)
        # Reorder columns
        category_stats = category_stats[["Correct", "Total", "Accuracy %", "Avg Latency (s)"]]
        print(tabulate(category_stats, headers="keys", tablefmt="github"))
    
    print()


if __name__ == "__main__":
    asyncio.run(main())
