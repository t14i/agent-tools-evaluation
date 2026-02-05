"""
03_tool_execution.py - Tool Execution Verification

Purpose: Verify tool execution mechanism
- Custom tool execution
- Behavior of multiple invocations
- Execution time measurement

LangGraph Comparison:
- Both frameworks support @tool decorator for tool definition
- CrewAI agents autonomously decide when to use tools
"""

import time
from datetime import datetime
from typing import Type

from crewai import Agent, Task, Crew
from crewai.tools import tool, BaseTool
from pydantic import BaseModel, Field


# =============================================================================
# Simple tool
# =============================================================================
@tool("Timestamp Generator")
def get_timestamp() -> str:
    """Get current timestamp. Each call returns a different value."""
    timestamp = datetime.now().isoformat()
    print(f"  [Tool Executed] get_timestamp() -> {timestamp}")
    return f"Current timestamp: {timestamp}"


# =============================================================================
# Tool that simulates a slow API
# =============================================================================
@tool("Slow API Call")
def slow_api_call(query: str) -> str:
    """
    Simulate a slow API call.

    Args:
        query: The search query
    """
    print(f"  [Tool Executed] slow_api_call('{query}') - Simulating 1 second delay...")
    time.sleep(1)  # Simulate slow API
    return f"API Result for '{query}': Sample data retrieved at {datetime.now().isoformat()}"


# =============================================================================
# Tool with multiple arguments
# =============================================================================
class SearchInput(BaseModel):
    """Input for search tool."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, description="Maximum results to return")


class SearchTool(BaseTool):
    """Search tool with multiple parameters."""

    name: str = "Advanced Search"
    description: str = "Search with advanced options including result limit."
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, max_results: int = 10) -> str:
        print(f"  [Tool Executed] SearchTool('{query}', max_results={max_results})")
        time.sleep(0.5)  # Simulate processing
        return f"Search result for '{query}': Found {max_results} items (limited)"


def main():
    print("=" * 60)
    print("Tool Execution Test")
    print("=" * 60)

    tools = [
        get_timestamp,
        slow_api_call,
        SearchTool(),
    ]

    # Display tool info
    print("\nAvailable Tools:")
    for t in tools:
        print(f"  - {t.name}: {type(t).__name__}")

    analyst = Agent(
        role="Tool Tester",
        goal="Test tool execution by calling various tools",
        backstory="You test tools methodically and report results.",
        tools=tools,
        verbose=True,
    )

    # Task designed to test tool execution
    test_task = Task(
        description="""Test tool execution:
        1. Call get_timestamp to get current time
        2. Call slow_api_call with query "AI agents"
        3. Call advanced_search with query "machine learning" and max_results=5

        Report all results.""",
        expected_output="""A report showing:
        - Timestamp result
        - Slow API call result
        - Advanced search result""",
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[test_task],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Executing Tool Test")
    print("=" * 60)

    start_time = time.time()
    result = crew.kickoff()
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Result (Total time: {elapsed:.2f}s):")
    print("=" * 60)
    print(result)

    # Check token usage if available
    if hasattr(result, "token_usage"):
        print(f"\nToken usage: {result.token_usage}")


if __name__ == "__main__":
    main()
