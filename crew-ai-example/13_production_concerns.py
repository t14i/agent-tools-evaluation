"""
13_production_concerns.py - Production Environment Concerns

Purpose: Verify features required for production operation
- Audit logging: How to save and retrieve execution history
- Token consumption: Cost visibility
- Observability: Tracing features
- Parallel execution: Parallel processing of multiple Crews
- Error handling: Graceful failure handling

LangGraph Comparison:
- LangGraph: LangSmith integration, explicit state management
- CrewAI: Built-in features + Enterprise edition extensions
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from crewai import Agent, Task, Crew
from crewai.tools import tool


# =============================================================================
# Logging and Audit Setup
# =============================================================================
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


class AuditLogger:
    """Custom audit logger for tracking crew executions."""

    def __init__(self, log_file: str = "audit.jsonl"):
        self.log_path = LOG_DIR / log_file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, event_type: str, data: dict[str, Any]):
        """Log an event to the audit file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            **data,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def get_logs(self, event_type: str | None = None) -> list[dict]:
        """Retrieve logs, optionally filtered by event type."""
        if not self.log_path.exists():
            return []

        logs = []
        with open(self.log_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                if event_type is None or entry.get("event_type") == event_type:
                    logs.append(entry)
        return logs


# Global logger instance
audit_logger = AuditLogger()


# =============================================================================
# Tools with logging
# =============================================================================
@tool("Logged API Call")
def logged_api_call(endpoint: str) -> str:
    """
    Make an API call with logging.

    Args:
        endpoint: The API endpoint to call
    """
    start_time = time.time()

    # Simulate API call
    time.sleep(0.5)
    result = f"Response from {endpoint}: OK"

    # Log the call
    audit_logger.log("tool_call", {
        "tool": "logged_api_call",
        "endpoint": endpoint,
        "duration_ms": (time.time() - start_time) * 1000,
        "status": "success",
    })

    return result


# =============================================================================
# 1. Token Usage and Cost Tracking
# =============================================================================
def test_token_tracking():
    """Test token usage tracking."""
    print("\n" + "=" * 60)
    print("1. Token Usage and Cost Tracking")
    print("=" * 60)

    agent = Agent(
        role="Token Counter",
        goal="Perform tasks while tracking token usage",
        backstory="You help track LLM costs.",
        verbose=True,
    )

    task = Task(
        description="Write a haiku about artificial intelligence.",
        expected_output="A 5-7-5 syllable haiku",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
    )

    audit_logger.log("crew_start", {"crew_id": "token_test"})

    result = crew.kickoff()

    # Check token usage
    token_usage = getattr(result, "token_usage", None)

    audit_logger.log("crew_complete", {
        "crew_id": "token_test",
        "token_usage": str(token_usage) if token_usage else "not_available",
    })

    print("\n[Token Usage Analysis]")
    if token_usage:
        print(f"Token Usage: {token_usage}")
        # Calculate estimated cost (example rates)
        # GPT-4: ~$0.03/1K input, ~$0.06/1K output
        if hasattr(token_usage, "total_tokens"):
            estimated_cost = token_usage.total_tokens * 0.00003
            print(f"Estimated Cost: ${estimated_cost:.4f}")
    else:
        print("Token usage not available in result object")
        print("Note: Check CrewAI version for token tracking support")

    return result


# =============================================================================
# 2. Audit Logging
# =============================================================================
def test_audit_logging():
    """Test audit logging capabilities."""
    print("\n" + "=" * 60)
    print("2. Audit Logging")
    print("=" * 60)

    agent = Agent(
        role="Audited Agent",
        goal="Perform tasks with full audit trail",
        backstory="Every action you take is logged.",
        tools=[logged_api_call],
        verbose=True,
    )

    task = Task(
        description="Call the API endpoint '/users' and '/orders'.",
        expected_output="Results from both API calls",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
    )

    audit_logger.log("crew_start", {
        "crew_id": "audit_test",
        "agents": ["Audited Agent"],
        "task_count": 1,
    })

    start_time = time.time()
    result = crew.kickoff()
    duration = time.time() - start_time

    audit_logger.log("crew_complete", {
        "crew_id": "audit_test",
        "duration_seconds": duration,
        "status": "success",
    })

    # Display audit logs
    print("\n[Audit Log Contents]")
    logs = audit_logger.get_logs()
    for log in logs[-5:]:  # Last 5 entries
        print(f"  {log['timestamp']} | {log['event_type']}: {json.dumps({k: v for k, v in log.items() if k not in ['timestamp', 'event_type', 'session_id']})}")

    return result


# =============================================================================
# 3. Observability / Tracing
# =============================================================================
def test_observability():
    """Test observability and tracing features."""
    print("\n" + "=" * 60)
    print("3. Observability and Tracing")
    print("=" * 60)
    print("""
CrewAI Observability Options:
1. verbose=True: Basic logging to stdout
2. output_log_file: Log to file
3. Third-party integrations:
   - OpenTelemetry
   - LangSmith (via LangChain integration)
   - Weights & Biases

Note: Native tracing requires CrewAI Enterprise or third-party tools.
""")

    agent = Agent(
        role="Traced Agent",
        goal="Demonstrate tracing capabilities",
        backstory="Your executions are traced.",
        verbose=True,
    )

    task = Task(
        description="Explain what observability means in 2 sentences.",
        expected_output="A brief explanation of observability",
        agent=agent,
    )

    # Crew with file logging
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        output_log_file=str(LOG_DIR / "crew_trace.log"),
    )

    result = crew.kickoff()

    # Check if trace log was created
    trace_file = LOG_DIR / "crew_trace.log"
    if trace_file.exists():
        print(f"\n[Trace Log Created: {trace_file}]")
        print(f"File size: {trace_file.stat().st_size} bytes")
    else:
        print("\nNote: output_log_file may not be supported in this version")

    return result


# =============================================================================
# 4. Parallel Execution
# =============================================================================
async def run_crew_async(crew_id: str, task_description: str) -> dict:
    """Run a crew asynchronously."""
    agent = Agent(
        role=f"Worker {crew_id}",
        goal="Complete assigned tasks efficiently",
        backstory=f"You are worker {crew_id}.",
        verbose=False,  # Reduce output for parallel runs
    )

    task = Task(
        description=task_description,
        expected_output="Task result",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    start_time = time.time()
    result = crew.kickoff()
    duration = time.time() - start_time

    return {
        "crew_id": crew_id,
        "duration": duration,
        "result": str(result.raw)[:100],
    }


def test_parallel_execution():
    """Test parallel crew execution."""
    print("\n" + "=" * 60)
    print("4. Parallel Execution")
    print("=" * 60)

    tasks_to_run = [
        ("crew_1", "What is 2 + 2? Answer briefly."),
        ("crew_2", "What is the capital of France? Answer briefly."),
        ("crew_3", "Name one color. Answer briefly."),
    ]

    print(f"\nRunning {len(tasks_to_run)} crews in parallel...")
    start_time = time.time()

    # Run crews in parallel using asyncio
    async def run_all():
        tasks = [run_crew_async(cid, desc) for cid, desc in tasks_to_run]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
    total_duration = time.time() - start_time

    print("\n[Parallel Execution Results]")
    individual_total = 0
    for r in results:
        print(f"  {r['crew_id']}: {r['duration']:.2f}s - {r['result'][:50]}...")
        individual_total += r["duration"]

    print(f"\n  Total wall time: {total_duration:.2f}s")
    print(f"  Sum of individual times: {individual_total:.2f}s")
    print(f"  Parallelism benefit: {individual_total - total_duration:.2f}s saved")

    return results


# =============================================================================
# 5. Error Handling and Recovery
# =============================================================================
def test_error_handling():
    """Test error handling and graceful degradation."""
    print("\n" + "=" * 60)
    print("5. Error Handling and Recovery")
    print("=" * 60)

    @tool("Unreliable Service")
    def unreliable_service(action: str) -> str:
        """
        A service that fails randomly.

        Args:
            action: The action to perform
        """
        import random
        if random.random() < 0.5:
            raise Exception("Service temporarily unavailable")
        return f"Action '{action}' completed successfully"

    agent = Agent(
        role="Resilient Agent",
        goal="Complete tasks despite failures",
        backstory="You handle errors gracefully and retry when needed.",
        tools=[unreliable_service],
        verbose=True,
        max_retry_limit=3,
    )

    task = Task(
        description="""Try to use the unreliable service with action "process".
        If it fails, try again or explain what happened.""",
        expected_output="Result of the service call or error explanation",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
    )

    try:
        result = crew.kickoff()
        print("\n[Error Handling Result]")
        print(result)
    except Exception as e:
        print(f"\n[Crew Failed with Error]: {e}")
        print("Note: In production, implement circuit breakers and fallbacks")


# =============================================================================
# 6. Summary and Recommendations
# =============================================================================
def print_production_summary():
    """Print production readiness summary."""
    print("\n" + "=" * 60)
    print("Production Readiness Summary")
    print("=" * 60)
    print("""
| Feature                | CrewAI Status          | LangGraph Comparison      |
|------------------------|------------------------|---------------------------|
| Token Tracking         | Built-in (token_usage) | Manual or LangSmith       |
| Audit Logging          | Custom implementation  | LangSmith or custom       |
| Tracing                | output_log_file        | LangSmith native          |
| Parallel Execution     | asyncio compatible     | Native async support      |
| Error Handling         | max_retry_limit        | handle_tool_errors        |
| Persistence            | @persist (Flow)        | Checkpointer              |
| Memory                 | memory=True            | Explicit state            |

Recommendations for Production:
1. Implement custom audit logging (as shown above)
2. Use structured logging (JSON format)
3. Integrate with observability platform (OpenTelemetry, DataDog, etc.)
4. Set up cost monitoring dashboards
5. Implement circuit breakers for external services
6. Use async execution for parallel workloads
7. Consider CrewAI Enterprise for advanced features

Enterprise Features (not tested):
- Advanced analytics dashboard
- Team collaboration tools
- Enhanced security features
- Priority support
""")


def main():
    print("=" * 60)
    print("Production Concerns Verification")
    print("=" * 60)
    print("""
This script tests production-critical features:
1. Token Usage and Cost Tracking
2. Audit Logging
3. Observability/Tracing
4. Parallel Execution
5. Error Handling

Each test demonstrates how to implement production-grade
features with CrewAI.
""")

    # Run all tests
    tests = [
        ("Token Tracking", test_token_tracking),
        ("Audit Logging", test_audit_logging),
        ("Observability", test_observability),
        ("Parallel Execution", test_parallel_execution),
        ("Error Handling", test_error_handling),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n[{test_name} Failed]: {e}")
            audit_logger.log("test_error", {
                "test": test_name,
                "error": str(e),
            })

    # Print summary
    print_production_summary()

    # Show final audit log stats
    logs = audit_logger.get_logs()
    print(f"\nTotal audit log entries: {len(logs)}")
    print(f"Log file: {audit_logger.log_path}")


if __name__ == "__main__":
    main()
