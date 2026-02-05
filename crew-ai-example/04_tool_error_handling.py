"""
04_tool_error_handling.py - Tool Error Handling and Rate Limiting (CX-02)

Purpose: Verify error handling and rate limiting during tool execution
- Behavior when exceptions occur inside tools
- Whether agent retries
- Error message propagation
- Rate limiting and Retry-After handling (CX-02)
- Exponential backoff implementation
- Token bucket rate limiting
- Partial failure handling

LangGraph Comparison:
- LangGraph: handle_tool_errors=True returns errors as results
- CrewAI: Errors are automatically propagated to the agent
"""

import time
import threading
from datetime import datetime
from typing import Type, Optional, Any

from crewai import Agent, Task, Crew
from crewai.tools import tool, BaseTool
from pydantic import BaseModel, Field


# =============================================================================
# Rate Limiting Infrastructure (CX-02)
# =============================================================================

class TokenBucket:
    """
    Token bucket rate limiter.

    Allows burst capacity while maintaining average rate limit.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> tuple[bool, float]:
        """
        Try to acquire tokens.

        Returns:
            (success, wait_time_if_failed)
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0

            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.refill_rate
            return False, wait_time


class ExponentialBackoff:
    """
    Exponential backoff calculator with jitter.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt = 0

    def get_delay(self) -> float:
        """Calculate delay for current attempt."""
        import random

        delay = min(
            self.base_delay * (self.multiplier ** self.attempt),
            self.max_delay
        )

        # Add jitter
        jitter_amount = delay * self.jitter
        delay += random.uniform(-jitter_amount, jitter_amount)

        self.attempt += 1
        return max(0, delay)

    def reset(self):
        """Reset attempt counter."""
        self.attempt = 0


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


# =============================================================================
# Rate Limited Tool Implementation
# =============================================================================

class RateLimitedAPIInput(BaseModel):
    """Input schema for rate-limited API tool."""

    endpoint: str = Field(..., description="API endpoint to call")
    method: str = Field(default="GET", description="HTTP method")


class RateLimitedAPITool(BaseTool):
    """
    API tool with rate limiting and exponential backoff.

    Demonstrates CX-02: Rate limiting and retry handling.
    """

    name: str = "Rate Limited API"
    description: str = """Make API calls with rate limiting protection.
    Automatically handles 429 errors with exponential backoff."""
    args_schema: Type[BaseModel] = RateLimitedAPIInput

    # Rate limiter (10 requests per second, burst of 20)
    rate_limiter: TokenBucket = None
    backoff: ExponentialBackoff = None

    # Simulated 429 behavior
    _call_count: int = 0
    _fail_on_calls: list = [3, 4]  # Simulate 429 on 3rd and 4th calls

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rate_limiter = TokenBucket(capacity=20, refill_rate=10)
        self.backoff = ExponentialBackoff()
        self._call_count = 0
        self._fail_on_calls = [3, 4]

    def _run(self, endpoint: str, method: str = "GET") -> str:
        """Execute API call with rate limiting."""
        self._call_count += 1
        current_call = self._call_count

        print(f"  [RateLimitedAPI] Call #{current_call}: {method} {endpoint}")

        # Check rate limit
        allowed, wait_time = self.rate_limiter.acquire()
        if not allowed:
            print(f"  [RateLimitedAPI] Rate limited! Wait {wait_time:.2f}s")
            time.sleep(wait_time)
            allowed, _ = self.rate_limiter.acquire()

        # Simulate 429 responses
        if current_call in self._fail_on_calls:
            retry_after = self.backoff.get_delay()
            print(f"  [RateLimitedAPI] 429 Too Many Requests! Retry-After: {retry_after:.2f}s")
            time.sleep(retry_after)
            return f"[RETRIED] {method} {endpoint} - Success after backoff"

        self.backoff.reset()
        return f"[SUCCESS] {method} {endpoint} - Call #{current_call}"


# =============================================================================
# Partial Failure Handling Tool
# =============================================================================

class BatchOperationInput(BaseModel):
    """Input for batch operations."""

    items: list[str] = Field(..., description="Items to process")
    fail_indices: Optional[list[int]] = Field(
        default=None,
        description="Indices that should fail (for testing)"
    )


class BatchProcessorTool(BaseTool):
    """
    Tool that demonstrates partial failure handling.

    Some items may fail while others succeed.
    Returns detailed status for each item.
    """

    name: str = "Batch Processor"
    description: str = """Process multiple items in a batch.
    Handles partial failures gracefully, returning status for each item."""
    args_schema: Type[BaseModel] = BatchOperationInput

    def _run(self, items: list[str], fail_indices: Optional[list[int]] = None) -> str:
        """Process batch with partial failure handling."""
        fail_indices = fail_indices or []
        results = []
        successes = 0
        failures = 0

        print(f"  [BatchProcessor] Processing {len(items)} items...")

        for i, item in enumerate(items):
            if i in fail_indices:
                results.append({
                    "item": item,
                    "status": "failed",
                    "error": "Simulated failure",
                })
                failures += 1
                print(f"  [BatchProcessor] Item {i} '{item}': FAILED")
            else:
                results.append({
                    "item": item,
                    "status": "success",
                    "result": f"Processed: {item}",
                })
                successes += 1
                print(f"  [BatchProcessor] Item {i} '{item}': SUCCESS")

        summary = f"""
Batch Processing Complete
=========================
Total: {len(items)}
Successes: {successes}
Failures: {failures}

Results:
{chr(10).join([f"  - {r['item']}: {r['status']}" for r in results])}
"""
        return summary


# =============================================================================
# Retry-After Handler
# =============================================================================

class RetryAfterHandler:
    """
    Handles Retry-After headers from API responses.

    Supports:
    - Numeric seconds: "120"
    - HTTP date: "Wed, 21 Oct 2025 07:28:00 GMT"
    """

    @staticmethod
    def parse_retry_after(value: str) -> float:
        """Parse Retry-After header value to seconds."""
        # Try numeric first
        try:
            return float(value)
        except ValueError:
            pass

        # Try HTTP date format
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(value)
            delta = dt - datetime.now(dt.tzinfo)
            return max(0, delta.total_seconds())
        except Exception:
            pass

        # Default fallback
        return 60.0

    @staticmethod
    def wait_and_retry(retry_after: str, operation: callable, *args, **kwargs) -> Any:
        """Wait for Retry-After duration then retry operation."""
        wait_seconds = RetryAfterHandler.parse_retry_after(retry_after)
        print(f"  [RetryAfter] Waiting {wait_seconds:.2f} seconds...")
        time.sleep(wait_seconds)
        return operation(*args, **kwargs)


# =============================================================================
# Tool that throws exceptions (original)
# =============================================================================
@tool("Failing Tool")
def failing_tool(should_fail: bool = True) -> str:
    """
    A tool that fails on purpose for testing error handling.

    Args:
        should_fail: If True, raises an exception
    """
    if should_fail:
        raise ValueError("Intentional failure for testing!")
    return "Success! Tool executed without errors."


# =============================================================================
# Tool that conditionally fails (for retry testing)
# =============================================================================
call_counter = {"count": 0}


@tool("Flaky Tool")
def flaky_tool(operation: str) -> str:
    """
    A tool that fails on first call but succeeds on subsequent calls.
    Useful for testing retry behavior.

    Args:
        operation: The operation to perform
    """
    call_counter["count"] += 1
    print(f"  [Flaky Tool] Call #{call_counter['count']} for operation: {operation}")

    if call_counter["count"] == 1:
        raise ConnectionError("Simulated network error on first attempt!")

    return f"Operation '{operation}' completed successfully on attempt #{call_counter['count']}"


# =============================================================================
# Tool that returns errors (error message instead of exception)
# =============================================================================
class DivisionInput(BaseModel):
    """Input for division tool."""

    dividend: float = Field(..., description="Number to be divided")
    divisor: float = Field(..., description="Number to divide by")


class SafeDivisionTool(BaseTool):
    """A division tool that returns error messages instead of raising exceptions."""

    name: str = "Safe Division"
    description: str = "Divide two numbers safely, returning error message for invalid operations."
    args_schema: Type[BaseModel] = DivisionInput

    def _run(self, dividend: float, divisor: float) -> str:
        print(f"  [Safe Division] {dividend} / {divisor}")

        if divisor == 0:
            return "Error: Division by zero is not allowed. Please provide a non-zero divisor."

        result = dividend / divisor
        return f"{dividend} / {divisor} = {result}"


# =============================================================================
# Tool that causes input validation errors
# =============================================================================
class StrictInput(BaseModel):
    """Strict input validation."""

    value: int = Field(..., ge=0, le=100, description="A value between 0 and 100")


class StrictValidationTool(BaseTool):
    """Tool with strict input validation."""

    name: str = "Strict Validator"
    description: str = "A tool that only accepts values between 0 and 100."
    args_schema: Type[BaseModel] = StrictInput

    def _run(self, value: int) -> str:
        return f"Validated value: {value}"


# =============================================================================
# Demonstration Functions
# =============================================================================

def demo_token_bucket():
    """Demonstrate token bucket rate limiting."""
    print("=" * 60)
    print("Demo: Token Bucket Rate Limiting")
    print("=" * 60)

    bucket = TokenBucket(capacity=5, refill_rate=2)  # 5 burst, 2/sec refill

    print("\nAttempting 10 rapid requests...")
    for i in range(10):
        allowed, wait_time = bucket.acquire()
        if allowed:
            print(f"  Request {i+1}: ALLOWED")
        else:
            print(f"  Request {i+1}: BLOCKED (wait {wait_time:.2f}s)")
            time.sleep(wait_time)
            bucket.acquire()
            print(f"  Request {i+1}: ALLOWED after wait")


def demo_exponential_backoff():
    """Demonstrate exponential backoff."""
    print("\n" + "=" * 60)
    print("Demo: Exponential Backoff")
    print("=" * 60)

    backoff = ExponentialBackoff(base_delay=1.0, max_delay=30.0, multiplier=2.0)

    print("\nCalculating delays for 6 retry attempts:")
    for i in range(6):
        delay = backoff.get_delay()
        print(f"  Attempt {i+1}: wait {delay:.2f}s")


def demo_rate_limited_tool():
    """Demonstrate rate-limited API tool."""
    print("\n" + "=" * 60)
    print("Demo: Rate Limited API Tool")
    print("=" * 60)

    tool = RateLimitedAPITool()

    print("\nMaking 6 API calls (429 simulated on calls 3,4)...")
    for i in range(6):
        result = tool._run(endpoint=f"/api/resource/{i}", method="GET")
        print(f"  Result: {result}")


def demo_batch_partial_failure():
    """Demonstrate partial failure handling."""
    print("\n" + "=" * 60)
    print("Demo: Batch Partial Failure Handling")
    print("=" * 60)

    tool = BatchProcessorTool()
    items = ["item_a", "item_b", "item_c", "item_d", "item_e"]
    fail_indices = [1, 3]  # item_b and item_d will fail

    result = tool._run(items=items, fail_indices=fail_indices)
    print(result)


def demo_retry_after_parsing():
    """Demonstrate Retry-After header parsing."""
    print("\n" + "=" * 60)
    print("Demo: Retry-After Header Parsing")
    print("=" * 60)

    handler = RetryAfterHandler()

    test_values = [
        "120",
        "60",
        "Wed, 21 Oct 2025 07:28:00 GMT",
        "invalid",
    ]

    for value in test_values:
        seconds = handler.parse_retry_after(value)
        print(f"  '{value}' -> {seconds:.2f} seconds")


def main():
    print("=" * 60)
    print("Tool Error Handling and Rate Limiting Test (CX-02)")
    print("=" * 60)
    print("""
This script verifies error handling and rate limiting capabilities.

Original Verification Items:
- Exception handling in tools
- Agent retry behavior
- Error message propagation

Enhanced Verification Items (CX-02):
- Token bucket rate limiting
- Exponential backoff with jitter
- Retry-After header handling
- Partial failure handling in batch operations

LangGraph Comparison:
- Both require custom implementation for rate limiting
- CrewAI max_retry_limit provides basic retry
- Advanced rate limiting needs custom tools
""")

    # Run rate limiting demos first (no LLM calls)
    demo_token_bucket()
    demo_exponential_backoff()
    demo_rate_limited_tool()
    demo_batch_partial_failure()
    demo_retry_after_parsing()

    # Original error handling test with agent
    print("\n" + "=" * 60)
    print("Original Error Handling Test with Agent")
    print("=" * 60)

    # Reset counter
    call_counter["count"] = 0

    tools = [
        failing_tool,
        flaky_tool,
        SafeDivisionTool(),
        StrictValidationTool(),
        RateLimitedAPITool(),
        BatchProcessorTool(),
    ]

    # Error handling specialist agent
    error_handler = Agent(
        role="Error Handling Specialist",
        goal="Test various error scenarios and observe how errors are handled",
        backstory="""You are an expert at testing error handling.
        You intentionally trigger errors to verify system robustness.
        When a tool fails, you analyze the error and try alternative approaches.""",
        tools=tools,
        verbose=True,
        max_retry_limit=3,  # Allow retries on failure
    )

    # Task designed to trigger various error conditions
    error_test_task = Task(
        description="""Test error handling by:
        1. Use the Safe Division tool to divide 10 by 0 (should return error message)
        2. Use the Safe Division tool to divide 10 by 2 (should succeed)
        3. Use the Flaky Tool with operation "test" (may fail first, retry should succeed)
        4. Try the Failing Tool with should_fail=True, then with should_fail=False

        Report what errors occurred and how they were handled.""",
        expected_output="""A report containing:
        - Each error encountered
        - How the agent recovered or handled the error
        - Final successful results where applicable""",
        agent=error_handler,
    )

    crew = Crew(
        agents=[error_handler],
        tasks=[error_test_task],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Executing Error Handling Test")
    print("=" * 60)

    try:
        result = crew.kickoff()

        print("\n" + "=" * 60)
        print("Result:")
        print("=" * 60)
        print(result)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Crew execution failed with error: {type(e).__name__}")
        print("=" * 60)
        print(f"Error message: {e}")

    print(f"\n[Debug] Flaky tool was called {call_counter['count']} times")


if __name__ == "__main__":
    main()
