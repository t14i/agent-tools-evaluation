"""
LangGraph Connectors - Rate Limiting & Retry (CX-02)
Token bucket, exponential backoff, circuit breaker patterns.

Evaluation: CX-02 (Rate Limit / Retry)
"""

import time
import random
from typing import Annotated, TypedDict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage


# =============================================================================
# RATE LIMITING INFRASTRUCTURE
# =============================================================================

@dataclass
class TokenBucket:
    """Token bucket rate limiter."""
    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(default=None)
    last_refill: datetime = field(default=None)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = self.capacity
        if self.last_refill is None:
            self.last_refill = datetime.now()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        with self.lock:
            now = datetime.now()
            elapsed = (now - self.last_refill).total_seconds()

            # Refill tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens to become available."""
        with self.lock:
            if self.tokens >= tokens:
                return 0
            return (tokens - self.tokens) / self.refill_rate


@dataclass
class ExponentialBackoff:
    """Exponential backoff calculator with jitter."""
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay
    factor: float = 2.0  # Multiplication factor
    jitter: float = 0.1  # Random jitter factor (0-1)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        delay = min(self.base_delay * (self.factor ** attempt), self.max_delay)
        jitter_range = delay * self.jitter
        return delay + random.uniform(-jitter_range, jitter_range)


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before half-open
    success_threshold: int = 2  # Successes needed to close

    failures: int = field(default=0)
    successes: int = field(default=0)
    state: str = field(default="closed")  # closed, open, half-open
    last_failure: Optional[datetime] = field(default=None)
    lock: Lock = field(default_factory=Lock)

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self.lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if self.last_failure and \
                   (datetime.now() - self.last_failure).total_seconds() > self.recovery_timeout:
                    self.state = "half-open"
                    self.successes = 0
                    return True
                return False
            else:  # half-open
                return True

    def record_success(self):
        """Record a successful execution."""
        with self.lock:
            if self.state == "half-open":
                self.successes += 1
                if self.successes >= self.success_threshold:
                    self.state = "closed"
                    self.failures = 0
            elif self.state == "closed":
                self.failures = 0

    def record_failure(self):
        """Record a failed execution."""
        with self.lock:
            self.failures += 1
            self.last_failure = datetime.now()
            if self.state == "half-open":
                self.state = "open"
            elif self.failures >= self.failure_threshold:
                self.state = "open"


# =============================================================================
# RATE-LIMITED TOOL NODE
# =============================================================================

class RateLimiter:
    """Multi-API rate limiter manager."""

    def __init__(self):
        self.buckets: dict[str, TokenBucket] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.backoff = ExponentialBackoff()
        self.retry_counts: dict[str, int] = defaultdict(int)
        self.lock = Lock()

    def configure_api(
        self,
        api_name: str,
        requests_per_second: float = 10,
        burst_capacity: int = 20,
        failure_threshold: int = 5
    ):
        """Configure rate limiting for an API."""
        with self.lock:
            self.buckets[api_name] = TokenBucket(
                capacity=burst_capacity,
                refill_rate=requests_per_second
            )
            self.circuit_breakers[api_name] = CircuitBreaker(
                failure_threshold=failure_threshold
            )

    def can_execute(self, api_name: str) -> tuple[bool, str]:
        """Check if API call is allowed. Returns (allowed, reason)."""
        if api_name not in self.buckets:
            self.configure_api(api_name)

        cb = self.circuit_breakers[api_name]
        if not cb.can_execute():
            return False, f"Circuit breaker open for {api_name}"

        bucket = self.buckets[api_name]
        if not bucket.acquire():
            wait = bucket.wait_time()
            return False, f"Rate limited: wait {wait:.2f}s"

        return True, "OK"

    def record_success(self, api_name: str):
        """Record successful API call."""
        if api_name in self.circuit_breakers:
            self.circuit_breakers[api_name].record_success()
            with self.lock:
                self.retry_counts[api_name] = 0

    def record_failure(self, api_name: str):
        """Record failed API call."""
        if api_name in self.circuit_breakers:
            self.circuit_breakers[api_name].record_failure()

    def get_retry_delay(self, api_name: str) -> float:
        """Get retry delay with exponential backoff."""
        with self.lock:
            attempt = self.retry_counts[api_name]
            self.retry_counts[api_name] = attempt + 1
            return self.backoff.get_delay(attempt)


class RateLimitedToolNode:
    """ToolNode wrapper with rate limiting, retry, and circuit breaker."""

    def __init__(
        self,
        tools: list,
        rate_limiter: RateLimiter,
        max_retries: int = 3
    ):
        self.tool_node = ToolNode(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries

        # Map tools to API names (can be customized)
        self.tool_to_api = {t.name: t.name for t in tools}

    def __call__(self, state: dict) -> dict:
        """Execute tools with rate limiting and retry."""
        last_message = state["messages"][-1]
        results = []
        metrics = state.get("rate_limit_metrics", {
            "rate_limited": 0,
            "circuit_open": 0,
            "retries": 0,
            "successes": 0,
            "failures": 0
        })

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            api_name = self.tool_to_api.get(tool_name, tool_name)

            result = None
            last_error = None

            for attempt in range(self.max_retries + 1):
                # Check rate limit and circuit breaker
                allowed, reason = self.rate_limiter.can_execute(api_name)

                if not allowed:
                    if "Circuit breaker" in reason:
                        metrics["circuit_open"] += 1
                        print(f"    [CB] {reason}")
                    else:
                        metrics["rate_limited"] += 1
                        print(f"    [RL] {reason}")

                    # Wait and retry
                    delay = self.rate_limiter.get_retry_delay(api_name)
                    print(f"    [WAIT] Sleeping {delay:.2f}s before retry")
                    time.sleep(min(delay, 2.0))  # Cap for demo
                    metrics["retries"] += 1
                    continue

                # Execute tool
                try:
                    tool_fn = self.tools_by_name[tool_name]
                    result = tool_fn.invoke(tool_args)
                    self.rate_limiter.record_success(api_name)
                    metrics["successes"] += 1
                    print(f"    [OK] {tool_name} succeeded on attempt {attempt + 1}")
                    break
                except Exception as e:
                    last_error = e
                    self.rate_limiter.record_failure(api_name)
                    print(f"    [ERR] {tool_name} failed: {e}")

                    if attempt < self.max_retries:
                        delay = self.rate_limiter.get_retry_delay(api_name)
                        print(f"    [RETRY] Waiting {delay:.2f}s, attempt {attempt + 2}/{self.max_retries + 1}")
                        time.sleep(min(delay, 2.0))  # Cap for demo
                        metrics["retries"] += 1

            if result is not None:
                results.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            else:
                metrics["failures"] += 1
                error_msg = f"Failed after {self.max_retries + 1} attempts: {last_error}"
                results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

        return {"messages": results, "rate_limit_metrics": metrics}


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    rate_limit_metrics: dict


# Simulated external APIs
api_call_count = defaultdict(int)


@tool
def github_api(query: str) -> str:
    """Query GitHub API. Rate limited to 60 requests/minute."""
    global api_call_count
    api_call_count["github"] += 1

    # Simulate occasional failures
    if random.random() < 0.3:
        raise ConnectionError("GitHub API temporarily unavailable")

    return f"GitHub API result for '{query}': Found 42 repositories"


@tool
def slack_api(channel: str, message: str) -> str:
    """Send message to Slack. Rate limited to 1 request/second."""
    global api_call_count
    api_call_count["slack"] += 1

    # Simulate rate limit response
    if random.random() < 0.2:
        raise Exception("rate_limited: Retry-After: 2")

    return f"Message sent to #{channel}: {message}"


@tool
def database_api(operation: str) -> str:
    """Database operation. Has circuit breaker for connection failures."""
    global api_call_count
    api_call_count["database"] += 1

    # Simulate connection failures
    if random.random() < 0.4:
        raise ConnectionError("Database connection failed")

    return f"Database operation '{operation}' completed successfully"


tools = [github_api, slack_api, database_api]


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

# Initialize rate limiter with API configurations
rate_limiter = RateLimiter()
rate_limiter.configure_api("github_api", requests_per_second=1.0, burst_capacity=5)
rate_limiter.configure_api("slack_api", requests_per_second=1.0, burst_capacity=1)
rate_limiter.configure_api("database_api", requests_per_second=10.0, failure_threshold=3)

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """Agent node."""
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "rate_limit_metrics": state.get("rate_limit_metrics", {})
    }


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    """Build graph with rate-limited tool node."""
    builder = StateGraph(State)

    builder.add_node("agent", agent)
    builder.add_node("tools", RateLimitedToolNode(tools, rate_limiter, max_retries=3))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    return builder.compile()


# =============================================================================
# TESTS
# =============================================================================

def test_rate_limiting():
    """Test rate limiting behavior."""
    print("\n" + "=" * 70)
    print("TEST: Rate Limiting with Token Bucket")
    print("=" * 70)

    bucket = TokenBucket(capacity=3, refill_rate=1.0)

    print("\nInitial capacity: 3 tokens")
    for i in range(5):
        result = bucket.acquire()
        print(f"  Request {i+1}: {'Allowed' if result else 'Denied'} (tokens: {bucket.tokens:.1f})")

    print("\nWaiting 2 seconds for refill...")
    time.sleep(2)

    for i in range(3):
        result = bucket.acquire()
        print(f"  Request {i+1}: {'Allowed' if result else 'Denied'} (tokens: {bucket.tokens:.1f})")


def test_circuit_breaker():
    """Test circuit breaker behavior."""
    print("\n" + "=" * 70)
    print("TEST: Circuit Breaker Pattern")
    print("=" * 70)

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    print(f"\nInitial state: {cb.state}")

    # Simulate failures
    for i in range(4):
        if cb.can_execute():
            print(f"  Request {i+1}: Executed (simulated failure)")
            cb.record_failure()
        else:
            print(f"  Request {i+1}: Blocked by circuit breaker")
        print(f"    State: {cb.state}, Failures: {cb.failures}")

    print("\nWaiting for recovery timeout (2s)...")
    time.sleep(2.1)

    if cb.can_execute():
        print("  Recovery request: Allowed (half-open)")
        cb.record_success()
        print(f"    State: {cb.state}")

    if cb.can_execute():
        print("  Second request: Allowed")
        cb.record_success()
        print(f"    State: {cb.state}")


def test_exponential_backoff():
    """Test exponential backoff calculation."""
    print("\n" + "=" * 70)
    print("TEST: Exponential Backoff with Jitter")
    print("=" * 70)

    backoff = ExponentialBackoff(base_delay=1.0, max_delay=30.0, jitter=0.1)

    print("\nDelay progression (base=1s, factor=2, max=30s):")
    for attempt in range(7):
        delay = backoff.get_delay(attempt)
        print(f"  Attempt {attempt}: {delay:.2f}s")


def test_integrated_rate_limiting():
    """Test full integration with LangGraph."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Rate-Limited Tool Execution")
    print("=" * 70)

    graph = build_graph()

    # Reset counters
    global api_call_count
    api_call_count = defaultdict(int)

    print("\nExecuting: Query GitHub and send Slack message")
    result = graph.invoke({
        "messages": [("user", "Search GitHub for 'langgraph' and notify #dev channel")],
        "rate_limit_metrics": {}
    })

    print(f"\nFinal response: {result['messages'][-1].content[:200]}...")

    metrics = result.get("rate_limit_metrics", {})
    print(f"\nRate Limit Metrics:")
    print(f"  Successes: {metrics.get('successes', 0)}")
    print(f"  Failures: {metrics.get('failures', 0)}")
    print(f"  Retries: {metrics.get('retries', 0)}")
    print(f"  Rate Limited: {metrics.get('rate_limited', 0)}")
    print(f"  Circuit Open: {metrics.get('circuit_open', 0)}")

    print(f"\nActual API Calls:")
    for api, count in api_call_count.items():
        print(f"  {api}: {count}")


def test_burst_handling():
    """Test handling of burst requests."""
    print("\n" + "=" * 70)
    print("TEST: Burst Request Handling")
    print("=" * 70)

    # Create new rate limiter for burst test
    burst_limiter = RateLimiter()
    burst_limiter.configure_api("burst_test", requests_per_second=2.0, burst_capacity=5)

    print("\nConfiguration: 5 burst capacity, 2 req/sec refill")
    print("Sending 10 rapid requests...\n")

    allowed = 0
    denied = 0

    for i in range(10):
        can_exec, reason = burst_limiter.can_execute("burst_test")
        status = "✓" if can_exec else "✗"
        print(f"  Request {i+1:2d}: {status} {reason}")
        if can_exec:
            allowed += 1
        else:
            denied += 1

    print(f"\nResults: {allowed} allowed, {denied} denied")
    print("First 5 allowed (burst), remaining rate-limited")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ CX-02: RATE LIMITING & RETRY - EVALUATION SUMMARY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   - Built-in rate limiting                                                  │
│   - Token bucket or leaky bucket algorithms                                 │
│   - Exponential backoff                                                     │
│   - Circuit breaker pattern                                                 │
│   - Retry-After header handling                                             │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ TokenBucket - Burst handling with refill                               │
│   ✅ ExponentialBackoff - Retry delays with jitter                          │
│   ✅ CircuitBreaker - Fail-fast pattern                                     │
│   ✅ RateLimitedToolNode - Wraps ToolNode with all above                    │
│                                                                             │
│ Production Considerations:                                                  │
│   - Track metrics per API endpoint                                          │
│   - Respect Retry-After headers from APIs                                   │
│   - Implement partial failure handling                                      │
│   - Consider distributed rate limiting (Redis)                              │
│   - Monitor circuit breaker state changes                                   │
│                                                                             │
│ VERDICT:                                                                    │
│   Fail-Close item. Custom implementation works but requires significant     │
│   effort. Essential for any production system calling external APIs.        │
│                                                                             │
│ Rating: ⭐⭐ (Experimental - with custom implementation)                     │
│   - No native support                                                       │
│   - Wrapper pattern works but adds complexity                               │
│   - Production needs distributed rate limiting                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_rate_limiting()
    test_circuit_breaker()
    test_exponential_backoff()
    test_burst_handling()
    test_integrated_rate_limiting()

    print(SUMMARY)
