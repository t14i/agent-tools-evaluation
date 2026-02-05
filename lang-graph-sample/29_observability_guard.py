"""
LangGraph Observability - SLO Manager & Cost Guard (OB-06, OB-07)
Service level objectives and budget-based kill switch.

Evaluation: OB-06 (SLO / Alerts), OB-07 (Cost Guard)
"""

import time
from typing import Annotated, TypedDict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from threading import Lock

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =============================================================================
# SLO MANAGER (OB-06)
# =============================================================================

class SLOType(Enum):
    """Types of SLO metrics."""
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective."""
    name: str
    type: SLOType
    target: float  # Target value (e.g., 200ms for latency, 0.99 for success rate)
    window_seconds: int = 300  # Rolling window for calculation
    alert_threshold: float = 0.9  # Alert if below this % of target


@dataclass
class SLOStatus:
    """Current status of an SLO."""
    slo: SLODefinition
    current_value: float
    target_value: float
    compliance: float  # 0-1, percentage meeting SLO
    is_healthy: bool
    last_updated: datetime


@dataclass
class Alert:
    """An SLO alert."""
    id: str
    severity: AlertSeverity
    slo_name: str
    message: str
    current_value: float
    target_value: float
    timestamp: datetime
    acknowledged: bool = False


class MetricsCollector:
    """Collects metrics for SLO calculation."""

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.latencies: deque = deque()
        self.success_count: int = 0
        self.error_count: int = 0
        self.request_count: int = 0
        self.lock = Lock()

    def record_latency(self, latency_ms: float, success: bool = True):
        """Record a request latency."""
        with self.lock:
            now = datetime.now()
            self.latencies.append((now, latency_ms))
            self.request_count += 1

            if success:
                self.success_count += 1
            else:
                self.error_count += 1

            # Cleanup old entries
            self._cleanup()

    def _cleanup(self):
        """Remove entries outside the window."""
        cutoff = datetime.now() - timedelta(seconds=self.window_seconds)
        while self.latencies and self.latencies[0][0] < cutoff:
            self.latencies.popleft()

    def get_percentile(self, percentile: float) -> Optional[float]:
        """Calculate latency percentile."""
        with self.lock:
            self._cleanup()
            if not self.latencies:
                return None

            sorted_latencies = sorted(l[1] for l in self.latencies)
            index = int(len(sorted_latencies) * percentile / 100)
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    def get_error_rate(self) -> float:
        """Calculate error rate."""
        with self.lock:
            total = self.success_count + self.error_count
            return self.error_count / total if total > 0 else 0

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        return 1 - self.get_error_rate()


class SLOManager:
    """
    Manages Service Level Objectives.
    Implements OB-06: SLO / Alerts.
    """

    def __init__(self):
        self.slos: dict[str, SLODefinition] = {}
        self.metrics = MetricsCollector()
        self.alerts: list[Alert] = []
        self.alert_handlers: list[Callable[[Alert], None]] = []
        self._alert_counter = 0

    def define_slo(self, slo: SLODefinition):
        """Define a new SLO."""
        self.slos[slo.name] = slo
        print(f"  [SLO] Defined: {slo.name} (target: {slo.target})")

    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request for SLO tracking."""
        self.metrics.record_latency(latency_ms, success)

        # Check SLOs
        for slo in self.slos.values():
            status = self.get_slo_status(slo.name)
            if status and not status.is_healthy:
                self._trigger_alert(slo, status)

    def get_slo_status(self, slo_name: str) -> Optional[SLOStatus]:
        """Get current status of an SLO."""
        slo = self.slos.get(slo_name)
        if not slo:
            return None

        current_value = self._get_metric_value(slo.type)
        if current_value is None:
            return None

        # Calculate compliance
        if slo.type in [SLOType.LATENCY_P50, SLOType.LATENCY_P95, SLOType.LATENCY_P99]:
            # For latency, lower is better
            compliance = min(1.0, slo.target / current_value) if current_value > 0 else 1.0
            is_healthy = current_value <= slo.target
        else:
            # For rates, higher is better
            compliance = current_value / slo.target if slo.target > 0 else 1.0
            is_healthy = current_value >= slo.target

        return SLOStatus(
            slo=slo,
            current_value=current_value,
            target_value=slo.target,
            compliance=compliance,
            is_healthy=is_healthy,
            last_updated=datetime.now()
        )

    def _get_metric_value(self, metric_type: SLOType) -> Optional[float]:
        """Get current value for a metric type."""
        if metric_type == SLOType.LATENCY_P50:
            return self.metrics.get_percentile(50)
        elif metric_type == SLOType.LATENCY_P95:
            return self.metrics.get_percentile(95)
        elif metric_type == SLOType.LATENCY_P99:
            return self.metrics.get_percentile(99)
        elif metric_type == SLOType.ERROR_RATE:
            return self.metrics.get_error_rate()
        elif metric_type == SLOType.SUCCESS_RATE:
            return self.metrics.get_success_rate()
        return None

    def _trigger_alert(self, slo: SLODefinition, status: SLOStatus):
        """Trigger an alert for SLO violation."""
        # Determine severity
        if status.compliance < 0.5:
            severity = AlertSeverity.CRITICAL
        elif status.compliance < 0.8:
            severity = AlertSeverity.ERROR
        elif status.compliance < 0.95:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        self._alert_counter += 1
        alert = Alert(
            id=f"alert_{self._alert_counter}",
            severity=severity,
            slo_name=slo.name,
            message=f"SLO violation: {slo.name} at {status.compliance:.1%} compliance",
            current_value=status.current_value,
            target_value=status.target_value,
            timestamp=datetime.now()
        )

        self.alerts.append(alert)
        print(f"  [ALERT] {severity.value.upper()}: {alert.message}")

        # Call handlers
        for handler in self.alert_handlers:
            handler(alert)

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler callback."""
        self.alert_handlers.append(handler)

    def get_all_slo_status(self) -> list[SLOStatus]:
        """Get status of all SLOs."""
        return [self.get_slo_status(name) for name in self.slos if self.get_slo_status(name)]


# =============================================================================
# COST GUARD (OB-07)
# =============================================================================

@dataclass
class CostBudget:
    """Budget definition."""
    name: str
    limit: float  # Dollar amount
    period: str  # "hourly", "daily", "monthly"
    current_spend: float = 0
    reset_at: datetime = field(default_factory=datetime.now)


@dataclass
class CostEvent:
    """A cost event."""
    timestamp: datetime
    operation: str
    cost: float
    tokens_used: int
    model: str


class CostGuard:
    """
    Cost monitoring and kill switch.
    Implements OB-07: Cost Guard.
    """

    # Approximate costs per 1K tokens
    MODEL_COSTS = {
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }

    def __init__(self):
        self.budgets: dict[str, CostBudget] = {}
        self.cost_history: deque = deque(maxlen=10000)
        self.is_killed: bool = False
        self.kill_reason: Optional[str] = None
        self.lock = Lock()
        self.alert_callbacks: list[Callable[[str, float], None]] = []

    def set_budget(self, name: str, limit: float, period: str = "daily"):
        """Set a cost budget."""
        self.budgets[name] = CostBudget(
            name=name,
            limit=limit,
            period=period,
            current_spend=0,
            reset_at=self._next_reset(period)
        )
        print(f"  [BUDGET] Set: {name} = ${limit:.2f}/{period}")

    def _next_reset(self, period: str) -> datetime:
        """Calculate next budget reset time."""
        now = datetime.now()
        if period == "hourly":
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif period == "daily":
            return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif period == "monthly":
            next_month = now.replace(day=1) + timedelta(days=32)
            return next_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return now + timedelta(days=1)

    def record_usage(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> tuple[float, bool]:
        """
        Record token usage and calculate cost.
        Returns: (cost, is_allowed)
        """
        with self.lock:
            if self.is_killed:
                return 0, False

            # Calculate cost
            model_cost = self.MODEL_COSTS.get(model, {"input": 0.01, "output": 0.03})
            cost = (input_tokens * model_cost["input"] / 1000) + \
                   (output_tokens * model_cost["output"] / 1000)

            # Record event
            event = CostEvent(
                timestamp=datetime.now(),
                operation=operation,
                cost=cost,
                tokens_used=input_tokens + output_tokens,
                model=model
            )
            self.cost_history.append(event)

            # Update budgets
            self._check_budget_resets()
            for budget in self.budgets.values():
                budget.current_spend += cost

                # Check limits
                if budget.current_spend >= budget.limit:
                    self._trigger_kill_switch(f"Budget '{budget.name}' exceeded: ${budget.current_spend:.2f} >= ${budget.limit:.2f}")
                    return cost, False

                # Warn at 80%
                if budget.current_spend >= budget.limit * 0.8:
                    self._warn(f"Budget '{budget.name}' at {budget.current_spend/budget.limit:.0%}")

            print(f"  [COST] ${cost:.4f} ({input_tokens}+{output_tokens} tokens)")
            return cost, True

    def _check_budget_resets(self):
        """Check and reset budgets if period passed."""
        now = datetime.now()
        for budget in self.budgets.values():
            if now >= budget.reset_at:
                print(f"  [BUDGET] Reset: {budget.name} (was ${budget.current_spend:.2f})")
                budget.current_spend = 0
                budget.reset_at = self._next_reset(budget.period)

    def _trigger_kill_switch(self, reason: str):
        """Activate kill switch."""
        self.is_killed = True
        self.kill_reason = reason
        print(f"\n  [KILL SWITCH] ACTIVATED: {reason}")

        for callback in self.alert_callbacks:
            callback("kill_switch", 0)

    def _warn(self, message: str):
        """Send warning."""
        print(f"  [WARNING] {message}")
        for callback in self.alert_callbacks:
            callback("budget_warning", 0)

    def reset_kill_switch(self):
        """Reset kill switch (manual override)."""
        self.is_killed = False
        self.kill_reason = None
        print("  [KILL SWITCH] Reset by manual override")

    def get_cost_summary(self) -> dict:
        """Get cost summary."""
        with self.lock:
            total_cost = sum(e.cost for e in self.cost_history)
            total_tokens = sum(e.tokens_used for e in self.cost_history)

            return {
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "budgets": {
                    name: {
                        "current": b.current_spend,
                        "limit": b.limit,
                        "remaining": b.limit - b.current_spend,
                        "percentage": b.current_spend / b.limit * 100 if b.limit > 0 else 0
                    }
                    for name, b in self.budgets.items()
                },
                "is_killed": self.is_killed,
                "kill_reason": self.kill_reason
            }

    def is_allowed(self) -> bool:
        """Check if operations are allowed (not killed)."""
        return not self.is_killed


# =============================================================================
# STATE AND TOOLS
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    metrics: dict


# Initialize managers
slo_manager = SLOManager()
cost_guard = CostGuard()

# Define SLOs
slo_manager.define_slo(SLODefinition(
    name="latency_p95",
    type=SLOType.LATENCY_P95,
    target=1000,  # 1 second
    window_seconds=60
))

slo_manager.define_slo(SLODefinition(
    name="success_rate",
    type=SLOType.SUCCESS_RATE,
    target=0.95,  # 95%
    window_seconds=60
))

# Set budgets
cost_guard.set_budget("hourly", 1.0, "hourly")
cost_guard.set_budget("daily", 10.0, "daily")


@tool
def expensive_operation(query: str) -> str:
    """Perform an expensive operation (simulates high token usage)."""
    time.sleep(0.1)
    return f"Processed expensive query: {query}"


@tool
def cheap_operation(query: str) -> str:
    """Perform a cheap operation."""
    time.sleep(0.05)
    return f"Processed cheap query: {query}"


tools = [expensive_operation, cheap_operation]

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def guarded_agent(state: State) -> State:
    """Agent node with cost guard and SLO tracking."""
    # Check kill switch
    if not cost_guard.is_allowed():
        raise RuntimeError(f"Operations blocked: {cost_guard.kill_reason}")

    start_time = time.time()
    success = True

    try:
        response = llm_with_tools.invoke(state["messages"])

        # Simulate token counts (in production, get from response)
        input_tokens = len(str(state["messages"])) // 4
        output_tokens = len(str(response.content)) // 4

        # Record cost
        cost, allowed = cost_guard.record_usage(
            "agent_call",
            "claude-sonnet-4-20250514",
            input_tokens,
            output_tokens
        )

        if not allowed:
            raise RuntimeError("Cost limit exceeded")

    except Exception as e:
        success = False
        raise

    finally:
        # Record metrics for SLO
        latency_ms = (time.time() - start_time) * 1000
        slo_manager.record_request(latency_ms, success)

    # Update metrics in state
    metrics = state.get("metrics", {})
    metrics["last_latency_ms"] = latency_ms
    metrics["total_requests"] = metrics.get("total_requests", 0) + 1

    return {"messages": [response], "metrics": metrics}


def should_continue(state: State) -> str:
    """Conditional edge."""
    # Check kill switch before continuing
    if not cost_guard.is_allowed():
        return END

    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    """Build graph with guards."""
    builder = StateGraph(State)

    builder.add_node("agent", guarded_agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    return builder.compile()


# =============================================================================
# TESTS
# =============================================================================

def test_slo_tracking():
    """Test SLO tracking (OB-06)."""
    print("\n" + "=" * 70)
    print("TEST: SLO Tracking (OB-06)")
    print("=" * 70)

    # Reset metrics
    slo_manager.metrics = MetricsCollector()

    # Simulate requests
    print("\n--- Simulating requests ---")
    latencies = [100, 150, 200, 300, 500, 800, 1200, 150, 200, 250]

    for latency in latencies:
        success = latency < 1000
        slo_manager.record_request(latency, success)
        print(f"  Request: {latency}ms, success={success}")

    # Check SLO status
    print("\n--- SLO Status ---")
    for status in slo_manager.get_all_slo_status():
        health = "✅" if status.is_healthy else "❌"
        print(f"  {health} {status.slo.name}: {status.current_value:.2f} "
              f"(target: {status.target_value}, compliance: {status.compliance:.1%})")


def test_cost_tracking():
    """Test cost tracking (OB-07)."""
    print("\n" + "=" * 70)
    print("TEST: Cost Tracking (OB-07)")
    print("=" * 70)

    # Reset cost guard
    cost_guard.is_killed = False
    cost_guard.kill_reason = None
    for budget in cost_guard.budgets.values():
        budget.current_spend = 0

    # Simulate usage
    print("\n--- Simulating API calls ---")
    operations = [
        ("chat", 500, 100),
        ("chat", 1000, 200),
        ("chat", 2000, 500),
    ]

    for op, input_tokens, output_tokens in operations:
        cost, allowed = cost_guard.record_usage(
            op, "claude-sonnet-4-20250514", input_tokens, output_tokens
        )
        print(f"  Operation: {op}, Cost: ${cost:.4f}, Allowed: {allowed}")

    # Get summary
    print("\n--- Cost Summary ---")
    summary = cost_guard.get_cost_summary()
    print(f"  Total cost: ${summary['total_cost']:.4f}")
    print(f"  Total tokens: {summary['total_tokens']}")
    for name, budget in summary['budgets'].items():
        print(f"  Budget '{name}': ${budget['current']:.4f}/${budget['limit']:.2f} ({budget['percentage']:.1f}%)")


def test_kill_switch():
    """Test cost guard kill switch (OB-07)."""
    print("\n" + "=" * 70)
    print("TEST: Kill Switch (OB-07)")
    print("=" * 70)

    # Set a very low budget
    cost_guard.set_budget("test_budget", 0.01, "hourly")

    # Reset state
    cost_guard.is_killed = False
    cost_guard.kill_reason = None
    cost_guard.budgets["test_budget"].current_spend = 0

    # Trigger kill switch
    print("\n--- Triggering kill switch ---")
    cost, allowed = cost_guard.record_usage("test", "claude-sonnet-4-20250514", 10000, 5000)

    print(f"\nKill switch active: {cost_guard.is_killed}")
    print(f"Reason: {cost_guard.kill_reason}")

    # Try another operation
    cost, allowed = cost_guard.record_usage("test2", "claude-sonnet-4-20250514", 100, 50)
    print(f"Second operation allowed: {allowed}")

    # Reset
    cost_guard.reset_kill_switch()
    print(f"\nAfter reset: is_killed={cost_guard.is_killed}")


def test_slo_alerts():
    """Test SLO alerting."""
    print("\n" + "=" * 70)
    print("TEST: SLO Alerts")
    print("=" * 70)

    # Reset
    slo_manager.metrics = MetricsCollector()
    slo_manager.alerts.clear()

    # Simulate SLO violations
    print("\n--- Simulating SLO violations ---")
    for _ in range(10):
        slo_manager.record_request(2000, False)  # Slow and failed

    print(f"\nAlerts generated: {len(slo_manager.alerts)}")
    for alert in slo_manager.alerts[-3:]:
        print(f"  [{alert.severity.value}] {alert.message}")


def test_integrated_guards():
    """Test integrated cost and SLO guards."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Guards")
    print("=" * 70)

    # Reset
    cost_guard.is_killed = False
    cost_guard.kill_reason = None
    for budget in cost_guard.budgets.values():
        budget.current_spend = 0
    slo_manager.metrics = MetricsCollector()

    graph = build_graph()

    try:
        result = graph.invoke({
            "messages": [("user", "Perform a cheap operation on 'test data'")],
            "metrics": {}
        })
        print(f"\nResult: {result['messages'][-1].content[:100]}...")
        print(f"Metrics: {result.get('metrics', {})}")

    except RuntimeError as e:
        print(f"\nOperation blocked: {e}")

    # Show final status
    print("\n--- Final Status ---")
    summary = cost_guard.get_cost_summary()
    print(f"Cost: ${summary['total_cost']:.4f}")
    print(f"Killed: {summary['is_killed']}")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ OB-06 & OB-07: SLO & COST GUARD - EVALUATION SUMMARY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ SLO definition/tracking                                                │
│   ❌ Metrics collection                                                     │
│   ❌ Alerting system                                                        │
│   ❌ Budget management                                                      │
│   ❌ Kill switch mechanism                                                  │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ MetricsCollector - Latency and error tracking                          │
│   ✅ SLOManager - Define and monitor SLOs                                   │
│   ✅ CostGuard - Budget tracking and kill switch                            │
│   ✅ Alert system - Callbacks on violations                                 │
│                                                                             │
│ OB-06 (SLO / Alerts) Features:                                              │
│   ✓ Latency percentiles (P50, P95, P99)                                     │
│   ✓ Error/success rate tracking                                             │
│   ✓ Rolling window calculations                                             │
│   ✓ Compliance percentage                                                   │
│   ✓ Multi-severity alerts                                                   │
│   ✓ Alert handlers/callbacks                                                │
│                                                                             │
│ OB-07 (Cost Guard) Features:                                                │
│   ✓ Per-model cost calculation                                              │
│   ✓ Multiple budget periods (hourly, daily, monthly)                        │
│   ✓ Automatic budget resets                                                 │
│   ✓ Warning at threshold (80%)                                              │
│   ✓ Kill switch on budget exceeded                                          │
│   ✓ Manual override/reset                                                   │
│                                                                             │
│ Production Considerations:                                                  │
│   - Integrate with Prometheus/Grafana                                       │
│   - Use proper alerting (PagerDuty, OpsGenie)                               │
│   - Distributed metrics aggregation                                         │
│   - Graceful degradation instead of hard kill                               │
│   - Cost attribution per tenant/user                                        │
│                                                                             │
│ Rating:                                                                     │
│   OB-06 (SLO): ⭐ (fully custom)                                            │
│   OB-07 (Cost Guard): ⭐ (fully custom)                                     │
│                                                                             │
│   Critical for production safety but requires complete custom build.        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_slo_tracking()
    test_cost_tracking()
    test_kill_switch()
    test_slo_alerts()
    test_integrated_guards()

    print(SUMMARY)
