"""
23_observability_guard.py - Observability Guard (OB-06, OB-07)

Purpose: Verify SLO/alerts and cost guard
- OB-06: SLO thresholds for failure rate, latency, cost
- OB-07: Cost guard with budget limits and kill switch

LangGraph Comparison:
- Neither has built-in cost/SLO management
- Both require custom implementation
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Any
from enum import Enum


# =============================================================================
# Metrics Collection
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """A single metric data point."""

    name: str
    type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and stores metrics for SLO monitoring.
    """

    def __init__(self, retention_seconds: int = 3600):
        self.metrics: dict[str, deque] = {}
        self.retention_seconds = retention_seconds

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[dict] = None,
    ):
        """Record a metric."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=10000)

        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {},
        )

        self.metrics[name].append(metric)
        self._cleanup(name)

    def increment(self, name: str, amount: float = 1.0, labels: Optional[dict] = None):
        """Increment a counter metric."""
        self.record(name, amount, MetricType.COUNTER, labels)

    def _cleanup(self, name: str):
        """Remove old metrics beyond retention period."""
        cutoff = time.time() - self.retention_seconds
        while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
            self.metrics[name].popleft()

    def get_metrics(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> list[Metric]:
        """Get metrics within time range."""
        if name not in self.metrics:
            return []

        metrics = list(self.metrics[name])

        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        return metrics

    def get_sum(self, name: str, window_seconds: int = 60) -> float:
        """Get sum of metric values in time window."""
        start_time = time.time() - window_seconds
        metrics = self.get_metrics(name, start_time=start_time)
        return sum(m.value for m in metrics)

    def get_average(self, name: str, window_seconds: int = 60) -> Optional[float]:
        """Get average of metric values in time window."""
        start_time = time.time() - window_seconds
        metrics = self.get_metrics(name, start_time=start_time)
        if not metrics:
            return None
        return sum(m.value for m in metrics) / len(metrics)

    def get_percentile(self, name: str, percentile: float, window_seconds: int = 60) -> Optional[float]:
        """Get percentile of metric values."""
        start_time = time.time() - window_seconds
        metrics = self.get_metrics(name, start_time=start_time)
        if not metrics:
            return None

        values = sorted(m.value for m in metrics)
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]


# =============================================================================
# SLO Definitions (OB-06)
# =============================================================================

class SLOType(str, Enum):
    """Types of SLO."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    COST = "cost"


@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str
    type: SLOType
    target: float  # Target value (e.g., 99.9 for availability)
    window_seconds: int = 3600  # Evaluation window
    metric_name: str = ""
    comparison: str = ">="  # >=, <=, >, <
    alert_threshold: Optional[float] = None  # Trigger alert if below this
    description: str = ""


@dataclass
class SLOStatus:
    """Current status of an SLO."""

    slo: SLO
    current_value: Optional[float]
    is_met: bool
    error_budget_remaining: float  # Percentage
    last_evaluated: float = field(default_factory=time.time)


class SLOManager:
    """
    Manages Service Level Objectives.

    Tracks SLO compliance and error budgets.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.slos: dict[str, SLO] = {}
        self.alert_handlers: list[Callable] = []

    def register_slo(self, slo: SLO):
        """Register an SLO."""
        self.slos[slo.name] = slo
        print(f"[SLO] Registered: {slo.name} ({slo.type.value})")

    def add_alert_handler(self, handler: Callable[[SLO, SLOStatus], None]):
        """Add a handler for SLO alerts."""
        self.alert_handlers.append(handler)

    def evaluate(self, slo_name: str) -> Optional[SLOStatus]:
        """Evaluate an SLO against current metrics."""
        if slo_name not in self.slos:
            return None

        slo = self.slos[slo_name]
        current_value = self._calculate_value(slo)

        if current_value is None:
            return SLOStatus(
                slo=slo,
                current_value=None,
                is_met=True,  # No data, assume OK
                error_budget_remaining=100.0,
            )

        is_met = self._compare(current_value, slo.target, slo.comparison)
        error_budget = self._calculate_error_budget(current_value, slo)

        status = SLOStatus(
            slo=slo,
            current_value=current_value,
            is_met=is_met,
            error_budget_remaining=error_budget,
        )

        # Check for alerts
        if slo.alert_threshold is not None:
            should_alert = not self._compare(current_value, slo.alert_threshold, slo.comparison)
            if should_alert:
                self._trigger_alerts(slo, status)

        return status

    def evaluate_all(self) -> dict[str, SLOStatus]:
        """Evaluate all registered SLOs."""
        return {name: self.evaluate(name) for name in self.slos}

    def _calculate_value(self, slo: SLO) -> Optional[float]:
        """Calculate current SLO value from metrics."""
        if slo.type == SLOType.AVAILABILITY:
            total = self.metrics.get_sum(f"{slo.metric_name}_total", slo.window_seconds)
            errors = self.metrics.get_sum(f"{slo.metric_name}_errors", slo.window_seconds)
            if total == 0:
                return None
            return ((total - errors) / total) * 100

        elif slo.type == SLOType.ERROR_RATE:
            total = self.metrics.get_sum(f"{slo.metric_name}_total", slo.window_seconds)
            errors = self.metrics.get_sum(f"{slo.metric_name}_errors", slo.window_seconds)
            if total == 0:
                return None
            return (errors / total) * 100

        elif slo.type == SLOType.LATENCY:
            return self.metrics.get_percentile(slo.metric_name, 95, slo.window_seconds)

        elif slo.type == SLOType.COST:
            return self.metrics.get_sum(slo.metric_name, slo.window_seconds)

        return self.metrics.get_average(slo.metric_name, slo.window_seconds)

    def _compare(self, value: float, target: float, comparison: str) -> bool:
        """Compare value against target."""
        if comparison == ">=":
            return value >= target
        elif comparison == "<=":
            return value <= target
        elif comparison == ">":
            return value > target
        elif comparison == "<":
            return value < target
        return False

    def _calculate_error_budget(self, current: float, slo: SLO) -> float:
        """Calculate remaining error budget percentage."""
        if slo.comparison in [">=", ">"]:
            if current >= slo.target:
                return 100.0
            # How far below target
            max_error = 100 - slo.target
            if max_error == 0:
                return 0.0
            error_used = slo.target - current
            return max(0, (max_error - error_used) / max_error * 100)
        else:
            if current <= slo.target:
                return 100.0
            # How far above target
            over = current - slo.target
            return max(0, 100 - over)

    def _trigger_alerts(self, slo: SLO, status: SLOStatus):
        """Trigger alert handlers."""
        print(f"[ALERT] SLO '{slo.name}' breached! Value: {status.current_value:.2f}, Target: {slo.target}")
        for handler in self.alert_handlers:
            try:
                handler(slo, status)
            except Exception as e:
                print(f"[ALERT] Handler error: {e}")


# =============================================================================
# Cost Guard (OB-07)
# =============================================================================

@dataclass
class CostBudget:
    """Cost budget configuration."""

    name: str
    limit: float  # Maximum allowed cost
    period: str = "daily"  # daily, weekly, monthly
    warning_threshold: float = 0.8  # Warn at 80%
    kill_switch_enabled: bool = True
    reset_time: Optional[datetime] = None


class CostTracker:
    """
    Tracks costs for budget management.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./db/costs")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.costs: dict[str, float] = {}  # category -> accumulated cost
        self.cost_log: list[dict] = []

    def record_cost(
        self,
        amount: float,
        category: str = "default",
        description: str = "",
        metadata: Optional[dict] = None,
    ):
        """Record a cost."""
        if category not in self.costs:
            self.costs[category] = 0.0

        self.costs[category] += amount

        entry = {
            "timestamp": datetime.now().isoformat(),
            "amount": amount,
            "category": category,
            "description": description,
            "metadata": metadata or {},
        }
        self.cost_log.append(entry)

        print(f"[Cost] +${amount:.4f} ({category}) - Total: ${self.costs[category]:.4f}")

    def get_total(self, category: Optional[str] = None) -> float:
        """Get total cost, optionally by category."""
        if category:
            return self.costs.get(category, 0.0)
        return sum(self.costs.values())

    def get_costs_since(self, since: datetime) -> float:
        """Get costs since a specific time."""
        total = 0.0
        for entry in self.cost_log:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= since:
                total += entry["amount"]
        return total

    def reset(self, category: Optional[str] = None):
        """Reset costs."""
        if category:
            self.costs[category] = 0.0
        else:
            self.costs = {}
        print(f"[Cost] Reset {'all' if not category else category}")


class CostGuard:
    """
    Cost guard with budget limits and kill switch.

    Prevents runaway costs by:
    - Monitoring against budgets
    - Sending warnings at thresholds
    - Triggering kill switch when budget exceeded
    """

    def __init__(self, cost_tracker: CostTracker):
        self.tracker = cost_tracker
        self.budgets: dict[str, CostBudget] = {}
        self.is_killed: bool = False
        self.kill_reason: str = ""
        self.warning_handlers: list[Callable] = []
        self.kill_handlers: list[Callable] = []

    def set_budget(self, budget: CostBudget):
        """Set a cost budget."""
        self.budgets[budget.name] = budget
        print(f"[CostGuard] Budget set: {budget.name} = ${budget.limit:.2f}/{budget.period}")

    def add_warning_handler(self, handler: Callable[[CostBudget, float], None]):
        """Add handler for budget warnings."""
        self.warning_handlers.append(handler)

    def add_kill_handler(self, handler: Callable[[CostBudget, float], None]):
        """Add handler for kill switch activation."""
        self.kill_handlers.append(handler)

    def check_budget(self, budget_name: str) -> dict:
        """Check a specific budget."""
        if budget_name not in self.budgets:
            return {"error": "Budget not found"}

        budget = self.budgets[budget_name]
        current = self._get_period_cost(budget)
        percentage = (current / budget.limit) * 100 if budget.limit > 0 else 0

        result = {
            "budget_name": budget_name,
            "limit": budget.limit,
            "current": current,
            "percentage": percentage,
            "remaining": max(0, budget.limit - current),
            "status": "ok",
        }

        if percentage >= 100:
            result["status"] = "exceeded"
            if budget.kill_switch_enabled:
                self._trigger_kill(budget, current)
        elif percentage >= budget.warning_threshold * 100:
            result["status"] = "warning"
            self._trigger_warning(budget, current)

        return result

    def check_all_budgets(self) -> dict[str, dict]:
        """Check all budgets."""
        return {name: self.check_budget(name) for name in self.budgets}

    def can_proceed(self, estimated_cost: float = 0) -> tuple[bool, str]:
        """
        Check if operation can proceed within budget.

        Returns (can_proceed, reason)
        """
        if self.is_killed:
            return False, f"Kill switch active: {self.kill_reason}"

        for name, budget in self.budgets.items():
            current = self._get_period_cost(budget)
            if current + estimated_cost > budget.limit:
                if budget.kill_switch_enabled:
                    return False, f"Would exceed budget '{name}'"

        return True, "OK"

    def _get_period_cost(self, budget: CostBudget) -> float:
        """Get cost for budget period."""
        now = datetime.now()

        if budget.period == "daily":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget.period == "weekly":
            since = now - timedelta(days=now.weekday())
            since = since.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget.period == "monthly":
            since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            since = datetime.min

        return self.tracker.get_costs_since(since)

    def _trigger_warning(self, budget: CostBudget, current: float):
        """Trigger warning handlers."""
        print(f"[WARNING] Budget '{budget.name}' at {current/budget.limit*100:.1f}%")
        for handler in self.warning_handlers:
            try:
                handler(budget, current)
            except Exception as e:
                print(f"[WARNING] Handler error: {e}")

    def _trigger_kill(self, budget: CostBudget, current: float):
        """Trigger kill switch."""
        self.is_killed = True
        self.kill_reason = f"Budget '{budget.name}' exceeded: ${current:.2f} > ${budget.limit:.2f}"
        print(f"[KILL SWITCH] {self.kill_reason}")

        for handler in self.kill_handlers:
            try:
                handler(budget, current)
            except Exception as e:
                print(f"[KILL] Handler error: {e}")

    def reset_kill_switch(self):
        """Manually reset the kill switch."""
        self.is_killed = False
        self.kill_reason = ""
        print("[CostGuard] Kill switch reset")


# =============================================================================
# Token Cost Calculator
# =============================================================================

class TokenCostCalculator:
    """
    Calculates costs based on token usage.
    """

    # Pricing per 1M tokens (example prices)
    PRICING = {
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    @classmethod
    def calculate(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for token usage."""
        if model not in cls.PRICING:
            print(f"[Cost] Unknown model: {model}, using default pricing")
            pricing = {"input": 1.00, "output": 2.00}
        else:
            pricing = cls.PRICING[model]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


# =============================================================================
# Alert Manager
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert notification."""

    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    acknowledged: bool = False


class AlertManager:
    """
    Manages alerts and notifications.
    """

    def __init__(self):
        self.alerts: list[Alert] = []
        self.handlers: dict[AlertSeverity, list[Callable]] = {
            s: [] for s in AlertSeverity
        }

    def add_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Add alert handler for severity level."""
        self.handlers[severity].append(handler)

    def fire(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: Optional[dict] = None,
    ) -> Alert:
        """Fire an alert."""
        import uuid
        alert = Alert(
            id=str(uuid.uuid4())[:8],
            severity=severity,
            title=title,
            message=message,
            metadata=metadata or {},
        )

        self.alerts.append(alert)
        print(f"[ALERT:{severity.value.upper()}] {title}: {message}")

        # Trigger handlers
        for handler in self.handlers[severity]:
            try:
                handler(alert)
            except Exception as e:
                print(f"[Alert] Handler error: {e}")

        return alert

    def acknowledge(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                print(f"[Alert] Acknowledged: {alert_id}")
                return

    def get_active_alerts(self) -> list[Alert]:
        """Get unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]


# =============================================================================
# Demonstrations
# =============================================================================

def demo_metrics_collection():
    """Demonstrate metrics collection."""
    print("=" * 60)
    print("Demo 1: Metrics Collection")
    print("=" * 60)

    collector = MetricsCollector()

    # Record some metrics
    for i in range(10):
        collector.record("request_latency", 100 + i * 10)
        collector.increment("requests_total")
        if i % 3 == 0:
            collector.increment("requests_errors")
        time.sleep(0.01)

    # Query metrics
    print(f"\nTotal requests: {collector.get_sum('requests_total')}")
    print(f"Total errors: {collector.get_sum('requests_errors')}")
    print(f"Avg latency: {collector.get_average('request_latency'):.2f}ms")
    print(f"P95 latency: {collector.get_percentile('request_latency', 95):.2f}ms")


def demo_slo_management():
    """Demonstrate SLO management."""
    print("\n" + "=" * 60)
    print("Demo 2: SLO Management (OB-06)")
    print("=" * 60)

    collector = MetricsCollector()
    slo_manager = SLOManager(collector)

    # Register SLOs
    slo_manager.register_slo(SLO(
        name="api_availability",
        type=SLOType.AVAILABILITY,
        target=99.9,
        metric_name="api",
        window_seconds=60,
        alert_threshold=99.5,
        description="API must be 99.9% available",
    ))

    slo_manager.register_slo(SLO(
        name="response_latency",
        type=SLOType.LATENCY,
        target=500,
        metric_name="latency_ms",
        comparison="<=",
        window_seconds=60,
        alert_threshold=1000,
        description="P95 latency must be under 500ms",
    ))

    # Add alert handler
    def alert_handler(slo: SLO, status: SLOStatus):
        print(f"  -> Alert handler called for {slo.name}")

    slo_manager.add_alert_handler(alert_handler)

    # Simulate metrics
    for i in range(100):
        collector.increment("api_total")
        if i % 50 == 0:  # 2% error rate
            collector.increment("api_errors")
        collector.record("latency_ms", 100 + (i % 20) * 30)

    # Evaluate SLOs
    print("\nSLO Evaluation:")
    print("-" * 40)
    results = slo_manager.evaluate_all()
    for name, status in results.items():
        if status.current_value is not None:
            print(f"{name}:")
            print(f"  Current: {status.current_value:.2f}")
            print(f"  Target: {status.slo.target}")
            print(f"  Met: {status.is_met}")
            print(f"  Error Budget: {status.error_budget_remaining:.1f}%")


def demo_cost_tracking():
    """Demonstrate cost tracking."""
    print("\n" + "=" * 60)
    print("Demo 3: Cost Tracking (OB-07)")
    print("=" * 60)

    tracker = CostTracker()

    # Record some costs
    tracker.record_cost(0.05, "llm", "GPT-4 call", {"model": "gpt-4", "tokens": 1000})
    tracker.record_cost(0.01, "llm", "GPT-3.5 call", {"model": "gpt-3.5-turbo", "tokens": 500})
    tracker.record_cost(0.02, "embedding", "Embedding generation")
    tracker.record_cost(0.03, "llm", "Another GPT-4 call")

    print(f"\nTotal cost: ${tracker.get_total():.4f}")
    print(f"LLM cost: ${tracker.get_total('llm'):.4f}")
    print(f"Embedding cost: ${tracker.get_total('embedding'):.4f}")


def demo_cost_guard():
    """Demonstrate cost guard with budget limits."""
    print("\n" + "=" * 60)
    print("Demo 4: Cost Guard with Kill Switch (OB-07)")
    print("=" * 60)

    tracker = CostTracker()
    guard = CostGuard(tracker)

    # Set budget
    guard.set_budget(CostBudget(
        name="daily_llm",
        limit=0.10,  # $0.10 daily limit for demo
        period="daily",
        warning_threshold=0.5,
        kill_switch_enabled=True,
    ))

    # Add handlers
    def warning_handler(budget, current):
        print(f"  -> Warning handler: ${current:.4f} of ${budget.limit:.2f}")

    def kill_handler(budget, current):
        print(f"  -> KILL handler activated!")

    guard.add_warning_handler(warning_handler)
    guard.add_kill_handler(kill_handler)

    # Simulate costs
    print("\nSimulating LLM costs...")
    for i in range(15):
        can_proceed, reason = guard.can_proceed(0.01)
        if not can_proceed:
            print(f"\nCall {i+1}: BLOCKED - {reason}")
            break

        tracker.record_cost(0.01, "llm", f"Call {i+1}")
        status = guard.check_budget("daily_llm")
        print(f"  Budget status: {status['status']} ({status['percentage']:.1f}%)")


def demo_token_cost_calculator():
    """Demonstrate token cost calculation."""
    print("\n" + "=" * 60)
    print("Demo 5: Token Cost Calculator")
    print("=" * 60)

    models = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]

    print("\nCost for 1000 input + 500 output tokens:")
    print("-" * 40)

    for model in models:
        cost = TokenCostCalculator.calculate(model, 1000, 500)
        print(f"{model:20}: ${cost:.6f}")


def demo_alert_manager():
    """Demonstrate alert management."""
    print("\n" + "=" * 60)
    print("Demo 6: Alert Manager")
    print("=" * 60)

    manager = AlertManager()

    # Add handlers
    def critical_handler(alert):
        print(f"  -> CRITICAL alert received: {alert.title}")

    manager.add_handler(AlertSeverity.CRITICAL, critical_handler)

    # Fire alerts
    manager.fire(AlertSeverity.INFO, "System Started", "CrewAI system initialized")
    manager.fire(AlertSeverity.WARNING, "High Latency", "P95 latency above 500ms")
    alert = manager.fire(AlertSeverity.CRITICAL, "Budget Exceeded", "Daily budget limit reached")

    # Show active alerts
    print(f"\nActive alerts: {len(manager.get_active_alerts())}")

    # Acknowledge
    manager.acknowledge(alert.id)
    print(f"After acknowledgment: {len(manager.get_active_alerts())} active")


def demo_integrated_monitoring():
    """Demonstrate integrated monitoring system."""
    print("\n" + "=" * 60)
    print("Demo 7: Integrated Monitoring")
    print("=" * 60)

    # Set up components
    collector = MetricsCollector()
    slo_manager = SLOManager(collector)
    tracker = CostTracker()
    guard = CostGuard(tracker)
    alerts = AlertManager()

    # Configure
    slo_manager.register_slo(SLO(
        name="error_rate",
        type=SLOType.ERROR_RATE,
        target=5.0,  # 5% max error rate
        metric_name="ops",
        comparison="<=",
        window_seconds=60,
    ))

    guard.set_budget(CostBudget(
        name="operation_budget",
        limit=0.05,
        period="daily",
    ))

    # Simulate operations
    print("\nSimulating 20 operations...")
    for i in range(20):
        # Check budget
        can_proceed, _ = guard.can_proceed(0.002)
        if not can_proceed:
            alerts.fire(AlertSeverity.CRITICAL, "Budget Exceeded", "Stopping operations")
            break

        # Record metrics
        collector.increment("ops_total")
        if i % 10 == 0:
            collector.increment("ops_errors")
            alerts.fire(AlertSeverity.WARNING, "Operation Error", f"Operation {i} failed")

        # Record cost
        tracker.record_cost(0.002, "ops", f"Operation {i}")

    # Final status
    print("\n--- Final Status ---")
    slo_status = slo_manager.evaluate("error_rate")
    if slo_status and slo_status.current_value:
        print(f"Error rate: {slo_status.current_value:.2f}% (target: {slo_status.slo.target}%)")

    budget_status = guard.check_budget("operation_budget")
    print(f"Budget used: ${budget_status['current']:.4f} of ${budget_status['limit']:.2f}")
    print(f"Active alerts: {len(alerts.get_active_alerts())}")


def main():
    print("=" * 60)
    print("Observability Guard Verification (OB-06, OB-07)")
    print("=" * 60)
    print("""
This script verifies SLO and cost guard capabilities.

Verification Items:
- OB-06: SLO/Alert Thresholds
  - Failure rate monitoring
  - Latency percentile tracking
  - Error budget calculation
  - Automatic alerting

- OB-07: Cost Guard
  - Budget limit enforcement
  - Token cost calculation
  - Kill switch mechanism
  - Usage tracking

Key Components:
- MetricsCollector: Time-series metrics storage
- SLOManager: SLO definition and evaluation
- CostTracker: Cost accumulation tracking
- CostGuard: Budget enforcement with kill switch
- AlertManager: Alert firing and acknowledgment

LangGraph Comparison:
- Neither has built-in cost management
- Both require custom implementation
- LangSmith provides some observability features

Production Considerations:
- Integrate with Prometheus/Grafana for metrics
- Use PagerDuty/OpsGenie for alerting
- Implement proper cost tracking per tenant
""")

    # Run all demos
    demo_metrics_collection()
    demo_slo_management()
    demo_cost_tracking()
    demo_cost_guard()
    demo_token_cost_calculator()
    demo_alert_manager()
    demo_integrated_monitoring()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
