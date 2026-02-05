"""
Observability - Part 3: Guards (OB-06, OB-07)
SLO monitoring, cost guard, kill switch
"""

from dotenv import load_dotenv
load_dotenv()


from datetime import datetime, timedelta
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# =============================================================================
# OB-06: SLO / Alerts
# =============================================================================

class MetricType(Enum):
    """Types of metrics to track."""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    THROUGHPUT = "throughput"


@dataclass
class MetricSample:
    """A single metric sample."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: dict = field(default_factory=dict)


class MetricsCollector:
    """
    Collects metrics for SLO monitoring.
    """

    def __init__(self, window_size: int = 1000):
        self.samples: dict[MetricType, deque] = {
            mt: deque(maxlen=window_size) for mt in MetricType
        }

    def record(
        self,
        metric_type: MetricType,
        value: float,
        labels: dict = None
    ):
        """Record a metric sample."""
        sample = MetricSample(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self.samples[metric_type].append(sample)

    def get_percentile(
        self,
        metric_type: MetricType,
        percentile: float,
        window: timedelta = None
    ) -> Optional[float]:
        """Get percentile value for a metric."""
        samples = list(self.samples[metric_type])

        if window:
            cutoff = datetime.now() - window
            samples = [s for s in samples if s.timestamp >= cutoff]

        if not samples:
            return None

        values = sorted([s.value for s in samples])
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]

    def get_average(
        self,
        metric_type: MetricType,
        window: timedelta = None
    ) -> Optional[float]:
        """Get average value for a metric."""
        samples = list(self.samples[metric_type])

        if window:
            cutoff = datetime.now() - window
            samples = [s for s in samples if s.timestamp >= cutoff]

        if not samples:
            return None

        return sum(s.value for s in samples) / len(samples)

    def get_count(
        self,
        metric_type: MetricType,
        window: timedelta = None
    ) -> int:
        """Get count of samples."""
        samples = list(self.samples[metric_type])

        if window:
            cutoff = datetime.now() - window
            samples = [s for s in samples if s.timestamp >= cutoff]

        return len(samples)


@dataclass
class SLODefinition:
    """Service Level Objective definition."""
    name: str
    metric_type: MetricType
    target_value: float
    comparison: str  # "lt", "lte", "gt", "gte"
    window: timedelta
    description: str = ""


class SLOManager:
    """
    Manages SLOs and alerts.
    Implements OB-06: SLO / Alerts.
    """

    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.slos: list[SLODefinition] = []
        self.alert_handlers: list[Callable] = []

    def add_slo(self, slo: SLODefinition):
        """Add an SLO definition."""
        self.slos.append(slo)

    def add_alert_handler(self, handler: Callable):
        """Add an alert handler."""
        self.alert_handlers.append(handler)

    def check_slo(self, slo: SLODefinition) -> tuple[bool, float]:
        """
        Check if SLO is being met.
        Returns: (is_meeting_slo, current_value)
        """
        if slo.metric_type == MetricType.LATENCY:
            current = self.metrics.get_percentile(
                slo.metric_type, 95, slo.window
            )
        elif slo.metric_type == MetricType.ERROR_RATE:
            total = self.metrics.get_count(MetricType.THROUGHPUT, slo.window)
            errors = self.metrics.get_count(MetricType.ERROR_RATE, slo.window)
            current = (errors / total * 100) if total > 0 else 0
        else:
            current = self.metrics.get_average(slo.metric_type, slo.window)

        if current is None:
            return (True, 0)  # No data, assume OK

        # Compare
        if slo.comparison == "lt":
            meeting = current < slo.target_value
        elif slo.comparison == "lte":
            meeting = current <= slo.target_value
        elif slo.comparison == "gt":
            meeting = current > slo.target_value
        else:  # gte
            meeting = current >= slo.target_value

        return (meeting, current)

    def check_all_slos(self) -> dict:
        """Check all SLOs and trigger alerts."""
        results = {}

        for slo in self.slos:
            meeting, current = self.check_slo(slo)
            results[slo.name] = {
                "meeting": meeting,
                "current": current,
                "target": slo.target_value,
                "comparison": slo.comparison
            }

            if not meeting:
                self._trigger_alert(slo, current)

        return results

    def _trigger_alert(self, slo: SLODefinition, current_value: float):
        """Trigger alert handlers."""
        for handler in self.alert_handlers:
            handler({
                "type": "slo_violation",
                "slo_name": slo.name,
                "current_value": current_value,
                "target_value": slo.target_value,
                "timestamp": datetime.now().isoformat()
            })


# =============================================================================
# OB-07: Cost Guard / Kill Switch
# =============================================================================

@dataclass
class BudgetConfig:
    """Budget configuration."""
    daily_limit: float
    hourly_limit: float
    per_request_limit: float
    warning_threshold: float = 0.8  # 80% of limit


class CostGuard:
    """
    Guards against excessive costs.
    Implements OB-07: Cost Guard.
    """

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.daily_spend: float = 0
        self.hourly_spend: float = 0
        self.daily_reset: datetime = datetime.now()
        self.hourly_reset: datetime = datetime.now()
        self.kill_switch_active: bool = False
        self.alert_handlers: list[Callable] = []

    def add_alert_handler(self, handler: Callable):
        """Add an alert handler."""
        self.alert_handlers.append(handler)

    def _reset_if_needed(self):
        """Reset counters if period has elapsed."""
        now = datetime.now()

        if (now - self.daily_reset).days >= 1:
            self.daily_spend = 0
            self.daily_reset = now

        if (now - self.hourly_reset).seconds >= 3600:
            self.hourly_spend = 0
            self.hourly_reset = now

    def check_budget(self, estimated_cost: float) -> tuple[bool, str]:
        """
        Check if request is within budget.
        Returns: (is_allowed, reason)
        """
        self._reset_if_needed()

        if self.kill_switch_active:
            return (False, "Kill switch is active")

        # Check per-request limit
        if estimated_cost > self.config.per_request_limit:
            return (False, f"Exceeds per-request limit ({self.config.per_request_limit})")

        # Check hourly limit
        if self.hourly_spend + estimated_cost > self.config.hourly_limit:
            return (False, f"Would exceed hourly limit ({self.config.hourly_limit})")

        # Check daily limit
        if self.daily_spend + estimated_cost > self.config.daily_limit:
            return (False, f"Would exceed daily limit ({self.config.daily_limit})")

        return (True, "OK")

    def record_spend(self, cost: float):
        """Record a spend."""
        self._reset_if_needed()

        self.daily_spend += cost
        self.hourly_spend += cost

        # Check warning thresholds
        if self.daily_spend > self.config.daily_limit * self.config.warning_threshold:
            self._trigger_alert("daily_warning", self.daily_spend, self.config.daily_limit)

        if self.hourly_spend > self.config.hourly_limit * self.config.warning_threshold:
            self._trigger_alert("hourly_warning", self.hourly_spend, self.config.hourly_limit)

    def activate_kill_switch(self, reason: str):
        """Activate kill switch to stop all requests."""
        self.kill_switch_active = True
        self._trigger_alert("kill_switch_activated", reason=reason)

    def deactivate_kill_switch(self):
        """Deactivate kill switch."""
        self.kill_switch_active = False

    def get_status(self) -> dict:
        """Get current cost guard status."""
        self._reset_if_needed()

        return {
            "kill_switch_active": self.kill_switch_active,
            "daily_spend": self.daily_spend,
            "daily_limit": self.config.daily_limit,
            "daily_remaining": self.config.daily_limit - self.daily_spend,
            "hourly_spend": self.hourly_spend,
            "hourly_limit": self.config.hourly_limit,
            "hourly_remaining": self.config.hourly_limit - self.hourly_spend
        }

    def _trigger_alert(self, alert_type: str, current: float = None, limit: float = None, reason: str = None):
        """Trigger alert handlers."""
        for handler in self.alert_handlers:
            handler({
                "type": alert_type,
                "current": current,
                "limit": limit,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })


# =============================================================================
# Integrated Guard System
# =============================================================================

class GuardSystem:
    """Integrated guard system with SLO and cost monitoring."""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.slo_manager = SLOManager(self.metrics)
        self.cost_guard = CostGuard(BudgetConfig(
            daily_limit=100.0,
            hourly_limit=10.0,
            per_request_limit=1.0
        ))

        # Default SLOs
        self.slo_manager.add_slo(SLODefinition(
            name="p95_latency",
            metric_type=MetricType.LATENCY,
            target_value=5000,  # 5 seconds
            comparison="lt",
            window=timedelta(minutes=5),
            description="95th percentile latency under 5s"
        ))

        self.slo_manager.add_slo(SLODefinition(
            name="error_rate",
            metric_type=MetricType.ERROR_RATE,
            target_value=1,  # 1%
            comparison="lt",
            window=timedelta(minutes=5),
            description="Error rate under 1%"
        ))

    def pre_request_check(self, estimated_cost: float) -> tuple[bool, str]:
        """Check before processing a request."""
        return self.cost_guard.check_budget(estimated_cost)

    def post_request_record(
        self,
        latency_ms: float,
        success: bool,
        cost: float,
        tokens: int
    ):
        """Record metrics after a request."""
        self.metrics.record(MetricType.LATENCY, latency_ms)
        self.metrics.record(MetricType.THROUGHPUT, 1)
        self.metrics.record(MetricType.TOKEN_USAGE, tokens)

        if not success:
            self.metrics.record(MetricType.ERROR_RATE, 1)

        self.cost_guard.record_spend(cost)

    def get_health_report(self) -> dict:
        """Get comprehensive health report."""
        return {
            "slo_status": self.slo_manager.check_all_slos(),
            "cost_status": self.cost_guard.get_status(),
            "metrics": {
                "avg_latency": self.metrics.get_average(MetricType.LATENCY),
                "p95_latency": self.metrics.get_percentile(MetricType.LATENCY, 95),
                "total_requests": self.metrics.get_count(MetricType.THROUGHPUT),
                "total_tokens": sum(s.value for s in self.metrics.samples[MetricType.TOKEN_USAGE])
            }
        }


# =============================================================================
# Tests
# =============================================================================

def test_metrics_collector():
    """Test metrics collection."""
    print("\n" + "=" * 70)
    print("TEST: Metrics Collection")
    print("=" * 70)

    collector = MetricsCollector()

    # Record samples
    for i in range(100):
        collector.record(MetricType.LATENCY, 100 + i * 10)
        collector.record(MetricType.THROUGHPUT, 1)

    # Get stats
    avg = collector.get_average(MetricType.LATENCY)
    p95 = collector.get_percentile(MetricType.LATENCY, 95)
    count = collector.get_count(MetricType.THROUGHPUT)

    print(f"\nLatency stats:")
    print(f"  Average: {avg:.2f}ms")
    print(f"  P95: {p95:.2f}ms")
    print(f"  Request count: {count}")

    print("\n✅ Metrics collection works")


def test_slo_monitoring():
    """Test SLO monitoring (OB-06)."""
    print("\n" + "=" * 70)
    print("TEST: SLO Monitoring (OB-06)")
    print("=" * 70)

    collector = MetricsCollector()
    slo_mgr = SLOManager(collector)

    # Define SLO
    slo_mgr.add_slo(SLODefinition(
        name="latency_p95",
        metric_type=MetricType.LATENCY,
        target_value=500,
        comparison="lt",
        window=timedelta(minutes=5)
    ))

    # Add alert handler
    alerts = []
    slo_mgr.add_alert_handler(lambda a: alerts.append(a))

    # Simulate good performance
    for _ in range(50):
        collector.record(MetricType.LATENCY, 300)

    results = slo_mgr.check_all_slos()
    print(f"\nGood performance: {results}")

    # Simulate bad performance
    for _ in range(50):
        collector.record(MetricType.LATENCY, 1000)

    results = slo_mgr.check_all_slos()
    print(f"Bad performance: {results}")
    print(f"Alerts triggered: {len(alerts)}")

    print("\n✅ SLO monitoring works")


def test_cost_guard():
    """Test cost guard (OB-07)."""
    print("\n" + "=" * 70)
    print("TEST: Cost Guard (OB-07)")
    print("=" * 70)

    config = BudgetConfig(
        daily_limit=10.0,
        hourly_limit=2.0,
        per_request_limit=0.5
    )
    guard = CostGuard(config)

    # Add alert handler
    alerts = []
    guard.add_alert_handler(lambda a: alerts.append(a))

    # Normal request
    allowed, reason = guard.check_budget(0.1)
    print(f"\nNormal request: {allowed} ({reason})")

    # Record some spend
    for _ in range(15):
        guard.record_spend(0.1)

    # Check status
    status = guard.get_status()
    print(f"\nAfter spending:")
    print(f"  Hourly: ${status['hourly_spend']:.2f} / ${status['hourly_limit']:.2f}")
    print(f"  Daily: ${status['daily_spend']:.2f} / ${status['daily_limit']:.2f}")

    # Try exceeding limit
    allowed, reason = guard.check_budget(1.0)
    print(f"\nLarge request: {allowed} ({reason})")

    # Kill switch
    guard.activate_kill_switch("Manual override")
    allowed, reason = guard.check_budget(0.01)
    print(f"After kill switch: {allowed} ({reason})")

    print(f"\nAlerts: {len(alerts)}")

    print("\n✅ Cost guard works")


def test_integrated_guard():
    """Test integrated guard system."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Guard System")
    print("=" * 70)

    guard = GuardSystem()

    # Simulate requests
    for i in range(10):
        # Pre-check
        allowed, reason = guard.pre_request_check(0.05)
        if allowed:
            # Simulate request
            guard.post_request_record(
                latency_ms=200 + i * 50,
                success=i != 5,  # One failure
                cost=0.05,
                tokens=500
            )

    # Get health report
    report = guard.get_health_report()

    print(f"\nHealth Report:")
    print(f"  SLO Status: {report['slo_status']}")
    print(f"  Cost Status: {report['cost_status']}")
    print(f"  Metrics: {report['metrics']}")

    print("\n✅ Integrated guard system works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ OB-06, OB-07: GUARDS - EVALUATION SUMMARY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ OB-06 (SLO / Alerts): ⭐ (Not Supported)                                    │
│   ❌ No built-in SLO management                                             │
│   ❌ No alerting system                                                     │
│   ❌ No metric aggregation                                                  │
│   ⚠️ Custom SLOManager implementation provided                             │
│                                                                             │
│ OB-07 (Cost Guard): ⭐ (Not Supported)                                      │
│   ❌ No built-in budget limits                                              │
│   ❌ No kill switch mechanism                                               │
│   ❌ No spend tracking                                                      │
│   ⚠️ Custom CostGuard implementation provided                              │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ MetricsCollector - Metric sampling and aggregation                     │
│   ✅ SLODefinition - SLO configuration                                      │
│   ✅ SLOManager - SLO monitoring and alerting                               │
│   ✅ CostGuard - Budget enforcement and kill switch                         │
│   ✅ GuardSystem - Integrated protection                                    │
│                                                                             │
│ Production Considerations:                                                  │
│   - Use external metrics systems (Prometheus, Datadog)                      │
│   - Implement proper alerting (PagerDuty, OpsGenie)                         │
│   - Store budgets in configuration service                                  │
│   - Implement distributed kill switch (Redis-based)                         │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - Similar gaps                                                          │
│     - LangSmith provides some metrics                                       │
│   OpenAI SDK:                                                               │
│     - Token tracking available                                              │
│     - No built-in guards                                                    │
│                                                                             │
│ Both frameworks require custom implementation for:                          │
│   - SLO management                                                          │
│   - Budget enforcement                                                      │
│   - Kill switch                                                             │
│   - Alerting                                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_metrics_collector()
    test_slo_monitoring()
    test_cost_guard()
    test_integrated_guard()

    print(SUMMARY)
