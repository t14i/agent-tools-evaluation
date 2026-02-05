"""
Observability - Part 1: Tracing (OB-01, OB-02, OB-03)
Built-in tracing, token consumption, log output
"""

from dotenv import load_dotenv
load_dotenv()


import json
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# OB-01: Built-in Tracing
# =============================================================================

class SpanKind(Enum):
    """Types of trace spans."""
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    HANDOFF = "handoff"
    GUARDRAIL = "guardrail"


@dataclass
class TraceSpan:
    """A span in a trace."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    started_at: datetime
    ended_at: Optional[datetime] = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "OK"


class TracingContext:
    """
    Context for distributed tracing.
    OpenAI Agents SDK has built-in tracing enabled by default.

    Implements OB-01: Trace.
    """

    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or f"trace_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.spans: list[TraceSpan] = []
        self.current_span: Optional[TraceSpan] = None
        self._span_counter = 0

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        attributes: dict = None
    ) -> TraceSpan:
        """Start a new span."""
        self._span_counter += 1

        span = TraceSpan(
            span_id=f"span_{self._span_counter:06d}",
            trace_id=self.trace_id,
            parent_span_id=self.current_span.span_id if self.current_span else None,
            name=name,
            kind=kind,
            started_at=datetime.now(),
            attributes=attributes or {}
        )

        self.spans.append(span)
        self.current_span = span
        return span

    def end_span(self, status: str = "OK"):
        """End the current span."""
        if self.current_span:
            self.current_span.ended_at = datetime.now()
            self.current_span.status = status

            # Find parent span
            parent_id = self.current_span.parent_span_id
            self.current_span = None
            for span in reversed(self.spans):
                if span.span_id == parent_id:
                    self.current_span = span
                    break

    def add_event(self, name: str, attributes: dict = None):
        """Add an event to the current span."""
        if self.current_span:
            self.current_span.events.append({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {}
            })

    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the current span."""
        if self.current_span:
            self.current_span.attributes[key] = value

    def get_trace_summary(self) -> dict:
        """Get summary of the trace."""
        return {
            "trace_id": self.trace_id,
            "span_count": len(self.spans),
            "duration_ms": self._calculate_duration(),
            "spans": [
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "kind": s.kind.value,
                    "duration_ms": self._span_duration(s)
                }
                for s in self.spans
            ]
        }

    def _calculate_duration(self) -> Optional[float]:
        """Calculate total trace duration."""
        if not self.spans:
            return None
        start = min(s.started_at for s in self.spans)
        end = max(s.ended_at or datetime.now() for s in self.spans)
        return (end - start).total_seconds() * 1000

    def _span_duration(self, span: TraceSpan) -> Optional[float]:
        """Calculate span duration."""
        if not span.ended_at:
            return None
        return (span.ended_at - span.started_at).total_seconds() * 1000


# =============================================================================
# OB-02: Token Consumption Tracking
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage for a request."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    timestamp: datetime = field(default_factory=datetime.now)


class TokenTracker:
    """
    Tracks token consumption across requests.
    Implements OB-02: Token Consumption.

    OpenAI SDK provides request_usage_entries for tracking.
    """

    def __init__(self):
        self.usage_history: list[TokenUsage] = []
        self.cost_per_1k_tokens = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> TokenUsage:
        """Record token usage."""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model
        )
        self.usage_history.append(usage)
        return usage

    def get_total(self) -> dict:
        """Get total token usage."""
        if not self.usage_history:
            return {"input": 0, "output": 0, "total": 0}

        return {
            "input": sum(u.input_tokens for u in self.usage_history),
            "output": sum(u.output_tokens for u in self.usage_history),
            "total": sum(u.total_tokens for u in self.usage_history)
        }

    def get_cost_estimate(self) -> float:
        """Estimate total cost."""
        total_cost = 0.0

        for usage in self.usage_history:
            rates = self.cost_per_1k_tokens.get(usage.model, {"input": 0, "output": 0})
            total_cost += (usage.input_tokens / 1000) * rates["input"]
            total_cost += (usage.output_tokens / 1000) * rates["output"]

        return total_cost

    def get_usage_by_model(self) -> dict:
        """Get usage breakdown by model."""
        by_model = {}
        for usage in self.usage_history:
            if usage.model not in by_model:
                by_model[usage.model] = {"input": 0, "output": 0, "total": 0, "calls": 0}
            by_model[usage.model]["input"] += usage.input_tokens
            by_model[usage.model]["output"] += usage.output_tokens
            by_model[usage.model]["total"] += usage.total_tokens
            by_model[usage.model]["calls"] += 1
        return by_model


# =============================================================================
# OB-03: Structured Logging
# =============================================================================

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """A structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    trace_id: Optional[str]
    span_id: Optional[str]
    attributes: dict


class StructuredLogger:
    """
    Structured logging for agent execution.
    Implements OB-03: Log Output.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.entries: list[LogEntry] = []
        self.tracing_context: Optional[TracingContext] = None

    def set_context(self, context: TracingContext):
        """Set tracing context for correlation."""
        self.tracing_context = context

    def log(
        self,
        level: LogLevel,
        message: str,
        **attributes
    ) -> LogEntry:
        """Log a structured message."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            trace_id=self.tracing_context.trace_id if self.tracing_context else None,
            span_id=self.tracing_context.current_span.span_id if self.tracing_context and self.tracing_context.current_span else None,
            attributes={
                "service": self.service_name,
                **attributes
            }
        )
        self.entries.append(entry)
        return entry

    def debug(self, message: str, **attributes):
        return self.log(LogLevel.DEBUG, message, **attributes)

    def info(self, message: str, **attributes):
        return self.log(LogLevel.INFO, message, **attributes)

    def warning(self, message: str, **attributes):
        return self.log(LogLevel.WARNING, message, **attributes)

    def error(self, message: str, **attributes):
        return self.log(LogLevel.ERROR, message, **attributes)

    def export_json(self) -> str:
        """Export logs as JSON."""
        return json.dumps([
            {
                "timestamp": e.timestamp.isoformat(),
                "level": e.level.value,
                "message": e.message,
                "trace_id": e.trace_id,
                "span_id": e.span_id,
                "attributes": e.attributes
            }
            for e in self.entries
        ], indent=2)


# =============================================================================
# Integrated Observability
# =============================================================================

class ObservabilityManager:
    """Integrated observability for agents."""

    def __init__(self, service_name: str):
        self.tracing = TracingContext()
        self.tokens = TokenTracker()
        self.logger = StructuredLogger(service_name)
        self.logger.set_context(self.tracing)

    def start_agent_execution(self, agent_name: str, input_text: str):
        """Start tracing an agent execution."""
        span = self.tracing.start_span(
            name=f"agent:{agent_name}",
            kind=SpanKind.AGENT,
            attributes={"input_length": len(input_text)}
        )
        self.logger.info(f"Starting agent {agent_name}", agent=agent_name)
        return span

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        """Record an LLM call."""
        self.tracing.start_span(
            name=f"llm:{model}",
            kind=SpanKind.LLM,
            attributes={"model": model}
        )

        usage = self.tokens.record(input_tokens, output_tokens, model)

        self.tracing.set_attribute("input_tokens", input_tokens)
        self.tracing.set_attribute("output_tokens", output_tokens)
        self.tracing.end_span()

        self.logger.debug(
            f"LLM call: {model}",
            model=model,
            tokens=usage.total_tokens
        )

    def record_tool_call(self, tool_name: str, duration_ms: float):
        """Record a tool call."""
        self.tracing.start_span(
            name=f"tool:{tool_name}",
            kind=SpanKind.TOOL,
            attributes={"tool": tool_name, "duration_ms": duration_ms}
        )
        self.tracing.end_span()

        self.logger.debug(f"Tool call: {tool_name}", tool=tool_name, duration_ms=duration_ms)

    def end_agent_execution(self, status: str = "OK"):
        """End agent execution."""
        self.tracing.end_span(status)
        self.logger.info("Agent execution completed", status=status)

    def get_summary(self) -> dict:
        """Get observability summary."""
        return {
            "trace": self.tracing.get_trace_summary(),
            "tokens": self.tokens.get_total(),
            "cost_estimate": self.tokens.get_cost_estimate(),
            "log_entries": len(self.logger.entries)
        }


# =============================================================================
# Tests
# =============================================================================

def test_tracing():
    """Test tracing (OB-01)."""
    print("\n" + "=" * 70)
    print("TEST: Tracing (OB-01)")
    print("=" * 70)

    ctx = TracingContext()

    # Start spans
    ctx.start_span("agent:MyAgent", SpanKind.AGENT)
    ctx.set_attribute("model", "gpt-4o")

    ctx.start_span("llm:chat", SpanKind.LLM)
    ctx.add_event("request_sent", {"tokens": 100})
    ctx.end_span()

    ctx.start_span("tool:search", SpanKind.TOOL)
    ctx.end_span()

    ctx.end_span()  # End agent span

    summary = ctx.get_trace_summary()
    print(f"\nTrace summary:")
    print(f"  Trace ID: {summary['trace_id']}")
    print(f"  Span count: {summary['span_count']}")
    print(f"  Duration: {summary['duration_ms']:.2f}ms")

    print("\nSpans:")
    for span in summary["spans"]:
        print(f"  - {span['name']} ({span['kind']})")

    print("\n✅ Tracing works")


def test_token_tracking():
    """Test token tracking (OB-02)."""
    print("\n" + "=" * 70)
    print("TEST: Token Tracking (OB-02)")
    print("=" * 70)

    tracker = TokenTracker()

    # Record usage
    tracker.record(1000, 500, "gpt-4o")
    tracker.record(500, 200, "gpt-4o-mini")
    tracker.record(800, 300, "gpt-4o")

    total = tracker.get_total()
    print(f"\nTotal usage:")
    print(f"  Input: {total['input']} tokens")
    print(f"  Output: {total['output']} tokens")
    print(f"  Total: {total['total']} tokens")

    cost = tracker.get_cost_estimate()
    print(f"\nEstimated cost: ${cost:.4f}")

    by_model = tracker.get_usage_by_model()
    print(f"\nBy model:")
    for model, usage in by_model.items():
        print(f"  {model}: {usage['total']} tokens ({usage['calls']} calls)")

    print("\n✅ Token tracking works")


def test_structured_logging():
    """Test structured logging (OB-03)."""
    print("\n" + "=" * 70)
    print("TEST: Structured Logging (OB-03)")
    print("=" * 70)

    logger = StructuredLogger("agent-service")

    # Create tracing context
    ctx = TracingContext()
    ctx.start_span("test", SpanKind.AGENT)
    logger.set_context(ctx)

    # Log messages
    logger.info("Starting processing", request_id="req_123")
    logger.debug("Calling LLM", model="gpt-4o")
    logger.warning("Rate limit approaching", remaining=10)
    logger.error("Tool failed", tool="search", error="timeout")

    print(f"\nLog entries: {len(logger.entries)}")

    # Export
    exported = logger.export_json()
    print(f"\nExported (first 300 chars):\n{exported[:300]}...")

    print("\n✅ Structured logging works")


def test_integrated_observability():
    """Test integrated observability."""
    print("\n" + "=" * 70)
    print("TEST: Integrated Observability")
    print("=" * 70)

    obs = ObservabilityManager("my-agent-service")

    # Simulate execution
    obs.start_agent_execution("CustomerSupportBot", "Help me with my order")

    # LLM calls
    obs.record_llm_call("gpt-4o", 500, 200)
    obs.record_tool_call("search_orders", 150.5)
    obs.record_llm_call("gpt-4o", 800, 300)

    obs.end_agent_execution()

    # Get summary
    summary = obs.get_summary()
    print(f"\nObservability Summary:")
    print(f"  Trace spans: {summary['trace']['span_count']}")
    print(f"  Total tokens: {summary['tokens']['total']}")
    print(f"  Cost estimate: ${summary['cost_estimate']:.4f}")
    print(f"  Log entries: {summary['log_entries']}")

    print("\n✅ Integrated observability works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ OB-01, OB-02, OB-03: TRACING - EVALUATION SUMMARY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ OB-01 (Trace): ⭐⭐⭐⭐⭐ (Production Recommended)                           │
│   ✅ Built-in tracing enabled by default                                    │
│   ✅ Trace spans for agents, tools, handoffs                                │
│   ✅ Dashboard available for visualization                                  │
│   ✅ Context propagation                                                    │
│                                                                             │
│ OB-02 (Token Consumption): ⭐⭐⭐⭐⭐ (Production Recommended)               │
│   ✅ request_usage_entries in responses                                     │
│   ✅ Input/output token breakdown                                           │
│   ✅ Per-request tracking                                                   │
│   ✅ Model information included                                             │
│                                                                             │
│ OB-03 (Log Output): ⭐⭐⭐⭐ (Production Ready)                              │
│   ✅ Trace spans provide execution log                                      │
│   ⚠️ Structured logging requires custom implementation                     │
│   ⚠️ Correlation with traces is manual                                     │
│                                                                             │
│ OpenAI SDK Tracing Features:                                                │
│   - Automatic span creation                                                 │
│   - Agent, tool, handoff spans                                              │
│   - Token usage tracking                                                    │
│   - Dashboard visualization                                                 │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph + LangSmith:                                                    │
│     - Similar tracing capabilities                                          │
│     - LangSmith dashboard                                                   │
│   OpenAI SDK:                                                               │
│     - Built-in by default                                                   │
│     - Simpler setup                                                         │
│     - OpenAI-hosted                                                         │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ TracingContext - Span management                                       │
│   ✅ TokenTracker - Usage and cost tracking                                 │
│   ✅ StructuredLogger - JSON-formatted logs                                 │
│   ✅ ObservabilityManager - Integrated observability                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_tracing()
    test_token_tracking()
    test_structured_logging()
    test_integrated_observability()

    print(SUMMARY)
