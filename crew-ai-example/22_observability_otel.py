"""
22_observability_otel.py - Observability OpenTelemetry (OB-05)

Purpose: Verify OpenTelemetry compliance
- OTel GenAI Semantic Conventions
- Span/trace structure
- Vendor lock-in avoidance
- Jaeger/Zipkin export capability

LangGraph Comparison:
- LangSmith provides built-in observability for LangChain
- CrewAI requires custom OTel integration
- Both can use standard OTel SDKs
"""

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# OTel Data Structures (Simplified Implementation)
# =============================================================================

class SpanKind(str, Enum):
    """OpenTelemetry span kinds."""
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class SpanStatus(str, Enum):
    """OpenTelemetry span status."""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class SpanContext:
    """Span context for trace propagation."""
    trace_id: str
    span_id: str
    trace_flags: int = 1  # Sampled
    trace_state: str = ""


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: float
    attributes: dict = field(default_factory=dict)


@dataclass
class Span:
    """Simplified OpenTelemetry span."""

    name: str
    context: SpanContext
    parent_span_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    attributes: dict = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict] = None):
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, message: str = ""):
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self):
        """End the span."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time_unix_nano": int(self.start_time * 1e9),
            "end_time_unix_nano": int(self.end_time * 1e9) if self.end_time else None,
            "status": {"code": self.status.value, "message": self.status_message},
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
        }


# =============================================================================
# GenAI Semantic Conventions (OTel standard for LLM observability)
# =============================================================================

class GenAISemanticConventions:
    """
    OpenTelemetry Semantic Conventions for Generative AI.

    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    # System attributes
    GEN_AI_SYSTEM = "gen_ai.system"  # openai, anthropic, etc.
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"

    # Response attributes
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Usage attributes
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Operation type
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"  # chat, completion, embedding

    # Prompt/completion (event attributes)
    GEN_AI_PROMPT = "gen_ai.prompt"
    GEN_AI_COMPLETION = "gen_ai.completion"

    # Tool calling
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

    # CrewAI-specific extensions
    CREWAI_AGENT_ROLE = "crewai.agent.role"
    CREWAI_AGENT_GOAL = "crewai.agent.goal"
    CREWAI_TASK_DESCRIPTION = "crewai.task.description"
    CREWAI_CREW_NAME = "crewai.crew.name"


# =============================================================================
# Tracer Implementation
# =============================================================================

class SimpleTracer:
    """
    Simplified tracer for demonstration.

    In production, use opentelemetry-sdk.
    """

    def __init__(self, service_name: str = "crewai-service"):
        self.service_name = service_name
        self.spans: list[Span] = []
        self.current_span: Optional[Span] = None
        self.span_stack: list[Span] = []

    def _generate_id(self, length: int = 16) -> str:
        """Generate hex ID."""
        return uuid.uuid4().hex[:length]

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[dict] = None,
    ):
        """Start a new span as context manager."""
        # Determine trace context
        if self.current_span:
            trace_id = self.current_span.context.trace_id
            parent_span_id = self.current_span.context.span_id
        else:
            trace_id = self._generate_id(32)
            parent_span_id = None

        context = SpanContext(
            trace_id=trace_id,
            span_id=self._generate_id(16),
        )

        span = Span(
            name=name,
            context=context,
            parent_span_id=parent_span_id,
            kind=kind,
            attributes=attributes or {},
        )

        # Push to stack
        self.span_stack.append(self.current_span)
        self.current_span = span

        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            span.end()
            self.spans.append(span)
            self.current_span = self.span_stack.pop()

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self.current_span

    def get_all_spans(self) -> list[Span]:
        """Get all recorded spans."""
        return self.spans

    def clear(self):
        """Clear all spans."""
        self.spans = []
        self.current_span = None
        self.span_stack = []


# =============================================================================
# Exporters
# =============================================================================

class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: list[Span]):
        raise NotImplementedError


class ConsoleExporter(SpanExporter):
    """Export spans to console."""

    def export(self, spans: list[Span]):
        print("\n[Console Exporter] Spans:")
        for span in spans:
            print(f"  {span.name} ({span.context.span_id})")
            print(f"    Trace: {span.context.trace_id}")
            print(f"    Duration: {span.duration_ms:.2f}ms")
            print(f"    Status: {span.status.value}")
            if span.attributes:
                print(f"    Attributes: {span.attributes}")


class JSONFileExporter(SpanExporter):
    """Export spans to JSON file."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: list[Span]):
        data = {
            "resourceSpans": [
                {
                    "resource": {"attributes": {}},
                    "scopeSpans": [
                        {
                            "scope": {"name": "crewai-tracer"},
                            "spans": [span.to_dict() for span in spans],
                        }
                    ],
                }
            ]
        }

        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[JSON Exporter] Exported {len(spans)} spans to {self.output_path}")


class JaegerExporter(SpanExporter):
    """Export spans to Jaeger (simulated)."""

    def __init__(self, endpoint: str = "http://localhost:14268/api/traces"):
        self.endpoint = endpoint

    def export(self, spans: list[Span]):
        # In production, would send HTTP POST to Jaeger
        print(f"[Jaeger Exporter] Would send {len(spans)} spans to {self.endpoint}")
        for span in spans:
            print(f"  - {span.name} (trace: {span.context.trace_id[:8]}...)")


class OTLPExporter(SpanExporter):
    """Export spans via OTLP protocol (simulated)."""

    def __init__(self, endpoint: str = "http://localhost:4317"):
        self.endpoint = endpoint

    def export(self, spans: list[Span]):
        # In production, would use OTLP gRPC or HTTP
        print(f"[OTLP Exporter] Would send {len(spans)} spans to {self.endpoint}")


# =============================================================================
# CrewAI Instrumentation
# =============================================================================

class CrewAIInstrumentor:
    """
    Instruments CrewAI operations with OpenTelemetry spans.

    Provides automatic tracing for:
    - Crew execution
    - Agent operations
    - Task processing
    - Tool calls
    - LLM interactions
    """

    def __init__(self, tracer: SimpleTracer):
        self.tracer = tracer
        self.conv = GenAISemanticConventions

    @contextmanager
    def trace_crew(self, crew_name: str, **kwargs):
        """Trace a crew execution."""
        with self.tracer.start_span(
            f"crew.{crew_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                self.conv.CREWAI_CREW_NAME: crew_name,
                **kwargs,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_agent(self, agent_role: str, agent_goal: str = "", **kwargs):
        """Trace an agent operation."""
        with self.tracer.start_span(
            f"agent.{agent_role}",
            kind=SpanKind.INTERNAL,
            attributes={
                self.conv.CREWAI_AGENT_ROLE: agent_role,
                self.conv.CREWAI_AGENT_GOAL: agent_goal,
                **kwargs,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_task(self, task_description: str, **kwargs):
        """Trace a task execution."""
        with self.tracer.start_span(
            "task.execute",
            kind=SpanKind.INTERNAL,
            attributes={
                self.conv.CREWAI_TASK_DESCRIPTION: task_description[:100],
                **kwargs,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_tool_call(self, tool_name: str, **kwargs):
        """Trace a tool call."""
        with self.tracer.start_span(
            f"tool.{tool_name}",
            kind=SpanKind.CLIENT,
            attributes={
                self.conv.GEN_AI_TOOL_NAME: tool_name,
                **kwargs,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_llm_call(
        self,
        model: str,
        system: str = "openai",
        operation: str = "chat",
        **kwargs
    ):
        """Trace an LLM call with GenAI semantic conventions."""
        with self.tracer.start_span(
            f"gen_ai.{operation}",
            kind=SpanKind.CLIENT,
            attributes={
                self.conv.GEN_AI_SYSTEM: system,
                self.conv.GEN_AI_REQUEST_MODEL: model,
                self.conv.GEN_AI_OPERATION_NAME: operation,
                **kwargs,
            },
        ) as span:
            yield span


# =============================================================================
# Demonstrations
# =============================================================================

def demo_basic_tracing():
    """Demonstrate basic span creation and nesting."""
    print("=" * 60)
    print("Demo 1: Basic Tracing")
    print("=" * 60)

    tracer = SimpleTracer("demo-service")

    with tracer.start_span("parent-operation") as parent:
        parent.set_attribute("custom.attribute", "value")
        time.sleep(0.01)

        with tracer.start_span("child-operation") as child:
            child.add_event("processing-started")
            time.sleep(0.01)
            child.add_event("processing-completed", {"items": 10})
            child.set_status(SpanStatus.OK)

    # Export
    exporter = ConsoleExporter()
    exporter.export(tracer.get_all_spans())


def demo_genai_conventions():
    """Demonstrate GenAI semantic conventions."""
    print("\n" + "=" * 60)
    print("Demo 2: GenAI Semantic Conventions (OB-05)")
    print("=" * 60)

    tracer = SimpleTracer("llm-service")
    conv = GenAISemanticConventions

    with tracer.start_span(
        "gen_ai.chat",
        kind=SpanKind.CLIENT,
        attributes={
            conv.GEN_AI_SYSTEM: "openai",
            conv.GEN_AI_REQUEST_MODEL: "gpt-4",
            conv.GEN_AI_REQUEST_MAX_TOKENS: 1000,
            conv.GEN_AI_REQUEST_TEMPERATURE: 0.7,
            conv.GEN_AI_OPERATION_NAME: "chat",
        },
    ) as span:
        # Simulate LLM call
        time.sleep(0.02)

        # Record prompt/completion as events
        span.add_event("gen_ai.prompt", {"content": "Tell me about AI"})
        span.add_event("gen_ai.completion", {"content": "AI is..."})

        # Record usage
        span.set_attribute(conv.GEN_AI_USAGE_INPUT_TOKENS, 50)
        span.set_attribute(conv.GEN_AI_USAGE_OUTPUT_TOKENS, 150)
        span.set_attribute(conv.GEN_AI_RESPONSE_ID, "chatcmpl-123")
        span.set_attribute(conv.GEN_AI_RESPONSE_FINISH_REASONS, ["stop"])

        span.set_status(SpanStatus.OK)

    # Export
    exporter = ConsoleExporter()
    exporter.export(tracer.get_all_spans())


def demo_crewai_instrumentation():
    """Demonstrate CrewAI-specific instrumentation."""
    print("\n" + "=" * 60)
    print("Demo 3: CrewAI Instrumentation")
    print("=" * 60)

    tracer = SimpleTracer("crewai-app")
    instrumentor = CrewAIInstrumentor(tracer)

    with instrumentor.trace_crew("Research Crew", agents_count=2) as crew_span:
        crew_span.add_event("crew_started")

        with instrumentor.trace_agent(
            agent_role="Researcher",
            agent_goal="Find relevant information",
        ) as agent_span:

            with instrumentor.trace_task(
                task_description="Search for AI trends in 2024",
            ) as task_span:

                # Tool call
                with instrumentor.trace_tool_call("web_search") as tool_span:
                    time.sleep(0.01)
                    tool_span.set_attribute("search.query", "AI trends")
                    tool_span.set_status(SpanStatus.OK)

                # LLM call
                with instrumentor.trace_llm_call(
                    model="gpt-4",
                    system="openai",
                    operation="chat",
                ) as llm_span:
                    time.sleep(0.02)
                    llm_span.set_attribute("gen_ai.usage.input_tokens", 100)
                    llm_span.set_attribute("gen_ai.usage.output_tokens", 200)
                    llm_span.set_status(SpanStatus.OK)

                task_span.set_status(SpanStatus.OK)

            agent_span.set_status(SpanStatus.OK)

        crew_span.add_event("crew_completed")
        crew_span.set_status(SpanStatus.OK)

    # Export to multiple destinations
    print("\n--- Console Export ---")
    ConsoleExporter().export(tracer.get_all_spans())

    print("\n--- JSON File Export ---")
    JSONFileExporter(Path("./db/traces/demo_trace.json")).export(tracer.get_all_spans())

    print("\n--- Jaeger Export (simulated) ---")
    JaegerExporter().export(tracer.get_all_spans())


def demo_trace_hierarchy():
    """Demonstrate trace hierarchy and parent-child relationships."""
    print("\n" + "=" * 60)
    print("Demo 4: Trace Hierarchy")
    print("=" * 60)

    tracer = SimpleTracer("hierarchy-demo")

    with tracer.start_span("level-1") as l1:
        with tracer.start_span("level-2a") as l2a:
            with tracer.start_span("level-3") as l3:
                l3.set_attribute("depth", 3)
            l2a.set_attribute("depth", 2)
        with tracer.start_span("level-2b") as l2b:
            l2b.set_attribute("depth", 2)
        l1.set_attribute("depth", 1)

    # Show hierarchy
    print("\nTrace Hierarchy:")
    print("-" * 40)

    spans = tracer.get_all_spans()
    trace_id = spans[0].context.trace_id if spans else "N/A"
    print(f"Trace ID: {trace_id}")

    for span in spans:
        indent = "  " if span.parent_span_id else ""
        parent_info = f" (parent: {span.parent_span_id[:8]}...)" if span.parent_span_id else " (root)"
        print(f"{indent}{span.name} [{span.context.span_id[:8]}...]{parent_info}")


def demo_error_tracing():
    """Demonstrate error tracing."""
    print("\n" + "=" * 60)
    print("Demo 5: Error Tracing")
    print("=" * 60)

    tracer = SimpleTracer("error-demo")

    try:
        with tracer.start_span("operation-with-error") as span:
            span.add_event("starting_risky_operation")
            time.sleep(0.01)
            raise ValueError("Something went wrong!")
    except ValueError:
        pass  # Expected

    # Show error span
    spans = tracer.get_all_spans()
    for span in spans:
        print(f"Span: {span.name}")
        print(f"  Status: {span.status.value}")
        print(f"  Message: {span.status_message}")


def demo_exporters():
    """Demonstrate different exporters."""
    print("\n" + "=" * 60)
    print("Demo 6: Multiple Exporters")
    print("=" * 60)

    tracer = SimpleTracer("exporter-demo")

    with tracer.start_span("exportable-operation") as span:
        span.set_attribute("export.demo", True)
        time.sleep(0.01)
        span.set_status(SpanStatus.OK)

    spans = tracer.get_all_spans()

    # Different exporters
    exporters = [
        ("Console", ConsoleExporter()),
        ("JSON File", JSONFileExporter(Path("./db/traces/export_demo.json"))),
        ("Jaeger (simulated)", JaegerExporter("http://jaeger:14268")),
        ("OTLP (simulated)", OTLPExporter("http://otel-collector:4317")),
    ]

    for name, exporter in exporters:
        print(f"\n--- {name} ---")
        exporter.export(spans)


def main():
    print("=" * 60)
    print("Observability OpenTelemetry Verification (OB-05)")
    print("=" * 60)
    print("""
This script verifies OpenTelemetry compliance for observability.

Verification Items:
- OTel GenAI Semantic Conventions
- Span/trace structure
- Parent-child relationships
- Event recording
- Error status propagation
- Multiple export formats

GenAI Semantic Conventions:
- gen_ai.system: LLM provider (openai, anthropic, etc.)
- gen_ai.request.model: Model name
- gen_ai.usage.input_tokens: Token usage
- gen_ai.operation.name: Operation type

Supported Exporters:
- Console (development)
- JSON File (debugging)
- Jaeger (distributed tracing)
- OTLP (OpenTelemetry Collector)

LangGraph Comparison:
- LangSmith provides integrated observability
- CrewAI requires custom OTel integration
- OTel provides vendor-neutral standard

Production Setup:
  pip install opentelemetry-sdk opentelemetry-exporter-otlp
  Configure OTEL_EXPORTER_OTLP_ENDPOINT environment variable
""")

    # Ensure output directory exists
    Path("./db/traces").mkdir(parents=True, exist_ok=True)

    # Run all demos
    demo_basic_tracing()
    demo_genai_conventions()
    demo_crewai_instrumentation()
    demo_trace_hierarchy()
    demo_error_tracing()
    demo_exporters()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
