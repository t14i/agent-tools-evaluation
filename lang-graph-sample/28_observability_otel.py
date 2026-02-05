"""
LangGraph Observability - OpenTelemetry Integration (OB-05)
OTel GenAI Semantic Conventions compliance.

Evaluation: OB-05 (OTel Compliance)
"""

import time
import json
from typing import Annotated, TypedDict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =============================================================================
# OPENTELEMETRY GENAI SEMANTIC CONVENTIONS
# =============================================================================
# Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/

# Span Names (GenAI Semantic Conventions)
GENAI_SPAN_NAMES = {
    "chat": "gen_ai.chat",
    "completion": "gen_ai.completion",
    "embedding": "gen_ai.embedding",
    "tool_call": "gen_ai.tool_call",
}

# Attribute Names (GenAI Semantic Conventions)
GENAI_ATTRIBUTES = {
    # System attributes
    "gen_ai.system": "gen_ai.system",  # e.g., "openai", "anthropic"
    "gen_ai.request.model": "gen_ai.request.model",
    "gen_ai.request.max_tokens": "gen_ai.request.max_tokens",
    "gen_ai.request.temperature": "gen_ai.request.temperature",
    "gen_ai.request.top_p": "gen_ai.request.top_p",

    # Response attributes
    "gen_ai.response.model": "gen_ai.response.model",
    "gen_ai.response.finish_reason": "gen_ai.response.finish_reason",

    # Usage attributes
    "gen_ai.usage.input_tokens": "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens": "gen_ai.usage.output_tokens",
    "gen_ai.usage.total_tokens": "gen_ai.usage.total_tokens",

    # Tool attributes
    "gen_ai.tool.name": "gen_ai.tool.name",
    "gen_ai.tool.call_id": "gen_ai.tool.call_id",
}


# =============================================================================
# SIMULATED OTEL TRACER (for demonstration without actual OTel dependency)
# =============================================================================

@dataclass
class Span:
    """Simulated OTel span."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "UNSET"  # UNSET, OK, ERROR

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now(),
            "attributes": attributes or {}
        })

    def set_status(self, status: str, description: str = None):
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def end(self):
        """End the span."""
        self.end_time = datetime.now()


class SimulatedTracer:
    """Simulated OTel tracer for demonstration."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: list[Span] = []
        self.current_span: Optional[Span] = None
        self._trace_counter = 0
        self._span_counter = 0

    def _generate_trace_id(self) -> str:
        self._trace_counter += 1
        return f"trace_{self._trace_counter:08x}"

    def _generate_span_id(self) -> str:
        self._span_counter += 1
        return f"span_{self._span_counter:08x}"

    @contextmanager
    def start_span(self, name: str, attributes: dict = None):
        """Start a new span."""
        parent_span_id = self.current_span.span_id if self.current_span else None
        trace_id = self.current_span.trace_id if self.current_span else self._generate_trace_id()

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id,
            start_time=datetime.now(),
            attributes=attributes or {}
        )

        self.spans.append(span)
        previous_span = self.current_span
        self.current_span = span

        try:
            yield span
            span.set_status("OK")
        except Exception as e:
            span.set_status("ERROR", str(e))
            raise
        finally:
            span.end()
            self.current_span = previous_span

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self.current_span


# =============================================================================
# LANGGRAPH OTEL INSTRUMENTOR
# =============================================================================

class LangGraphInstrumentor:
    """
    Instruments LangGraph with OpenTelemetry.
    Implements OB-05: OTel Compliance with GenAI Semantic Conventions.
    """

    def __init__(self, tracer: SimulatedTracer):
        self.tracer = tracer

    def trace_llm_call(
        self,
        model: str,
        messages: list,
        response: Any,
        usage: dict = None
    ):
        """Trace an LLM call following GenAI conventions."""
        with self.tracer.start_span(GENAI_SPAN_NAMES["chat"]) as span:
            # System attributes
            span.set_attribute(GENAI_ATTRIBUTES["gen_ai.system"], "anthropic")
            span.set_attribute(GENAI_ATTRIBUTES["gen_ai.request.model"], model)

            # Request attributes
            span.set_attribute("gen_ai.request.messages_count", len(messages))

            # Response attributes
            if hasattr(response, "content"):
                span.set_attribute("gen_ai.response.content_length", len(str(response.content)))

            if hasattr(response, "tool_calls") and response.tool_calls:
                span.set_attribute("gen_ai.response.tool_calls_count", len(response.tool_calls))

            # Usage attributes
            if usage:
                span.set_attribute(GENAI_ATTRIBUTES["gen_ai.usage.input_tokens"], usage.get("input_tokens", 0))
                span.set_attribute(GENAI_ATTRIBUTES["gen_ai.usage.output_tokens"], usage.get("output_tokens", 0))
                span.set_attribute(GENAI_ATTRIBUTES["gen_ai.usage.total_tokens"], usage.get("total_tokens", 0))

            # Events for message details
            span.add_event("gen_ai.messages", {
                "input_messages": len(messages),
                "model": model
            })

    def trace_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: Any,
        call_id: str = None
    ):
        """Trace a tool call following GenAI conventions."""
        with self.tracer.start_span(GENAI_SPAN_NAMES["tool_call"]) as span:
            span.set_attribute(GENAI_ATTRIBUTES["gen_ai.tool.name"], tool_name)
            if call_id:
                span.set_attribute(GENAI_ATTRIBUTES["gen_ai.tool.call_id"], call_id)

            span.set_attribute("gen_ai.tool.args", json.dumps(tool_args, default=str))
            span.set_attribute("gen_ai.tool.result_length", len(str(tool_result)))

            span.add_event("tool_execution", {
                "tool_name": tool_name,
                "args_keys": list(tool_args.keys())
            })

    def trace_graph_execution(self, graph_name: str, thread_id: str):
        """Trace a complete graph execution."""
        return self.tracer.start_span(f"langgraph.{graph_name}", {
            "langgraph.graph_name": graph_name,
            "langgraph.thread_id": thread_id
        })

    def trace_node_execution(self, node_name: str):
        """Trace a node execution."""
        return self.tracer.start_span(f"langgraph.node.{node_name}", {
            "langgraph.node_name": node_name
        })


# =============================================================================
# INSTRUMENTED COMPONENTS
# =============================================================================

tracer = SimulatedTracer("langgraph-service")
instrumentor = LangGraphInstrumentor(tracer)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    trace_info: dict


@tool
def search_database(query: str) -> str:
    """Search the database."""
    time.sleep(0.1)  # Simulate latency
    return f"Found 10 results for '{query}'"


@tool
def send_notification(message: str) -> str:
    """Send a notification."""
    time.sleep(0.05)
    return f"Notification sent: {message}"


tools = [search_database, send_notification]

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def instrumented_agent(state: State) -> State:
    """Agent node with OTel instrumentation."""
    with instrumentor.trace_node_execution("agent"):
        messages = state["messages"]

        # Trace LLM call
        start_time = time.time()
        response = llm_with_tools.invoke(messages)
        latency = time.time() - start_time

        # Simulated usage (in production, get from response metadata)
        usage = {
            "input_tokens": len(str(messages)) // 4,  # Rough estimate
            "output_tokens": len(str(response.content)) // 4,
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        instrumentor.trace_llm_call(
            model="claude-sonnet-4-20250514",
            messages=messages,
            response=response,
            usage=usage
        )

        # Update trace info
        trace_info = state.get("trace_info", {})
        trace_info["llm_calls"] = trace_info.get("llm_calls", 0) + 1
        trace_info["total_tokens"] = trace_info.get("total_tokens", 0) + usage["total_tokens"]
        trace_info["latency_ms"] = trace_info.get("latency_ms", 0) + int(latency * 1000)

        return {"messages": [response], "trace_info": trace_info}


class InstrumentedToolNode:
    """Tool node with OTel instrumentation."""

    def __init__(self, tools: list, instrumentor: LangGraphInstrumentor):
        self.tool_node = ToolNode(tools)
        self.tools_by_name = {t.name: t for t in tools}
        self.instrumentor = instrumentor

    def __call__(self, state: dict) -> dict:
        """Execute tools with instrumentation."""
        last_message = state["messages"][-1]
        results = []
        trace_info = state.get("trace_info", {})

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            with self.instrumentor.trace_node_execution(f"tool:{tool_name}"):
                start_time = time.time()
                tool_fn = self.tools_by_name[tool_name]
                result = tool_fn.invoke(tool_args)
                latency = time.time() - start_time

                self.instrumentor.trace_tool_call(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=result,
                    call_id=tool_id
                )

                results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

                # Update trace info
                trace_info["tool_calls"] = trace_info.get("tool_calls", 0) + 1
                trace_info["latency_ms"] = trace_info.get("latency_ms", 0) + int(latency * 1000)

        return {"messages": results, "trace_info": trace_info}


def should_continue(state: State) -> str:
    """Conditional edge."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_instrumented_graph():
    """Build graph with OTel instrumentation."""
    builder = StateGraph(State)

    builder.add_node("agent", instrumented_agent)
    builder.add_node("tools", InstrumentedToolNode(tools, instrumentor))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    return builder.compile()


# =============================================================================
# TESTS
# =============================================================================

def test_otel_conventions():
    """Test OTel GenAI Semantic Conventions compliance."""
    print("\n" + "=" * 70)
    print("TEST: OTel GenAI Semantic Conventions (OB-05)")
    print("=" * 70)

    print("\n--- Standard Attribute Names ---")
    for key, value in GENAI_ATTRIBUTES.items():
        print(f"  {key}: {value}")

    print("\n--- Standard Span Names ---")
    for key, value in GENAI_SPAN_NAMES.items():
        print(f"  {key}: {value}")


def test_span_generation():
    """Test span generation with instrumentation."""
    print("\n" + "=" * 70)
    print("TEST: Span Generation")
    print("=" * 70)

    # Clear previous spans
    tracer.spans.clear()

    # Create some spans
    with tracer.start_span("test_operation", {"test.attribute": "value"}) as span:
        span.add_event("test_event", {"detail": "something"})

        with tracer.start_span("nested_operation") as nested:
            nested.set_attribute("nested.value", 42)

    print(f"\nGenerated {len(tracer.spans)} spans:")
    for span in tracer.spans:
        print(f"\n  Span: {span.name}")
        print(f"    Trace ID: {span.trace_id}")
        print(f"    Span ID: {span.span_id}")
        print(f"    Parent: {span.parent_span_id}")
        print(f"    Attributes: {span.attributes}")
        print(f"    Events: {len(span.events)}")
        print(f"    Status: {span.status}")


def test_instrumented_execution():
    """Test instrumented graph execution."""
    print("\n" + "=" * 70)
    print("TEST: Instrumented Execution")
    print("=" * 70)

    # Clear previous spans
    tracer.spans.clear()

    graph = build_instrumented_graph()

    with instrumentor.trace_graph_execution("test_graph", "thread-123"):
        result = graph.invoke({
            "messages": [("user", "Search the database for users")],
            "trace_info": {}
        })

    print(f"\n--- Execution Result ---")
    print(f"Response: {result['messages'][-1].content[:100]}...")
    print(f"\nTrace Info:")
    for key, value in result.get("trace_info", {}).items():
        print(f"  {key}: {value}")

    print(f"\n--- Generated Spans ({len(tracer.spans)}) ---")
    for span in tracer.spans:
        indent = "    " if span.parent_span_id else "  "
        print(f"{indent}[{span.status}] {span.name}")
        if span.attributes:
            for key, value in list(span.attributes.items())[:3]:
                print(f"{indent}  {key}: {value}")


def test_trace_hierarchy():
    """Test trace hierarchy."""
    print("\n" + "=" * 70)
    print("TEST: Trace Hierarchy")
    print("=" * 70)

    # Clear previous spans
    tracer.spans.clear()

    # Build trace hierarchy
    with tracer.start_span("graph_execution") as root:
        with tracer.start_span("agent_node") as agent:
            with tracer.start_span("llm_call") as llm:
                llm.set_attribute("model", "claude-sonnet-4-20250514")

            with tracer.start_span("tool_call") as t1:
                t1.set_attribute("tool", "search")

            with tracer.start_span("tool_call") as t2:
                t2.set_attribute("tool", "notify")

    # Print hierarchy
    print("\nTrace hierarchy:")

    def print_hierarchy(span_id: Optional[str], indent: int = 0):
        for span in tracer.spans:
            if span.parent_span_id == span_id:
                print("  " * indent + f"└─ {span.name} ({span.span_id})")
                print_hierarchy(span.span_id, indent + 1)

    # Find root spans (no parent)
    for span in tracer.spans:
        if span.parent_span_id is None:
            print(f"  {span.name} ({span.span_id})")
            print_hierarchy(span.span_id, 1)


def test_metrics_extraction():
    """Test extracting metrics from traces."""
    print("\n" + "=" * 70)
    print("TEST: Metrics Extraction from Traces")
    print("=" * 70)

    # Analyze spans
    llm_spans = [s for s in tracer.spans if s.name == GENAI_SPAN_NAMES["chat"]]
    tool_spans = [s for s in tracer.spans if s.name == GENAI_SPAN_NAMES["tool_call"]]

    print(f"\nLLM calls: {len(llm_spans)}")
    print(f"Tool calls: {len(tool_spans)}")

    # Calculate total tokens
    total_tokens = sum(
        s.attributes.get(GENAI_ATTRIBUTES["gen_ai.usage.total_tokens"], 0)
        for s in llm_spans
    )
    print(f"Total tokens: {total_tokens}")

    # Calculate durations
    for span in tracer.spans[:5]:
        if span.end_time:
            duration = (span.end_time - span.start_time).total_seconds() * 1000
            print(f"  {span.name}: {duration:.2f}ms")


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ OB-05: OPENTELEMETRY COMPLIANCE - EVALUATION SUMMARY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LangGraph Native Support: ⭐ (Not Supported)                                │
│                                                                             │
│ LangGraph does NOT provide:                                                 │
│   ❌ Built-in OpenTelemetry support                                         │
│   ❌ GenAI Semantic Conventions compliance                                  │
│   ❌ Automatic span generation                                              │
│   ❌ Trace context propagation                                              │
│                                                                             │
│ Note: LangSmith provides observability but is vendor-specific,              │
│ not OTel compliant.                                                         │
│                                                                             │
│ Custom Implementation Required:                                             │
│   ✅ Tracer wrapper - Span generation and management                        │
│   ✅ LangGraphInstrumentor - Instrument graph components                    │
│   ✅ GenAI attribute mapping - Semantic conventions                         │
│   ✅ Node/tool tracing - Execution instrumentation                          │
│                                                                             │
│ GenAI Semantic Conventions Supported:                                       │
│   ✓ gen_ai.system - AI provider identification                              │
│   ✓ gen_ai.request.model - Model name                                       │
│   ✓ gen_ai.usage.* - Token usage metrics                                    │
│   ✓ gen_ai.tool.* - Tool call attributes                                    │
│   ✓ Standard span names (gen_ai.chat, gen_ai.tool_call)                     │
│                                                                             │
│ Production Implementation:                                                  │
│   - Use opentelemetry-sdk for real tracing                                  │
│   - Export to Jaeger, Zipkin, or cloud providers                            │
│   - Integrate with langchain callbacks                                      │
│   - Consider opentelemetry-instrumentation-langchain                        │
│                                                                             │
│ Benefits of OTel Compliance:                                                │
│   - Vendor-agnostic observability                                           │
│   - Standardized metrics across AI systems                                  │
│   - Integration with existing APM tools                                     │
│   - Cross-service tracing                                                   │
│                                                                             │
│ Rating: ⭐ (Not Supported - fully custom)                                   │
│   - No native OTel support                                                  │
│   - LangSmith is vendor-locked                                              │
│   - Community instrumentors exist but are third-party                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_otel_conventions()
    test_span_generation()
    test_trace_hierarchy()
    test_instrumented_execution()
    test_metrics_extraction()

    print(SUMMARY)
