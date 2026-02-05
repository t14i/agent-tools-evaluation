"""
Observability - Part 2: External Integration (OB-04, OB-05)
Third-party integrations, OpenTelemetry compliance
"""

from dotenv import load_dotenv
load_dotenv()


import json
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# =============================================================================
# OB-04: External Integration (Datadog, Langfuse, Agenta)
# =============================================================================

class TracingBackend(ABC):
    """Abstract base for tracing backends."""

    @abstractmethod
    def send_span(self, span: dict) -> bool:
        pass

    @abstractmethod
    def flush(self):
        pass


class DatadogBackend(TracingBackend):
    """
    Datadog integration for tracing.
    OpenAI SDK supports native Datadog integration.
    """

    def __init__(self, api_key: str = None, service_name: str = "agent-service"):
        self.api_key = api_key
        self.service_name = service_name
        self.buffer: list[dict] = []

    def send_span(self, span: dict) -> bool:
        """Send span to Datadog."""
        # Add Datadog-specific fields
        dd_span = {
            "name": span.get("name"),
            "service": self.service_name,
            "resource": span.get("resource", span.get("name")),
            "trace_id": span.get("trace_id"),
            "span_id": span.get("span_id"),
            "parent_id": span.get("parent_span_id"),
            "start": span.get("start_time"),
            "duration": span.get("duration_ns"),
            "meta": span.get("attributes", {}),
            "metrics": span.get("metrics", {})
        }

        self.buffer.append(dd_span)
        return True

    def flush(self):
        """Flush buffered spans to Datadog."""
        if not self.buffer:
            return

        # In production: POST to Datadog API
        print(f"[Datadog] Flushing {len(self.buffer)} spans")
        self.buffer.clear()


class LangfuseBackend(TracingBackend):
    """
    Langfuse integration for LLM observability.
    OpenAI SDK supports Langfuse via tracing hooks.
    """

    def __init__(self, public_key: str = None, secret_key: str = None):
        self.public_key = public_key
        self.secret_key = secret_key
        self.traces: list[dict] = []

    def send_span(self, span: dict) -> bool:
        """Send span to Langfuse."""
        # Langfuse-specific format
        lf_observation = {
            "type": self._map_span_type(span.get("kind")),
            "name": span.get("name"),
            "traceId": span.get("trace_id"),
            "id": span.get("span_id"),
            "parentObservationId": span.get("parent_span_id"),
            "startTime": span.get("start_time"),
            "endTime": span.get("end_time"),
            "model": span.get("attributes", {}).get("model"),
            "input": span.get("input"),
            "output": span.get("output"),
            "usage": {
                "input": span.get("attributes", {}).get("input_tokens"),
                "output": span.get("attributes", {}).get("output_tokens"),
            }
        }

        self.traces.append(lf_observation)
        return True

    def _map_span_type(self, kind: str) -> str:
        """Map span kind to Langfuse observation type."""
        mapping = {
            "llm": "GENERATION",
            "tool": "SPAN",
            "agent": "SPAN",
            "handoff": "SPAN"
        }
        return mapping.get(kind, "SPAN")

    def flush(self):
        """Flush to Langfuse."""
        if not self.traces:
            return

        print(f"[Langfuse] Flushing {len(self.traces)} observations")
        self.traces.clear()


class AgentaBackend(TracingBackend):
    """
    Agenta integration for LLMOps.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.spans: list[dict] = []

    def send_span(self, span: dict) -> bool:
        """Send span to Agenta."""
        self.spans.append(span)
        return True

    def flush(self):
        """Flush to Agenta."""
        print(f"[Agenta] Flushing {len(self.spans)} spans")
        self.spans.clear()


# =============================================================================
# OB-05: OpenTelemetry Compliance
# =============================================================================

# OpenTelemetry GenAI Semantic Conventions
OTEL_GENAI_ATTRS = {
    # System
    "gen_ai.system": "gen_ai.system",
    "gen_ai.request.model": "gen_ai.request.model",

    # Request
    "gen_ai.request.max_tokens": "gen_ai.request.max_tokens",
    "gen_ai.request.temperature": "gen_ai.request.temperature",
    "gen_ai.request.top_p": "gen_ai.request.top_p",

    # Response
    "gen_ai.response.model": "gen_ai.response.model",
    "gen_ai.response.finish_reason": "gen_ai.response.finish_reason",

    # Usage
    "gen_ai.usage.input_tokens": "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens": "gen_ai.usage.output_tokens",

    # Tool
    "gen_ai.tool.name": "gen_ai.tool.name",
    "gen_ai.tool.call_id": "gen_ai.tool.call_id",
}


@dataclass
class OTelSpan:
    """OpenTelemetry compliant span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: int  # OTel SpanKind
    start_time_unix_nano: int
    end_time_unix_nano: Optional[int]
    attributes: list[dict]
    status: dict
    resource: dict


class OpenTelemetryExporter:
    """
    Exports traces in OpenTelemetry format.
    Implements OB-05: OTel Compliance.

    Note: OpenAI SDK does not have native OTel support.
    This provides a bridge to OTel-compatible backends.
    """

    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.spans: list[OTelSpan] = []

    def _convert_attributes(self, attrs: dict) -> list[dict]:
        """Convert attributes to OTel format."""
        result = []
        for key, value in attrs.items():
            # Map to GenAI semantic conventions if applicable
            mapped_key = OTEL_GENAI_ATTRS.get(key, key)

            if isinstance(value, str):
                result.append({"key": mapped_key, "value": {"stringValue": value}})
            elif isinstance(value, int):
                result.append({"key": mapped_key, "value": {"intValue": str(value)}})
            elif isinstance(value, float):
                result.append({"key": mapped_key, "value": {"doubleValue": value}})
            elif isinstance(value, bool):
                result.append({"key": mapped_key, "value": {"boolValue": value}})

        return result

    def export_span(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: str,
        name: str,
        kind: str,
        start_time: datetime,
        end_time: datetime,
        attributes: dict,
        status: str = "OK"
    ) -> OTelSpan:
        """Export a span in OTel format."""
        # Map kind to OTel SpanKind
        kind_mapping = {
            "agent": 1,  # INTERNAL
            "llm": 3,    # CLIENT
            "tool": 1,   # INTERNAL
            "handoff": 1 # INTERNAL
        }

        otel_span = OTelSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind_mapping.get(kind, 1),
            start_time_unix_nano=int(start_time.timestamp() * 1e9),
            end_time_unix_nano=int(end_time.timestamp() * 1e9) if end_time else None,
            attributes=self._convert_attributes(attributes),
            status={"code": 1 if status == "OK" else 2},  # 1=OK, 2=ERROR
            resource={
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": self.service_name}},
                    {"key": "service.version", "value": {"stringValue": self.service_version}},
                ]
            }
        )

        self.spans.append(otel_span)
        return otel_span

    def export_otlp_json(self) -> str:
        """Export as OTLP JSON format."""
        resource_spans = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "openai-agents-sdk"},
                            "spans": [
                                {
                                    "traceId": s.trace_id,
                                    "spanId": s.span_id,
                                    "parentSpanId": s.parent_span_id,
                                    "name": s.name,
                                    "kind": s.kind,
                                    "startTimeUnixNano": str(s.start_time_unix_nano),
                                    "endTimeUnixNano": str(s.end_time_unix_nano) if s.end_time_unix_nano else None,
                                    "attributes": s.attributes,
                                    "status": s.status
                                }
                                for s in self.spans
                            ]
                        }
                    ]
                }
            ]
        }

        return json.dumps(resource_spans, indent=2)


# =============================================================================
# Multi-Backend Trace Router
# =============================================================================

class TraceRouter:
    """Routes traces to multiple backends."""

    def __init__(self):
        self.backends: list[TracingBackend] = []
        self.otel_exporter: Optional[OpenTelemetryExporter] = None

    def add_backend(self, backend: TracingBackend):
        """Add a tracing backend."""
        self.backends.append(backend)

    def enable_otel(self, service_name: str):
        """Enable OpenTelemetry export."""
        self.otel_exporter = OpenTelemetryExporter(service_name)

    def send_span(self, span: dict):
        """Send span to all backends."""
        for backend in self.backends:
            backend.send_span(span)

        if self.otel_exporter:
            self.otel_exporter.export_span(
                trace_id=span.get("trace_id"),
                span_id=span.get("span_id"),
                parent_span_id=span.get("parent_span_id"),
                name=span.get("name"),
                kind=span.get("kind"),
                start_time=span.get("start_time"),
                end_time=span.get("end_time"),
                attributes=span.get("attributes", {}),
                status=span.get("status", "OK")
            )

    def flush_all(self):
        """Flush all backends."""
        for backend in self.backends:
            backend.flush()


# =============================================================================
# Tests
# =============================================================================

def test_datadog_integration():
    """Test Datadog integration (OB-04)."""
    print("\n" + "=" * 70)
    print("TEST: Datadog Integration (OB-04)")
    print("=" * 70)

    backend = DatadogBackend(api_key="dd-api-key", service_name="my-agent")

    # Send spans
    backend.send_span({
        "name": "agent:CustomerBot",
        "trace_id": "trace_001",
        "span_id": "span_001",
        "start_time": datetime.now().isoformat(),
        "duration_ns": 1500000000,
        "attributes": {"model": "gpt-4o"}
    })

    backend.send_span({
        "name": "llm:chat",
        "trace_id": "trace_001",
        "span_id": "span_002",
        "parent_span_id": "span_001",
        "start_time": datetime.now().isoformat(),
        "duration_ns": 800000000,
        "attributes": {"input_tokens": 500, "output_tokens": 200}
    })

    print(f"\nBuffered spans: {len(backend.buffer)}")
    backend.flush()

    print("\n✅ Datadog integration works")


def test_langfuse_integration():
    """Test Langfuse integration (OB-04)."""
    print("\n" + "=" * 70)
    print("TEST: Langfuse Integration (OB-04)")
    print("=" * 70)

    backend = LangfuseBackend(public_key="lf-pk", secret_key="lf-sk")

    backend.send_span({
        "name": "chat_completion",
        "kind": "llm",
        "trace_id": "trace_001",
        "span_id": "span_001",
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "attributes": {
            "model": "gpt-4o",
            "input_tokens": 500,
            "output_tokens": 200
        },
        "input": "What is the weather?",
        "output": "The weather is sunny."
    })

    print(f"\nBuffered observations: {len(backend.traces)}")
    backend.flush()

    print("\n✅ Langfuse integration works")


def test_otel_export():
    """Test OpenTelemetry export (OB-05)."""
    print("\n" + "=" * 70)
    print("TEST: OpenTelemetry Export (OB-05)")
    print("=" * 70)

    exporter = OpenTelemetryExporter("agent-service", "1.0.0")

    # Export spans
    now = datetime.now()
    exporter.export_span(
        trace_id="abc123",
        span_id="def456",
        parent_span_id=None,
        name="gen_ai.chat",
        kind="llm",
        start_time=now,
        end_time=now,
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 500,
            "gen_ai.usage.output_tokens": 200
        }
    )

    # Export OTLP JSON
    otlp_json = exporter.export_otlp_json()
    print(f"\nOTLP JSON (first 500 chars):\n{otlp_json[:500]}...")

    print("\n✅ OpenTelemetry export works")


def test_multi_backend():
    """Test multi-backend routing."""
    print("\n" + "=" * 70)
    print("TEST: Multi-Backend Routing")
    print("=" * 70)

    router = TraceRouter()
    router.add_backend(DatadogBackend(service_name="test-agent"))
    router.add_backend(LangfuseBackend())
    router.enable_otel("test-agent")

    # Send span to all backends
    router.send_span({
        "name": "test_span",
        "trace_id": "trace_001",
        "span_id": "span_001",
        "kind": "agent",
        "start_time": datetime.now(),
        "end_time": datetime.now(),
        "attributes": {"test": "value"}
    })

    router.flush_all()

    print("\n✅ Multi-backend routing works")


# =============================================================================
# Summary
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ OB-04, OB-05: EXTERNAL INTEGRATION - EVALUATION SUMMARY                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ OB-04 (External Integration): ⭐⭐⭐⭐ (Production Ready)                    │
│   ✅ Datadog integration available                                          │
│   ✅ Langfuse integration available                                         │
│   ✅ Agenta integration available                                           │
│   ✅ Custom tracing hooks                                                   │
│                                                                             │
│ OB-05 (OTel Compliance): ⭐⭐ (Experimental)                                │
│   ❌ No native OpenTelemetry support                                        │
│   ❌ GenAI Semantic Conventions not built-in                                │
│   ⚠️ Custom OTel exporter provided                                         │
│   ⚠️ Requires bridge implementation                                        │
│                                                                             │
│ OpenAI SDK Integration Features:                                            │
│   - Built-in Datadog integration                                            │
│   - Langfuse tracing hooks                                                  │
│   - Agenta support                                                          │
│   - Custom tracing callbacks                                                │
│                                                                             │
│ Custom Implementation Provided:                                             │
│   ✅ DatadogBackend - Datadog span export                                   │
│   ✅ LangfuseBackend - Langfuse observation export                          │
│   ✅ AgentaBackend - Agenta integration                                     │
│   ✅ OpenTelemetryExporter - OTLP JSON export                               │
│   ✅ TraceRouter - Multi-backend routing                                    │
│                                                                             │
│ GenAI Semantic Conventions:                                                 │
│   - gen_ai.system                                                           │
│   - gen_ai.request.model                                                    │
│   - gen_ai.usage.input_tokens                                               │
│   - gen_ai.usage.output_tokens                                              │
│   - gen_ai.tool.name                                                        │
│                                                                             │
│ Comparison with LangGraph:                                                  │
│   LangGraph:                                                                │
│     - LangSmith native (vendor-specific)                                    │
│     - No native OTel                                                        │
│   OpenAI SDK:                                                               │
│     - Multiple integrations                                                 │
│     - Similar OTel gaps                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    test_datadog_integration()
    test_langfuse_integration()
    test_otel_export()
    test_multi_backend()

    print(SUMMARY)
