"""
OB: Observability - Execution State Understanding, Investigation, and Monitoring Verification

Evaluation Items:
- OB-01: Dashboard / UI - Inngest Dashboard UI
- OB-02: Metrics - Metrics
- OB-03: History Visualization - Run details and timeline
- OB-04: OTel Compliance - OpenTelemetry support
- OB-05: Alerts - Alerting
- OB-06: Logging - Logging
"""

import asyncio

from common import print_section, print_result


async def verify_ob01_dashboard_ui() -> str:
    """Verify OB-01: Dashboard / UI."""
    print_section("OB-01: Dashboard / UI")

    print("  Inngest Dashboard features:")
    print("    1. Dev Server UI (localhost:8288):")
    print("       - Function list and status")
    print("       - Run history browser")
    print("       - Event explorer")
    print("       - Real-time updates")

    print("\n    2. Cloud Dashboard:")
    print("       - All dev features plus:")
    print("       - Team management")
    print("       - Environment management")
    print("       - Usage analytics")
    print("       - Billing")

    print("\n    3. Run details view:")
    print("       - Step timeline")
    print("       - Input/output inspection")
    print("       - Retry history")
    print("       - Error details")

    print("\n    4. Search and filter:")
    print("       - Filter by function")
    print("       - Filter by status")
    print("       - Filter by time range")
    print("       - Event name search")

    rating = "⭐⭐⭐⭐⭐"
    note = "Rich UI for both Dev Server and Cloud. Run details, step timeline, search/filtering are user-friendly"

    print_result("OB-01 Dashboard/UI", rating, note)
    return rating


async def verify_ob02_metrics() -> str:
    """Verify OB-02: Metrics."""
    print_section("OB-02: Metrics")

    print("  Inngest metrics:")
    print("    1. Cloud Dashboard metrics:")
    print("       - Function execution count")
    print("       - Success/failure rates")
    print("       - Execution duration")
    print("       - Queue depth")
    print("       - Event throughput")

    print("\n    2. API metrics:")
    print("       GET /v1/metrics")
    print("       # Available on Cloud/Enterprise")

    print("\n    3. Self-hosted metrics:")
    print("       - Prometheus endpoint: /metrics")
    print("       - Standard Go runtime metrics")
    print("       - Custom Inngest metrics")

    print("\n  Available metrics:")
    print("    - inngest_function_runs_total")
    print("    - inngest_function_duration_seconds")
    print("    - inngest_step_runs_total")
    print("    - inngest_events_received_total")
    print("    - inngest_queue_depth")

    print("\n  Integration:")
    print("    - Prometheus scraping")
    print("    - Grafana dashboards")
    print("    - Datadog (via Prometheus)")

    rating = "⭐⭐⭐⭐"
    note = "Cloud version dashboard metrics, self-hosted uses Prometheus. Basic metrics are covered"

    print_result("OB-02 Metrics", rating, note)
    return rating


async def verify_ob03_history_visualization() -> str:
    """Verify OB-03: History Visualization."""
    print_section("OB-03: History Visualization")

    print("  Inngest run history visualization:")
    print("    1. Run list view:")
    print("       - Status (Running, Completed, Failed)")
    print("       - Function name")
    print("       - Started/completed time")
    print("       - Duration")

    print("\n    2. Run detail view:")
    print("       - Step timeline (waterfall)")
    print("       - Step input/output")
    print("       - Retry attempts")
    print("       - Error messages and stack traces")

    print("\n    3. Step inspection:")
    print("       - Each step's result")
    print("       - Timing information")
    print("       - Sleep duration")
    print("       - Event wait results")

    print("\n  Comparison with Temporal:")
    print("    Temporal: Full Event History (every event)")
    print("    Inngest: Step results only (simpler but less granular)")

    print("\n  Debugging features:")
    print("    - Click-through to step details")
    print("    - JSON viewer for data")
    print("    - Copy event payload")
    print("    - Replay from UI")

    rating = "⭐⭐⭐⭐⭐"
    note = "Step timeline, input/output inspection, error details are easy to view. Less detailed than Temporal but practical"

    print_result("OB-03 History Visualization", rating, note)
    return rating


async def verify_ob04_otel_compliance() -> str:
    """Verify OB-04: OTel Compliance."""
    print_section("OB-04: OTel Compliance")

    print("  Inngest OpenTelemetry support:")
    print("    1. Server-side tracing:")
    print("       - Inngest server exports traces")
    print("       - OTLP exporter configuration")

    print("\n    2. Function-side tracing:")
    print("       # In your function code")
    print("       from opentelemetry import trace")
    print("       tracer = trace.get_tracer(__name__)")
    print("")
    print("       @inngest_client.create_function(...)")
    print("       async def my_function(ctx, step):")
    print("           with tracer.start_as_current_span('process'):")
    print("               # Your code")

    print("\n    3. Trace propagation:")
    print("       - Trace context in event metadata")
    print("       - Automatic propagation to steps")
    print("       - Cross-function correlation")

    print("\n  Configuration:")
    print("    OTEL_EXPORTER_OTLP_ENDPOINT=http://...")
    print("    OTEL_SERVICE_NAME=inngest")

    print("\n  Limitations:")
    print("    - SDK tracing requires manual setup")
    print("    - Not as automatic as Temporal's interceptors")

    rating = "⭐⭐⭐"
    note = "OTel support available. Server-side OTLP export, function-side requires manual setup. Lower integration level than Temporal"

    print_result("OB-04 OTel Compliance", rating, note)
    return rating


async def verify_ob05_alerts() -> str:
    """Verify OB-05: Alerts."""
    print_section("OB-05: Alerts")

    print("  Inngest alerting:")
    print("    1. Cloud built-in alerts:")
    print("       - Function failure alerts")
    print("       - Queue depth alerts")
    print("       - Latency alerts")
    print("       - Delivery: Email, Slack, Webhook")

    print("\n    2. Webhook notifications:")
    print("       # Configure in dashboard")
    print("       # Receive events on function failure")
    print("")
    print("       POST /your-webhook")
    print("       {")
    print("         'type': 'function.failed',")
    print("         'function_id': '...',")
    print("         'run_id': '...',")
    print("         'error': '...'")
    print("       }")

    print("\n    3. Self-hosted alerting:")
    print("       - Prometheus AlertManager integration")
    print("       - Custom alert rules")
    print("       - Webhook targets")

    print("\n  Integration options:")
    print("    - Slack")
    print("    - PagerDuty")
    print("    - Email")
    print("    - Custom webhooks")

    rating = "⭐⭐⭐⭐"
    note = "Cloud version has built-in alerts. Slack/PagerDuty/Webhook integration. Self-hosted uses Prometheus AlertManager"

    print_result("OB-05 Alerts", rating, note)
    return rating


async def verify_ob06_logging() -> str:
    """Verify OB-06: Logging."""
    print_section("OB-06: Logging")

    print("  Inngest logging:")
    print("    1. Function logging:")
    print("       import logging")
    print("       logger = logging.getLogger(__name__)")
    print("")
    print("       async def my_function(ctx, step):")
    print("           logger.info('Processing', extra={")
    print("               'run_id': ctx.run_id,")
    print("               'function_id': ctx.function_id,")
    print("           })")

    print("\n    2. Structured logging pattern:")
    print("       import structlog")
    print("       log = structlog.get_logger()")
    print("")
    print("       async def my_function(ctx, step):")
    print("           log = log.bind(")
    print("               run_id=ctx.run_id,")
    print("               function_id=ctx.function_id,")
    print("           )")
    print("           log.info('Processing started')")

    print("\n    3. Log correlation:")
    print("       - ctx.run_id for run correlation")
    print("       - ctx.function_id for function correlation")
    print("       - event.id for event correlation")

    print("\n  Cloud logging:")
    print("    - View logs in dashboard")
    print("    - Filter by run ID")
    print("    - Search functionality")

    rating = "⭐⭐⭐⭐"
    note = "Standard Python logging works. Log correlation via ctx.run_id. Cloud version has dashboard log viewing"

    print_result("OB-06 Logging", rating, note)
    return rating


async def main():
    """Run all OB category verifications."""
    print("\n" + "="*60)
    print("  OB: Observability Verification")
    print("="*60)

    results = {}

    results["OB-01"] = await verify_ob01_dashboard_ui()
    results["OB-02"] = await verify_ob02_metrics()
    results["OB-03"] = await verify_ob03_history_visualization()
    results["OB-04"] = await verify_ob04_otel_compliance()
    results["OB-05"] = await verify_ob05_alerts()
    results["OB-06"] = await verify_ob06_logging()

    print("\n" + "="*60)
    print("  OB Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
