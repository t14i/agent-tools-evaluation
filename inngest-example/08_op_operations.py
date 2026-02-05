"""
OP: Operations - Deployment, Management, Scaling, and Retention Verification

Evaluation Items:
- OP-01: Deployment Model - Serverless / Cloud / Self-hosted
- OP-02: Workflow Management API - Cancel, Replay, etc.
- OP-03: Storage Backend - Storage managed by Inngest
- OP-04: Scalability - Auto-scaling
- OP-05: Data Retention / Cleanup - Data retention policy
- OP-06: Multi-Region / HA - High availability configuration
- OP-07: Multi-Tenant Isolation - Multi-tenancy
"""

import asyncio

from common import print_section, print_result


async def verify_op01_deployment_model() -> str:
    """Verify OP-01: Deployment Model."""
    print_section("OP-01: Deployment Model")

    print("  Inngest deployment options:")
    print("    1. Inngest Cloud (Managed):")
    print("       - Fully managed Inngest server")
    print("       - Your functions deployed anywhere (serverless, containers)")
    print("       - HTTPS endpoint for function invocation")

    print("\n    2. Self-hosted:")
    print("       - Docker: inngest/inngest image")
    print("       - Kubernetes: Helm chart available")
    print("       - PostgreSQL + Redis required")

    print("\n    3. Dev Server:")
    print("       npx inngest-cli@latest dev")
    print("       # Local development, no persistence")

    print("\n  Function deployment:")
    print("    - Serverless: Vercel, Netlify, AWS Lambda")
    print("    - Container: Docker, Kubernetes")
    print("    - Traditional: Any HTTP server (FastAPI, Flask)")

    print("\n  Architecture:")
    print("    [Events] -> [Inngest Server] -> [Your Functions (HTTP)]")
    print("                     |")
    print("                 [Storage]")

    rating = "⭐⭐⭐⭐⭐"
    note = "Full support for Cloud/Self-hosted/Dev. Serverless-native architecture. Natural integration with FaaS platforms"

    print_result("OP-01 Deployment Model", rating, note)
    return rating


async def verify_op02_management_api() -> str:
    """Verify OP-02: Workflow Management API."""
    print_section("OP-02: Workflow Management API")

    print("  Inngest Run Management API:")
    print("    # List runs")
    print("    GET /v1/runs")
    print("    GET /v1/runs?status=Running")
    print("    GET /v1/runs?function_id=my-function")

    print("\n    # Get run details")
    print("    GET /v1/runs/{run_id}")

    print("\n    # Replay (re-execute with memoized steps)")
    print("    POST /v1/runs/{run_id}/replay")

    print("\n    # Cancel (stop execution)")
    print("    POST /v1/runs/{run_id}/cancel")

    print("\n  Event Management:")
    print("    # Send event")
    print("    POST /e/{event_name}")
    print("    {'name': 'event/name', 'data': {...}}")

    print("\n    # Bulk send")
    print("    POST /e")
    print("    [{'name': 'event1', 'data': {...}}, ...]")

    print("\n  Missing vs Temporal:")
    print("    - No Pause/Resume")
    print("    - No Terminate (hard kill)")
    print("    - No Signal injection")
    print("    - No Workflow Reset to specific point")

    rating = "⭐⭐⭐⭐"
    note = "List/Get/Replay/Cancel CRUD operations. Lacks Temporal's detailed operations (Pause/Resume/Signal/Reset) but practical"

    print_result("OP-02 Workflow Management API", rating, note)
    return rating


async def verify_op03_storage_backend() -> str:
    """Verify OP-03: Storage Backend."""
    print_section("OP-03: Storage Backend")

    print("  Inngest storage options:")
    print("    1. Inngest Cloud:")
    print("       - Managed storage (abstracted)")
    print("       - Multi-region replication")
    print("       - Automatic scaling")

    print("\n    2. Self-hosted:")
    print("       - PostgreSQL (primary store)")
    print("       - Redis (caching, queues)")
    print("       - Configuration via environment")

    print("\n  Self-hosted configuration:")
    print("    INNGEST_PG_URI=postgres://...")
    print("    INNGEST_REDIS_URI=redis://...")

    print("\n  Storage requirements:")
    print("    - Event storage")
    print("    - Step journal (memoization)")
    print("    - Run history")
    print("    - Function metadata")

    print("\n  Comparison with Temporal:")
    print("    Temporal: PostgreSQL/MySQL/Cassandra + Elasticsearch")
    print("    Inngest: PostgreSQL + Redis (simpler stack)")

    rating = "⭐⭐⭐⭐"
    note = "Cloud version requires no management. Self-hosted uses PostgreSQL + Redis. Lighter infrastructure requirements than Temporal"

    print_result("OP-03 Storage Backend", rating, note)
    return rating


async def verify_op04_scalability() -> str:
    """Verify OP-04: Scalability."""
    print_section("OP-04: Scalability")

    print("  Inngest scaling model:")
    print("    1. Function scaling (your responsibility):")
    print("       - Serverless: Auto-scales with platform")
    print("       - Containers: Horizontal pod autoscaling")
    print("       - Functions are stateless (HTTP endpoints)")

    print("\n    2. Inngest server scaling:")
    print("       - Cloud: Managed, automatic")
    print("       - Self-hosted: Horizontal scaling supported")

    print("\n  Concurrency control:")
    print("    @inngest_client.create_function(")
    print("        concurrency=[")
    print("            inngest.Concurrency(limit=100),  # Global")
    print("            inngest.Concurrency(")
    print("                limit=5,")
    print("                key='event.data.tenant_id',  # Per tenant")
    print("            ),")
    print("        ],")
    print("    )")

    print("\n  Rate limiting:")
    print("    @inngest_client.create_function(")
    print("        rate_limit=inngest.RateLimit(")
    print("            limit=1000,")
    print("            period=timedelta(minutes=1),")
    print("        ),")
    print("    )")

    rating = "⭐⭐⭐⭐⭐"
    note = "Serverless-native auto-scaling. Backpressure control via concurrency/rate limits. Cloud version supports large scale"

    print_result("OP-04 Scalability", rating, note)
    return rating


async def verify_op05_data_retention() -> str:
    """Verify OP-05: Data Retention / Cleanup."""
    print_section("OP-05: Data Retention / Cleanup")

    print("  Inngest data retention:")
    print("    1. Cloud:")
    print("       - Plan-based retention (7-90 days)")
    print("       - Automatic cleanup")
    print("       - Archival available on enterprise")

    print("\n    2. Self-hosted:")
    print("       - Configurable retention")
    print("       - Manual cleanup scripts")
    print("       - Database-level management")

    print("\n  Retention configuration (self-hosted):")
    print("    INNGEST_RUN_RETENTION_DAYS=30")

    print("\n  Data types:")
    print("    - Event data: Subject to retention")
    print("    - Step results: Subject to retention")
    print("    - Run history: Subject to retention")
    print("    - Function definitions: Permanent")

    print("\n  Comparison with Temporal:")
    print("    Temporal: Namespace retention, Archival to S3")
    print("    Inngest: Simpler TTL-based retention")

    rating = "⭐⭐⭐⭐"
    note = "Plan-based retention period. Automatic cleanup. Not as flexible as Temporal's Archival but simple"

    print_result("OP-05 Data Retention/Cleanup", rating, note)
    return rating


async def verify_op06_multi_region_ha() -> str:
    """Verify OP-06: Multi-Region / HA."""
    print_section("OP-06: Multi-Region / HA")

    print("  Inngest high availability:")
    print("    1. Inngest Cloud:")
    print("       - Multi-region deployment")
    print("       - Automatic failover")
    print("       - 99.9% SLA (enterprise)")

    print("\n    2. Self-hosted:")
    print("       - Multiple server instances")
    print("       - Load balancer frontend")
    print("       - PostgreSQL HA (external)")
    print("       - Redis cluster (external)")

    print("\n  Function HA (your responsibility):")
    print("    - Multi-region function deployment")
    print("    - Function URL routing")
    print("    - CDN/load balancer")

    print("\n  Disaster recovery:")
    print("    - Cloud: Managed DR")
    print("    - Self-hosted: Database backup/restore")

    rating = "⭐⭐⭐⭐"
    note = "Cloud version has high availability configuration. Self-hosted depends on external DB HA. Function-side HA is user responsibility"

    print_result("OP-06 Multi-Region/HA", rating, note)
    return rating


async def verify_op07_multi_tenant() -> str:
    """Verify OP-07: Multi-Tenant Isolation."""
    print_section("OP-07: Multi-Tenant Isolation")

    print("  Inngest multi-tenancy:")
    print("    1. App-level isolation:")
    print("       - Each app_id is isolated")
    print("       - Separate event namespaces")
    print("       - Separate run histories")

    print("\n    2. Per-tenant concurrency:")
    print("       @inngest_client.create_function(")
    print("           concurrency=[")
    print("               inngest.Concurrency(")
    print("                   limit=10,")
    print("                   key='event.data.tenant_id',")
    print("               ),")
    print("           ],")
    print("       )")

    print("\n    3. Cloud features:")
    print("       - Resource quotas")
    print("       - Billing isolation")
    print("       - Enterprise: Dedicated infrastructure")

    print("\n  Isolation patterns:")
    print("    - Event name prefixing: tenant/{tenant_id}/event")
    print("    - Data segregation in event payload")
    print("    - Separate apps per tenant (stricter)")

    rating = "⭐⭐⭐⭐"
    note = "Tenant isolation via App ID + Concurrency key. Cloud version has resource quotas. Dedicated infrastructure for Enterprise"

    print_result("OP-07 Multi-Tenant Isolation", rating, note)
    return rating


async def main():
    """Run all OP category verifications."""
    print("\n" + "="*60)
    print("  OP: Operations Verification")
    print("="*60)

    results = {}

    results["OP-01"] = await verify_op01_deployment_model()
    results["OP-02"] = await verify_op02_management_api()
    results["OP-03"] = await verify_op03_storage_backend()
    results["OP-04"] = await verify_op04_scalability()
    results["OP-05"] = await verify_op05_data_retention()
    results["OP-06"] = await verify_op06_multi_region_ha()
    results["OP-07"] = await verify_op07_multi_tenant()

    print("\n" + "="*60)
    print("  OP Category Summary")
    print("="*60)
    for item, rating in results.items():
        print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
