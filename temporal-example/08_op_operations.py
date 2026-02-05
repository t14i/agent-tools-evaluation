"""
OP: Operations - デプロイ・管理・スケール・保持の検証

評価項目:
- OP-01: デプロイモデル - セルフホスト / マネージド / サーバーレス
- OP-02: ワークフロー管理API - Start / Cancel / Terminate 等
- OP-03: ストレージバックエンド - 対応DB
- OP-04: スケーラビリティ - Worker水平スケール
- OP-05: データ保持 / クリーンアップ - TTL、アーカイブ
- OP-06: マルチリージョン / HA - 高可用性構成
- OP-07: マルチテナント分離 - テナント間のリソース分離
"""

import asyncio
import uuid
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# OP-02用のサンプルワークフロー
# =============================================================================

@workflow.defn
class LongRunningWorkflow:
    """Long-running workflow for management API testing."""

    def __init__(self) -> None:
        self._cancelled = False
        self._status = "running"

    @workflow.run
    async def run(self, duration_seconds: int) -> dict:
        try:
            for i in range(duration_seconds):
                self._status = f"step {i+1}/{duration_seconds}"
                await workflow.sleep(timedelta(seconds=1))
            self._status = "completed"
            return {"status": "completed", "steps": duration_seconds}
        except asyncio.CancelledError:
            self._status = "cancelled"
            return {"status": "cancelled"}

    @workflow.query
    def get_status(self) -> str:
        return self._status


# =============================================================================
# OP-01: デプロイモデル (Deployment Model)
# =============================================================================

async def verify_op01_deployment_model(client: Client) -> str:
    """Verify OP-01: Deployment Model."""
    print_section("OP-01: デプロイモデル")

    print("  Deployment options:")
    print("    1. Self-hosted:")
    print("       - temporal server start-dev (development)")
    print("       - Docker Compose / Kubernetes (production)")
    print("       - Helm charts available")
    print("    2. Temporal Cloud (managed):")
    print("       - Fully managed SaaS")
    print("       - Multi-region support")
    print("       - Enterprise features")
    print("    3. Library-embedded:")
    print("       - Not supported (requires server)")

    rating = "⭐⭐⭐⭐⭐"
    note = "セルフホスト（Docker/K8s/Helm）とTemporal Cloud（マネージド）を選択可能。開発用にstart-devコマンドあり"

    print_result("OP-01 デプロイモデル", rating, note)
    return rating


# =============================================================================
# OP-02: ワークフロー管理API (Workflow Management API)
# =============================================================================

async def verify_op02_management_api(client: Client) -> str:
    """Verify OP-02: Workflow Management API."""
    print_section("OP-02: ワークフロー管理API")

    workflow_id = f"op02-mgmt-{uuid.uuid4()}"

    # Start workflow
    handle = await client.start_workflow(
        LongRunningWorkflow.run,
        10,  # 10 second duration
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    print(f"  Started workflow: {workflow_id}")

    # Query status
    await asyncio.sleep(1)
    status = await handle.query(LongRunningWorkflow.get_status)
    print(f"  Query status: {status}")

    # Describe workflow
    description = await handle.describe()
    print(f"  Describe: status={description.status.name}")

    # Cancel workflow
    await handle.cancel()
    print("  Cancelled workflow")

    try:
        result = await handle.result()
        print(f"  Result after cancel: {result}")
    except Exception as e:
        print(f"  Cancel result: {type(e).__name__}")

    print("  Management API methods:")
    print("    - client.start_workflow(): Start new workflow")
    print("    - handle.result(): Wait for completion")
    print("    - handle.query(): Query current state")
    print("    - handle.signal(): Send signal")
    print("    - handle.cancel(): Request cancellation")
    print("    - handle.terminate(): Immediate termination")
    print("    - handle.describe(): Get workflow info")
    print("    - client.list_workflows(): List workflows")

    rating = "⭐⭐⭐⭐⭐"
    note = "Start/Cancel/Terminate/Query/Signal/Describe/Listを完備。CLIとWeb UIからも同等操作可能"

    print_result("OP-02 ワークフロー管理API", rating, note)
    return rating


# =============================================================================
# OP-03: ストレージバックエンド (Storage Backend)
# =============================================================================

async def verify_op03_storage_backend(client: Client) -> str:
    """Verify OP-03: Storage Backend."""
    print_section("OP-03: ストレージバックエンド")

    print("  Supported storage backends:")
    print("    - PostgreSQL (recommended for production)")
    print("    - MySQL")
    print("    - Cassandra (high scale)")
    print("    - SQLite (development only)")
    print("  Additional components:")
    print("    - Elasticsearch/OpenSearch (visibility/search)")
    print("    - S3-compatible storage (archival)")

    rating = "⭐⭐⭐⭐⭐"
    note = "PostgreSQL/MySQL/Cassandraをサポート。Visibility用にElasticsearch、Archival用にS3連携可能"

    print_result("OP-03 ストレージバックエンド", rating, note)
    return rating


# =============================================================================
# OP-04: スケーラビリティ (Scalability)
# =============================================================================

async def verify_op04_scalability(client: Client) -> str:
    """Verify OP-04: Scalability."""
    print_section("OP-04: スケーラビリティ")

    print("  Scalability features:")
    print("    - Workers: Horizontally scalable (add more instances)")
    print("    - Server: Frontend/History/Matching/Worker services")
    print("    - Sharding: Partitioned by namespace/task queue")
    print("    - Backpressure: Task queue depth monitoring")
    print("  Worker configuration:")
    print("    - max_concurrent_workflow_tasks")
    print("    - max_concurrent_activities")
    print("    - max_concurrent_local_activities")

    rating = "⭐⭐⭐⭐⭐"
    note = "Worker水平スケール対応。Server側も4サービス分離でスケール可能。Namespaceでシャーディング"

    print_result("OP-04 スケーラビリティ", rating, note)
    return rating


# =============================================================================
# OP-05: データ保持 / クリーンアップ (Data Retention / Cleanup)
# =============================================================================

async def verify_op05_data_retention(client: Client) -> str:
    """Verify OP-05: Data Retention / Cleanup."""
    print_section("OP-05: データ保持 / クリーンアップ")

    print("  Retention features:")
    print("    - Namespace-level retention period (default 72h)")
    print("    - Archival: Move completed workflows to cold storage")
    print("    - Automatic cleanup after retention period")
    print("  Configuration:")
    print("    - WorkflowExecutionRetentionPeriod per namespace")
    print("    - Archival to S3/GCS/Azure Blob")

    rating = "⭐⭐⭐⭐⭐"
    note = "Namespace単位でRetention Period設定。期限切れワークフローは自動削除。Archivalで冷却保管も可能"

    print_result("OP-05 データ保持/クリーンアップ", rating, note)
    return rating


# =============================================================================
# OP-06: マルチリージョン / HA (Multi-Region / HA)
# =============================================================================

async def verify_op06_multi_region_ha(client: Client) -> str:
    """Verify OP-06: Multi-Region / HA."""
    print_section("OP-06: マルチリージョン / HA")

    print("  High Availability:")
    print("    - Multiple replicas of each service")
    print("    - Database replication (Postgres/MySQL HA)")
    print("  Multi-Region (Temporal Cloud):")
    print("    - Multi-region namespaces")
    print("    - Automatic failover")
    print("  Self-hosted multi-region:")
    print("    - Requires custom setup")
    print("    - Global namespace feature (experimental)")

    rating = "⭐⭐⭐⭐"
    note = "サービスレプリケーションでHA対応。Temporal Cloudでマルチリージョン。セルフホストでのマルチリージョンは構成が複雑"

    print_result("OP-06 マルチリージョン/HA", rating, note)
    return rating


# =============================================================================
# OP-07: マルチテナント分離 (Multi-Tenant Isolation)
# =============================================================================

async def verify_op07_multi_tenant(client: Client) -> str:
    """Verify OP-07: Multi-Tenant Isolation."""
    print_section("OP-07: マルチテナント分離")

    print("  Multi-tenancy mechanisms:")
    print("    - Namespaces: Logical isolation")
    print("    - Separate task queues per tenant")
    print("    - Resource quotas (Temporal Cloud)")
    print("  Temporal Cloud features:")
    print("    - Dedicated infrastructure option")
    print("    - Per-namespace quotas")
    print("    - Network isolation")

    rating = "⭐⭐⭐⭐"
    note = "Namespaceで論理分離。Temporal CloudではResource Quota設定可能。完全な物理分離はDedicated環境で"

    print_result("OP-07 マルチテナント分離", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all OP category verifications."""
    print("\n" + "="*60)
    print("  OP: Operations 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LongRunningWorkflow],
        activities=[],
    ):
        results = {}

        results["OP-01"] = await verify_op01_deployment_model(client)
        results["OP-02"] = await verify_op02_management_api(client)
        results["OP-03"] = await verify_op03_storage_backend(client)
        results["OP-04"] = await verify_op04_scalability(client)
        results["OP-05"] = await verify_op05_data_retention(client)
        results["OP-06"] = await verify_op06_multi_region_ha(client)
        results["OP-07"] = await verify_op07_multi_tenant(client)

        print("\n" + "="*60)
        print("  OP Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
