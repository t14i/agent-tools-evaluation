"""
VR: Versioning & Migration - コード変更・スキーマ変更への対応の検証

評価項目:
- VR-01: ワークフローバージョニング - コード変更時に既存実行を壊さない仕組み
- VR-02: 非互換変更の検出 - 破壊的変更を事前に検出できるか
- VR-03: マイグレーション戦略 - 旧→新への移行手順
- VR-04: スキーマ進化 - 入出力スキーマ変更への耐性
"""

import asyncio
import uuid
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# VR-01: ワークフローバージョニング (Workflow Versioning)
# =============================================================================

@activity.defn
async def process_v1(data: str) -> str:
    """V1 processing activity."""
    return f"v1_processed:{data}"


@activity.defn
async def process_v2(data: str) -> str:
    """V2 processing activity."""
    return f"v2_processed:{data}"


@workflow.defn
class VersionedWorkflow:
    """Workflow demonstrating versioning with workflow.patched()."""

    @workflow.run
    async def run(self, data: str) -> dict:
        results = {}

        # Version check using patched()
        # When code changes, use patched() to maintain compatibility
        if workflow.patched("v2-processing"):
            # New code path for new executions
            results["processing"] = await workflow.execute_activity(
                process_v2,
                data,
                start_to_close_timeout=timedelta(seconds=30),
            )
            results["version"] = "v2"
        else:
            # Old code path for existing executions
            results["processing"] = await workflow.execute_activity(
                process_v1,
                data,
                start_to_close_timeout=timedelta(seconds=30),
            )
            results["version"] = "v1"

        # Another version check
        if workflow.patched("v2-extra-step"):
            results["extra"] = "new_feature_enabled"

        return results


async def verify_vr01_workflow_versioning(client: Client) -> str:
    """Verify VR-01: Workflow Versioning."""
    print_section("VR-01: ワークフローバージョニング")

    workflow_id = f"vr01-version-{uuid.uuid4()}"

    handle = await client.start_workflow(
        VersionedWorkflow.run,
        "test-data",
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Versioning mechanisms:")
    print("    1. workflow.patched('patch-id'): Code branching for compatibility")
    print("    2. Worker Versioning (Build ID): Route workflows to specific workers")
    print("    3. deprecate_patch(): Mark old code paths for removal")

    rating = "⭐⭐⭐⭐⭐"
    note = "workflow.patched()でコード内分岐、Worker Versioning（Build ID）で新旧Workerを分離。自動drain/retire機能あり"

    print_result("VR-01 ワークフローバージョニング", rating, note)
    return rating


# =============================================================================
# VR-02: 非互換変更の検出 (Breaking Change Detection)
# =============================================================================

async def verify_vr02_breaking_change_detection(client: Client) -> str:
    """Verify VR-02: Breaking Change Detection."""
    print_section("VR-02: 非互換変更の検出")

    print("  Breaking change detection mechanisms:")
    print("    - Replay testing: Run workflow code against Event History")
    print("    - NonDeterministicWorkflowError on mismatch")
    print("    - Temporal CLI: temporal workflow show for history inspection")
    print("    - SDK replay test utilities")
    print("  Detection timing:")
    print("    - Fail-fast on replay (production)")
    print("    - Can test pre-deployment with history export")

    rating = "⭐⭐⭐⭐"
    note = "Replay時にNonDeterministicWorkflowErrorでFail-fast。事前検出はReplay Test + Event History exportで可能だがセットアップが必要"

    print_result("VR-02 非互換変更の検出", rating, note)
    return rating


# =============================================================================
# VR-03: マイグレーション戦略 (Migration Strategy)
# =============================================================================

async def verify_vr03_migration_strategy(client: Client) -> str:
    """Verify VR-03: Migration Strategy."""
    print_section("VR-03: マイグレーション戦略")

    print("  Migration strategies:")
    print("    1. Worker Versioning with Build IDs:")
    print("       - Assign Build ID to worker")
    print("       - Route new workflows to new Build ID")
    print("       - Old workers drain existing workflows")
    print("    2. Task Queue based routing:")
    print("       - Different queues for different versions")
    print("       - Explicit routing control")
    print("    3. Gradual rollout:")
    print("       - Start new workflows on new version")
    print("       - Wait for old workflows to complete")

    rating = "⭐⭐⭐⭐⭐"
    note = "Build ID Versioning + Task Queue ルーティングで新旧分離。旧Workerの自動drainあり。段階的移行が標準サポート"

    print_result("VR-03 マイグレーション戦略", rating, note)
    return rating


# =============================================================================
# VR-04: スキーマ進化 (Schema Evolution)
# =============================================================================

@workflow.defn
class SchemaEvolutionWorkflow:
    """Workflow demonstrating schema evolution handling."""

    @workflow.run
    async def run(self, input_data: dict) -> dict:
        # Handle optional new fields with defaults
        version = input_data.get("version", 1)
        name = input_data.get("name", "default")

        # New field in v2
        extra = input_data.get("extra_field", None)

        return {
            "version": version,
            "name": name,
            "extra": extra,
            "processed": True,
        }


async def verify_vr04_schema_evolution(client: Client) -> str:
    """Verify VR-04: Schema Evolution."""
    print_section("VR-04: スキーマ進化")

    # Test with v1 schema (minimal)
    workflow_id_v1 = f"vr04-schema-v1-{uuid.uuid4()}"
    handle_v1 = await client.start_workflow(
        SchemaEvolutionWorkflow.run,
        {"name": "test"},  # v1 schema
        id=workflow_id_v1,
        task_queue=TASK_QUEUE,
    )
    result_v1 = await handle_v1.result()
    print(f"  V1 schema result: {result_v1}")

    # Test with v2 schema (extended)
    workflow_id_v2 = f"vr04-schema-v2-{uuid.uuid4()}"
    handle_v2 = await client.start_workflow(
        SchemaEvolutionWorkflow.run,
        {"name": "test", "version": 2, "extra_field": "new"},  # v2 schema
        id=workflow_id_v2,
        task_queue=TASK_QUEUE,
    )
    result_v2 = await handle_v2.result()
    print(f"  V2 schema result: {result_v2}")

    print("  Schema evolution strategies:")
    print("    - Use dict with .get() for optional fields")
    print("    - Pydantic models with default values")
    print("    - Payload converters for custom serialization")

    rating = "⭐⭐⭐⭐"
    note = "dict.get()やPydanticのdefaultでスキーマ進化に対応。Payload Converterで高度なシリアライズ制御も可能"

    print_result("VR-04 スキーマ進化", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all VR category verifications."""
    print("\n" + "="*60)
    print("  VR: Versioning & Migration 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[VersionedWorkflow, SchemaEvolutionWorkflow],
        activities=[process_v1, process_v2],
    ):
        results = {}

        results["VR-01"] = await verify_vr01_workflow_versioning(client)
        results["VR-02"] = await verify_vr02_breaking_change_detection(client)
        results["VR-03"] = await verify_vr03_migration_strategy(client)
        results["VR-04"] = await verify_vr04_schema_evolution(client)

        print("\n" + "="*60)
        print("  VR Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
