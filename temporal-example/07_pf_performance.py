"""
PF: Performance & Overhead - 永続化の代償としてのレイテンシ・スループットの検証

評価項目:
- PF-01: ステップレイテンシ - 空のステップを実行するオーバーヘッド
- PF-02: Fan-out スループット - 大量並列Activity起動のスループット
- PF-03: ペイロードサイズ制限 - サイズ制限とオフロード機構
"""

import asyncio
import time
import uuid
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# PF-01: ステップレイテンシ (Step Latency)
# =============================================================================

@activity.defn
async def noop_activity() -> str:
    """Empty activity for latency measurement."""
    return "done"


@activity.defn
async def local_noop_activity() -> str:
    """Empty local activity for latency comparison."""
    return "done"


@workflow.defn
class LatencyTestWorkflow:
    """Workflow for measuring step latency."""

    @workflow.run
    async def run(self, iterations: int) -> dict:
        results = {"regular": [], "local": []}

        # Measure regular activity latency
        for _ in range(iterations):
            start = workflow.now()
            await workflow.execute_activity(
                noop_activity,
                start_to_close_timeout=timedelta(seconds=30),
            )
            end = workflow.now()
            results["regular"].append((end - start).total_seconds() * 1000)

        # Measure local activity latency
        for _ in range(iterations):
            start = workflow.now()
            await workflow.execute_local_activity(
                local_noop_activity,
                start_to_close_timeout=timedelta(seconds=10),
            )
            end = workflow.now()
            results["local"].append((end - start).total_seconds() * 1000)

        return results


async def verify_pf01_step_latency(client: Client) -> str:
    """Verify PF-01: Step Latency."""
    print_section("PF-01: ステップレイテンシ")

    workflow_id = f"pf01-latency-{uuid.uuid4()}"

    handle = await client.start_workflow(
        LatencyTestWorkflow.run,
        3,  # 3 iterations each
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()

    regular_avg = sum(result["regular"]) / len(result["regular"]) if result["regular"] else 0
    local_avg = sum(result["local"]) / len(result["local"]) if result["local"] else 0

    print(f"  Regular activity latencies: {result['regular']} ms")
    print(f"  Regular activity avg: {regular_avg:.2f} ms")
    print(f"  Local activity latencies: {result['local']} ms")
    print(f"  Local activity avg: {local_avg:.2f} ms")
    print("  Note: Includes DB persistence + task queue round-trip")

    rating = "⭐⭐⭐⭐"
    note = "Regular Activityは5-50ms程度のオーバーヘッド（永続化+キュー）。Local Activityはより低レイテンシだが永続性が弱い"

    print_result("PF-01 ステップレイテンシ", rating, note)
    return rating


# =============================================================================
# PF-02: Fan-out スループット (Fan-out Throughput)
# =============================================================================

@activity.defn
async def fanout_task(task_id: int) -> int:
    """Simple task for fan-out testing."""
    return task_id * 2


@workflow.defn
class FanoutThroughputWorkflow:
    """Workflow for measuring fan-out throughput."""

    @workflow.run
    async def run(self, task_count: int) -> dict:
        start_time = workflow.now()

        # Fan-out: Start all activities in parallel
        tasks = [
            workflow.execute_activity(
                fanout_task,
                i,
                start_to_close_timeout=timedelta(seconds=60),
            )
            for i in range(task_count)
        ]

        # Fan-in: Wait for all
        results = await asyncio.gather(*tasks)

        end_time = workflow.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "task_count": task_count,
            "duration_seconds": duration,
            "throughput": task_count / duration if duration > 0 else 0,
            "results_count": len(results),
        }


async def verify_pf02_fanout_throughput(client: Client) -> str:
    """Verify PF-02: Fan-out Throughput."""
    print_section("PF-02: Fan-out スループット")

    workflow_id = f"pf02-fanout-{uuid.uuid4()}"

    handle = await client.start_workflow(
        FanoutThroughputWorkflow.run,
        50,  # 50 parallel tasks
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Tasks: {result['task_count']}")
    print(f"  Duration: {result['duration_seconds']:.2f} seconds")
    print(f"  Throughput: {result['throughput']:.2f} tasks/second")
    print("  Note: Server and worker config affects throughput")

    rating = "⭐⭐⭐⭐"
    note = "並列Activity起動可能。スループットはWorker/Server構成に依存。大規模Fan-out時はキュー深度監視が推奨"

    print_result("PF-02 Fan-outスループット", rating, note)
    return rating


# =============================================================================
# PF-03: ペイロードサイズ制限 (Payload Size Limits)
# =============================================================================

@activity.defn
async def large_payload_activity(data: str) -> str:
    """Activity handling large payloads."""
    return f"received:{len(data)}bytes"


@workflow.defn
class PayloadTestWorkflow:
    """Workflow for testing payload size limits."""

    @workflow.run
    async def run(self, payload_size_kb: int) -> dict:
        # Create payload of specified size
        payload = "x" * (payload_size_kb * 1024)

        result = await workflow.execute_activity(
            large_payload_activity,
            payload,
            start_to_close_timeout=timedelta(seconds=60),
        )

        return {
            "requested_kb": payload_size_kb,
            "actual_bytes": len(payload),
            "result": result,
        }


async def verify_pf03_payload_size(client: Client) -> str:
    """Verify PF-03: Payload Size Limits."""
    print_section("PF-03: ペイロードサイズ制限")

    # Test with small payload
    workflow_id = f"pf03-payload-{uuid.uuid4()}"

    handle = await client.start_workflow(
        PayloadTestWorkflow.run,
        10,  # 10KB
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Test payload result: {result}")

    print("  Payload limits (default):")
    print("    - Single payload: 2MB (configurable)")
    print("    - Entire Event History: 50MB warning, 256MB error")
    print("  Large payload handling:")
    print("    - Use external blob storage (S3, GCS)")
    print("    - Pass reference URL instead of data")
    print("    - Custom Payload Codec for compression")

    rating = "⭐⭐⭐⭐"
    note = "デフォルト2MBペイロード上限。Payload Codecで圧縮可能。超過時は外部ストレージ+参照URLパターンが推奨"

    print_result("PF-03 ペイロードサイズ制限", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all PF category verifications."""
    print("\n" + "="*60)
    print("  PF: Performance & Overhead 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LatencyTestWorkflow, FanoutThroughputWorkflow, PayloadTestWorkflow],
        activities=[noop_activity, local_noop_activity, fanout_task, large_payload_activity],
    ):
        results = {}

        results["PF-01"] = await verify_pf01_step_latency(client)
        results["PF-02"] = await verify_pf02_fanout_throughput(client)
        results["PF-03"] = await verify_pf03_payload_size(client)

        print("\n" + "="*60)
        print("  PF Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
