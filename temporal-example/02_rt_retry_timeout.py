"""
RT: Retry & Timeout - リトライ・タイムアウト・バックプレッシャーの検証

評価項目:
- RT-01: リトライ戦略 - 固定/指数バックオフ/ジッター、最大試行数
- RT-02: タイムアウト体系 - Start-to-close / Schedule-to-start等
- RT-03: サーキットブレーカ - 連続失敗時の自動停止
- RT-04: Heartbeat - 長時間Activityの生存確認
"""

import asyncio
import uuid
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# RT-01: リトライ戦略 (Retry Strategy)
# =============================================================================

retry_counter: dict[str, int] = {}


@activity.defn
async def flaky_activity(operation_id: str, fail_times: int) -> str:
    """Activity that fails a specified number of times before succeeding."""
    retry_counter[operation_id] = retry_counter.get(operation_id, 0) + 1
    count = retry_counter[operation_id]

    if count <= fail_times:
        raise ValueError(f"Intentional failure {count}/{fail_times}")

    return f"success after {count} attempts"


@workflow.defn
class RetryStrategyWorkflow:
    """Workflow demonstrating retry strategies."""

    @workflow.run
    async def run(self, operation_id: str, fail_times: int) -> str:
        # Configurable retry policy
        result = await workflow.execute_activity(
            flaky_activity,
            args=[operation_id, fail_times],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100),
                backoff_coefficient=2.0,  # Exponential backoff
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=5,
                non_retryable_error_types=["NonRetryableError"],
            ),
        )
        return result


async def verify_rt01_retry_strategy(client: Client) -> str:
    """Verify RT-01: Retry Strategy."""
    print_section("RT-01: リトライ戦略 (Retry Strategy)")

    operation_id = f"retry-{uuid.uuid4()}"
    workflow_id = f"rt01-retry-{uuid.uuid4()}"

    handle = await client.start_workflow(
        RetryStrategyWorkflow.run,
        args=[operation_id, 2],  # Fail 2 times, succeed on 3rd
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    attempts = retry_counter.get(operation_id, 0)
    print(f"  Result: {result}")
    print(f"  Total attempts: {attempts}")
    print("  RetryPolicy options:")
    print("    - initial_interval: Initial delay")
    print("    - backoff_coefficient: Exponential multiplier")
    print("    - maximum_interval: Max delay cap")
    print("    - maximum_attempts: Max retry count")
    print("    - non_retryable_error_types: Skip retry for specific errors")

    rating = "⭐⭐⭐⭐⭐"
    note = "指数バックオフ+ジッター対応。initial_interval/backoff_coefficient/max_attempts/non_retryable_typesを細かく設定可能"

    print_result("RT-01 リトライ戦略", rating, note)
    return rating


# =============================================================================
# RT-02: タイムアウト体系 (Timeout System)
# =============================================================================

@activity.defn
async def slow_activity(duration_seconds: float) -> str:
    """Activity that takes specified duration."""
    await asyncio.sleep(duration_seconds)
    return f"completed after {duration_seconds}s"


@workflow.defn
class TimeoutWorkflow:
    """Workflow demonstrating timeout configurations."""

    @workflow.run
    async def run(self, duration: float) -> str:
        result = await workflow.execute_activity(
            slow_activity,
            duration,
            # Temporal supports multiple timeout types
            schedule_to_close_timeout=timedelta(seconds=60),  # Total time budget
            start_to_close_timeout=timedelta(seconds=30),     # Per-attempt time
            schedule_to_start_timeout=timedelta(seconds=10),  # Queue wait time
            heartbeat_timeout=timedelta(seconds=5),           # Heartbeat interval
            retry_policy=RetryPolicy(maximum_attempts=1),     # No retry for this test
        )
        return result


async def verify_rt02_timeout_system(client: Client) -> str:
    """Verify RT-02: Timeout System."""
    print_section("RT-02: タイムアウト体系 (Timeout System)")

    workflow_id = f"rt02-timeout-{uuid.uuid4()}"

    handle = await client.start_workflow(
        TimeoutWorkflow.run,
        0.5,  # 0.5 second activity
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Timeout types available:")
    print("    - schedule_to_close_timeout: Total time from schedule to completion")
    print("    - start_to_close_timeout: Time per execution attempt")
    print("    - schedule_to_start_timeout: Max queue wait time")
    print("    - heartbeat_timeout: Heartbeat interval for long activities")

    rating = "⭐⭐⭐⭐⭐"
    note = "4種のタイムアウト（schedule-to-close/start-to-close/schedule-to-start/heartbeat）を完備。粒度の細かい制御が可能"

    print_result("RT-02 タイムアウト体系", rating, note)
    return rating


# =============================================================================
# RT-03: サーキットブレーカ (Circuit Breaker)
# =============================================================================

async def verify_rt03_circuit_breaker(client: Client) -> str:
    """Verify RT-03: Circuit Breaker."""
    print_section("RT-03: サーキットブレーカ (Circuit Breaker)")

    print("  Temporal's approach to circuit breaking:")
    print("    - No native circuit breaker in SDK")
    print("    - Can implement via max_attempts + non_retryable_error_types")
    print("    - Task queue rate limiting available on server side")
    print("    - Custom implementation needed for full circuit breaker pattern")

    rating = "⭐⭐⭐"
    note = "ネイティブサーキットブレーカはないが、max_attempts + non_retryable_typesで疑似的に実現可。Server側のRate Limitingも利用可能"

    print_result("RT-03 サーキットブレーカ", rating, note)
    return rating


# =============================================================================
# RT-04: Heartbeat
# =============================================================================

@activity.defn
async def long_running_activity(total_steps: int) -> str:
    """Long-running activity with heartbeat."""
    for step in range(total_steps):
        # Report progress via heartbeat
        activity.heartbeat(f"Step {step + 1}/{total_steps}")
        await asyncio.sleep(0.2)
    return f"completed {total_steps} steps"


@workflow.defn
class HeartbeatWorkflow:
    """Workflow demonstrating heartbeat mechanism."""

    @workflow.run
    async def run(self, steps: int) -> str:
        result = await workflow.execute_activity(
            long_running_activity,
            steps,
            start_to_close_timeout=timedelta(seconds=60),
            heartbeat_timeout=timedelta(seconds=5),  # Expect heartbeat every 5s
        )
        return result


async def verify_rt04_heartbeat(client: Client) -> str:
    """Verify RT-04: Heartbeat."""
    print_section("RT-04: Heartbeat")

    workflow_id = f"rt04-heartbeat-{uuid.uuid4()}"

    handle = await client.start_workflow(
        HeartbeatWorkflow.run,
        5,  # 5 steps
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Heartbeat features:")
    print("    - activity.heartbeat(details) sends progress")
    print("    - heartbeat_timeout detects stuck activities")
    print("    - Heartbeat details can carry progress info")
    print("    - On timeout, activity is considered failed and can retry")

    rating = "⭐⭐⭐⭐⭐"
    note = "activity.heartbeat()でプログレス報告。heartbeat_timeoutでハング検知。詳細データも送信可能で長時間Activity管理に最適"

    print_result("RT-04 Heartbeat", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all RT category verifications."""
    print("\n" + "="*60)
    print("  RT: Retry & Timeout 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RetryStrategyWorkflow, TimeoutWorkflow, HeartbeatWorkflow],
        activities=[flaky_activity, slow_activity, long_running_activity],
    ):
        results = {}

        results["RT-01"] = await verify_rt01_retry_strategy(client)
        results["RT-02"] = await verify_rt02_timeout_system(client)
        results["RT-03"] = await verify_rt03_circuit_breaker(client)
        results["RT-04"] = await verify_rt04_heartbeat(client)

        print("\n" + "="*60)
        print("  RT Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
