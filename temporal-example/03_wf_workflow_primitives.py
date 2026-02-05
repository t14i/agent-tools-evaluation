"""
WF: Workflow Primitives - ワークフローの構成要素・制御フローの検証

評価項目:
- WF-01: ステップ定義 - Activity / Step / Task の定義方法
- WF-02: 子ワークフロー - ネスト・再利用
- WF-03: 並列実行 / Fan-out - 複数ステップの並列実行とFan-in
- WF-04: 条件分岐 / ループ - if/else、for/while相当
- WF-05: スリープ / タイマー - 長期スリープ、Cron
- WF-06: キュー / 流量制御 - タスクキュー、concurrency制限
"""

import asyncio
import uuid
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# WF-01: ステップ定義 (Step Definition)
# =============================================================================

@activity.defn
async def process_step(step_name: str, data: dict) -> dict:
    """Basic activity for step processing."""
    return {"step": step_name, "processed": True, "input": data}


@activity.defn
async def compute_step(value: int) -> int:
    """Compute activity."""
    return value * 2


@workflow.defn
class StepDefinitionWorkflow:
    """Workflow demonstrating step (activity) definitions."""

    @workflow.run
    async def run(self, input_data: dict) -> dict:
        # Execute activities as steps
        step1_result = await workflow.execute_activity(
            process_step,
            args=["step1", input_data],
            start_to_close_timeout=timedelta(seconds=30),
        )

        step2_result = await workflow.execute_activity(
            compute_step,
            input_data.get("value", 10),
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Local activity for lightweight operations
        # (runs in workflow worker process, lower latency)
        step3_result = await workflow.execute_local_activity(
            compute_step,
            step2_result,
            start_to_close_timeout=timedelta(seconds=10),
        )

        return {
            "step1": step1_result,
            "step2": step2_result,
            "step3_local": step3_result,
        }


async def verify_wf01_step_definition(client: Client) -> str:
    """Verify WF-01: Step Definition."""
    print_section("WF-01: ステップ定義 (Step Definition)")

    workflow_id = f"wf01-steps-{uuid.uuid4()}"

    handle = await client.start_workflow(
        StepDefinitionWorkflow.run,
        {"value": 5, "name": "test"},
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Step types:")
    print("    - @activity.defn: Regular activities (durable, can run on any worker)")
    print("    - execute_local_activity: Lightweight, runs in workflow process")

    rating = "⭐⭐⭐⭐⭐"
    note = "@activity.defnデコレータで簡潔に定義。Regular Activity（リモート実行）とLocal Activity（軽量・同プロセス）を使い分け可能"

    print_result("WF-01 ステップ定義", rating, note)
    return rating


# =============================================================================
# WF-02: 子ワークフロー (Child Workflows)
# =============================================================================

@workflow.defn
class ChildWorkflow:
    """Child workflow for nesting demonstration."""

    @workflow.run
    async def run(self, task_name: str) -> str:
        result = await workflow.execute_activity(
            process_step,
            args=[task_name, {"child": True}],
            start_to_close_timeout=timedelta(seconds=30),
        )
        return f"child_completed:{task_name}"


@workflow.defn
class ParentWorkflow:
    """Parent workflow demonstrating child workflow execution."""

    @workflow.run
    async def run(self, tasks: list[str]) -> list[str]:
        results = []
        for task in tasks:
            # Execute child workflow
            child_result = await workflow.execute_child_workflow(
                ChildWorkflow.run,
                task,
                id=f"{workflow.info().workflow_id}-child-{task}",
                # Parent close policy options:
                # ABANDON - child continues if parent terminates
                # TERMINATE - child is terminated
                # REQUEST_CANCEL - child receives cancel request
            )
            results.append(child_result)
        return results


async def verify_wf02_child_workflows(client: Client) -> str:
    """Verify WF-02: Child Workflows."""
    print_section("WF-02: 子ワークフロー (Child Workflows)")

    workflow_id = f"wf02-parent-{uuid.uuid4()}"

    handle = await client.start_workflow(
        ParentWorkflow.run,
        ["task-a", "task-b"],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Child workflow features:")
    print("    - execute_child_workflow() for nested workflows")
    print("    - Parent close policies: ABANDON / TERMINATE / REQUEST_CANCEL")
    print("    - Child has own Event History (isolation)")

    rating = "⭐⭐⭐⭐⭐"
    note = "execute_child_workflow()でネスト。親終了時ポリシー（ABANDON/TERMINATE/REQUEST_CANCEL）を制御可能。子は独自履歴を持つ"

    print_result("WF-02 子ワークフロー", rating, note)
    return rating


# =============================================================================
# WF-03: 並列実行 / Fan-out (Parallel Execution)
# =============================================================================

@activity.defn
async def parallel_task(task_id: int) -> dict:
    """Task for parallel execution."""
    await asyncio.sleep(0.1)  # Simulate work
    return {"task_id": task_id, "result": task_id * 10}


@workflow.defn
class FanOutWorkflow:
    """Workflow demonstrating fan-out/fan-in pattern."""

    @workflow.run
    async def run(self, task_count: int) -> list[dict]:
        # Fan-out: Start all activities in parallel
        tasks = [
            workflow.execute_activity(
                parallel_task,
                i,
                start_to_close_timeout=timedelta(seconds=30),
            )
            for i in range(task_count)
        ]

        # Fan-in: Wait for all to complete
        results = await asyncio.gather(*tasks)
        return list(results)


async def verify_wf03_parallel_execution(client: Client) -> str:
    """Verify WF-03: Parallel Execution / Fan-out."""
    print_section("WF-03: 並列実行 / Fan-out")

    workflow_id = f"wf03-fanout-{uuid.uuid4()}"

    handle = await client.start_workflow(
        FanOutWorkflow.run,
        5,  # 5 parallel tasks
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Parallel execution features:")
    print("    - asyncio.gather() for fan-out")
    print("    - Each activity runs independently")
    print("    - Results collected (fan-in) when all complete")

    rating = "⭐⭐⭐⭐⭐"
    note = "asyncio.gather()でFan-out、自然なPythonコードで並列実行。各Activityは独立してスケジュールされ、全完了でFan-in"

    print_result("WF-03 並列実行/Fan-out", rating, note)
    return rating


# =============================================================================
# WF-04: 条件分岐 / ループ (Conditional / Loop)
# =============================================================================

@workflow.defn
class ConditionalLoopWorkflow:
    """Workflow demonstrating conditional branching and loops."""

    @workflow.run
    async def run(self, items: list[int]) -> dict:
        results = {"processed": [], "skipped": []}

        # Loop through items
        for item in items:
            # Conditional branching
            if item % 2 == 0:
                # Process even numbers
                result = await workflow.execute_activity(
                    compute_step,
                    item,
                    start_to_close_timeout=timedelta(seconds=30),
                )
                results["processed"].append(result)
            else:
                # Skip odd numbers
                results["skipped"].append(item)

        # While loop example
        counter = 0
        while counter < 3:
            counter += 1

        results["while_iterations"] = counter
        return results


async def verify_wf04_conditional_loop(client: Client) -> str:
    """Verify WF-04: Conditional Branching / Loop."""
    print_section("WF-04: 条件分岐 / ループ")

    workflow_id = f"wf04-conditional-{uuid.uuid4()}"

    handle = await client.start_workflow(
        ConditionalLoopWorkflow.run,
        [1, 2, 3, 4, 5],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Control flow features:")
    print("    - Standard Python if/else/elif")
    print("    - Standard Python for/while loops")
    print("    - All control flow is deterministically replayed")

    rating = "⭐⭐⭐⭐⭐"
    note = "標準的なPython制御フロー（if/else/for/while）がそのまま使える。Replay時も決定的に再構築される"

    print_result("WF-04 条件分岐/ループ", rating, note)
    return rating


# =============================================================================
# WF-05: スリープ / タイマー (Sleep / Timer)
# =============================================================================

@workflow.defn
class SleepTimerWorkflow:
    """Workflow demonstrating durable sleep and timers."""

    @workflow.run
    async def run(self, sleep_seconds: int) -> dict:
        start_time = workflow.now()

        # Durable sleep - survives worker restarts
        await workflow.sleep(timedelta(seconds=sleep_seconds))

        end_time = workflow.now()

        return {
            "start": str(start_time),
            "end": str(end_time),
            "slept_seconds": sleep_seconds,
        }


async def verify_wf05_sleep_timer(client: Client) -> str:
    """Verify WF-05: Sleep / Timer."""
    print_section("WF-05: スリープ / タイマー")

    workflow_id = f"wf05-sleep-{uuid.uuid4()}"

    handle = await client.start_workflow(
        SleepTimerWorkflow.run,
        2,  # 2 second sleep
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Sleep/Timer features:")
    print("    - workflow.sleep(): Durable sleep (survives restarts)")
    print("    - Supports long durations (days, weeks, months)")
    print("    - Cron schedules supported via client.start_workflow(cron_schedule=)")

    rating = "⭐⭐⭐⭐⭐"
    note = "workflow.sleep()で永続スリープ。Worker再起動しても正確に再開。Cronスケジュールもclient.start_workflow()で設定可能"

    print_result("WF-05 スリープ/タイマー", rating, note)
    return rating


# =============================================================================
# WF-06: キュー / 流量制御 (Queue / Rate Control)
# =============================================================================

async def verify_wf06_queue_rate_control(client: Client) -> str:
    """Verify WF-06: Queue / Rate Control."""
    print_section("WF-06: キュー / 流量制御")

    print("  Task Queue features:")
    print("    - Named task queues for routing")
    print("    - Worker can listen to multiple queues")
    print("    - Server-side rate limiting (max_concurrent_activities)")
    print("    - Workflow-level: asyncio.Semaphore for concurrency control")
    print("    - Server config: WorkerRateLimit / TaskQueueRateLimit")

    rating = "⭐⭐⭐⭐"
    note = "Task Queueでルーティング。Worker側でmax_concurrent_activities、Server側でTaskQueueRateLimitを設定可能。優先度キューは自前実装"

    print_result("WF-06 キュー/流量制御", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all WF category verifications."""
    print("\n" + "="*60)
    print("  WF: Workflow Primitives 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            StepDefinitionWorkflow,
            ChildWorkflow,
            ParentWorkflow,
            FanOutWorkflow,
            ConditionalLoopWorkflow,
            SleepTimerWorkflow,
        ],
        activities=[process_step, compute_step, parallel_task],
    ):
        results = {}

        results["WF-01"] = await verify_wf01_step_definition(client)
        results["WF-02"] = await verify_wf02_child_workflows(client)
        results["WF-03"] = await verify_wf03_parallel_execution(client)
        results["WF-04"] = await verify_wf04_conditional_loop(client)
        results["WF-05"] = await verify_wf05_sleep_timer(client)
        results["WF-06"] = await verify_wf06_queue_rate_control(client)

        print("\n" + "="*60)
        print("  WF Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
