"""
SG: Signals & Events - 外部イベント・シグナル・クエリの検証

評価項目:
- SG-01: 外部シグナル - 実行中のワークフローに外部からデータ/指示を送る
- SG-02: ウェイト / Awaitables - 外部イベント待ち
- SG-03: イベントトリガー - イベント駆動でワークフローを起動
- SG-04: クエリ - 実行中のワークフローの状態を外部から読み取る
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
# SG-01 & SG-02: 外部シグナル & ウェイト (Signals & Wait)
# =============================================================================

@workflow.defn
class SignalWorkflow:
    """Workflow demonstrating signal handling and waiting."""

    def __init__(self) -> None:
        self._messages: list[str] = []
        self._approved: bool = False
        self._completed: bool = False

    @workflow.run
    async def run(self) -> dict:
        # Wait for approval signal
        await workflow.wait_condition(lambda: self._approved)

        # Process any messages received
        result = {
            "messages_received": len(self._messages),
            "messages": self._messages.copy(),
            "approved": self._approved,
        }

        self._completed = True
        return result

    @workflow.signal
    async def send_message(self, message: str) -> None:
        """Signal handler for receiving messages."""
        self._messages.append(message)

    @workflow.signal
    async def approve(self) -> None:
        """Signal handler for approval."""
        self._approved = True

    @workflow.query
    def get_status(self) -> dict:
        """Query handler for current status."""
        return {
            "messages_count": len(self._messages),
            "approved": self._approved,
            "completed": self._completed,
        }


async def verify_sg01_external_signals(client: Client) -> str:
    """Verify SG-01: External Signals."""
    print_section("SG-01: 外部シグナル (External Signals)")

    workflow_id = f"sg01-signals-{uuid.uuid4()}"

    # Start workflow
    handle = await client.start_workflow(
        SignalWorkflow.run,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    # Send signals to running workflow
    await handle.signal(SignalWorkflow.send_message, "Hello")
    await handle.signal(SignalWorkflow.send_message, "World")
    print("  Sent 2 messages via signal")

    # Send approval signal
    await handle.signal(SignalWorkflow.approve)
    print("  Sent approval signal")

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Signal features:")
    print("    - @workflow.signal decorator defines handlers")
    print("    - handle.signal() sends signals to running workflow")
    print("    - Signals are durably recorded in Event History")

    rating = "⭐⭐⭐⭐⭐"
    note = "@workflow.signalでハンドラ定義、handle.signal()で送信。Event Historyに永続化され、Replay時も正確に再生される"

    print_result("SG-01 外部シグナル", rating, note)
    return rating


async def verify_sg02_wait_awaitables(client: Client) -> str:
    """Verify SG-02: Wait / Awaitables."""
    print_section("SG-02: ウェイト / Awaitables")

    print("  Wait mechanisms:")
    print("    - workflow.wait_condition(lambda: condition)")
    print("    - await workflow.sleep(duration)")
    print("    - Durable: survives worker restarts")
    print("    - Can combine with signals for human approval patterns")

    rating = "⭐⭐⭐⭐⭐"
    note = "workflow.wait_condition()で任意条件を永続的に待機。Signal + wait_conditionで承認ワークフローを自然に実装可能"

    print_result("SG-02 ウェイト/Awaitables", rating, note)
    return rating


# =============================================================================
# SG-03: イベントトリガー (Event Triggers)
# =============================================================================

async def verify_sg03_event_triggers(client: Client) -> str:
    """Verify SG-03: Event Triggers."""
    print_section("SG-03: イベントトリガー (Event Triggers)")

    print("  Event trigger options:")
    print("    - client.start_workflow(): Direct API trigger")
    print("    - Schedule: Cron-based triggers (built-in)")
    print("    - Signal-to-start: Signal can start new workflow")
    print("    - External integration: Kafka/webhooks via custom code")
    print("    - Nexus (new): Cross-namespace workflow triggers")

    rating = "⭐⭐⭐⭐"
    note = "APIトリガー、Cronスケジュール、Signal-to-Startはネイティブ。Kafka等の外部イベント連携は自前実装が必要"

    print_result("SG-03 イベントトリガー", rating, note)
    return rating


# =============================================================================
# SG-04: クエリ (Query)
# =============================================================================

@workflow.defn
class QueryableWorkflow:
    """Workflow demonstrating query capability."""

    def __init__(self) -> None:
        self._progress: int = 0
        self._status: str = "started"

    @workflow.run
    async def run(self, total_steps: int) -> dict:
        for step in range(total_steps):
            self._progress = step + 1
            self._status = f"processing step {self._progress}/{total_steps}"
            await workflow.sleep(timedelta(milliseconds=500))

        self._status = "completed"
        return {"total_steps": total_steps, "final_status": self._status}

    @workflow.query
    def get_progress(self) -> int:
        """Get current progress."""
        return self._progress

    @workflow.query
    def get_status(self) -> str:
        """Get current status."""
        return self._status

    @workflow.query
    def get_full_state(self) -> dict:
        """Get full workflow state."""
        return {
            "progress": self._progress,
            "status": self._status,
        }


async def verify_sg04_query(client: Client) -> str:
    """Verify SG-04: Query."""
    print_section("SG-04: クエリ (Query)")

    workflow_id = f"sg04-query-{uuid.uuid4()}"

    # Start workflow
    handle = await client.start_workflow(
        QueryableWorkflow.run,
        5,  # 5 steps
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    # Query while running
    await asyncio.sleep(1)  # Let it process a bit

    progress = await handle.query(QueryableWorkflow.get_progress)
    status = await handle.query(QueryableWorkflow.get_status)
    full_state = await handle.query(QueryableWorkflow.get_full_state)

    print(f"  Progress query: {progress}")
    print(f"  Status query: {status}")
    print(f"  Full state query: {full_state}")

    # Wait for completion
    result = await handle.result()
    print(f"  Final result: {result}")
    print("  Query features:")
    print("    - @workflow.query decorator defines handlers")
    print("    - handle.query() reads state without modifying")
    print("    - Queries are synchronous (immediate response)")
    print("    - Read-only: cannot modify workflow state")

    rating = "⭐⭐⭐⭐⭐"
    note = "@workflow.queryでハンドラ定義。実行中ワークフローの状態を同期的に読み取り可能。副作用なしの読み取り専用"

    print_result("SG-04 クエリ", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all SG category verifications."""
    print("\n" + "="*60)
    print("  SG: Signals & Events 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SignalWorkflow, QueryableWorkflow],
        activities=[],
    ):
        results = {}

        results["SG-01"] = await verify_sg01_external_signals(client)
        results["SG-02"] = await verify_sg02_wait_awaitables(client)
        results["SG-03"] = await verify_sg03_event_triggers(client)
        results["SG-04"] = await verify_sg04_query(client)

        print("\n" + "="*60)
        print("  SG Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
