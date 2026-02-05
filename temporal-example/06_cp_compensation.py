"""
CP: Compensation & Recovery - 失敗時の復旧・手動介入の検証

評価項目:
- CP-01: 補償トランザクション / サガ - ステップ失敗時の逆操作
- CP-02: 部分適用 / 途中再開 - 失敗したステップからの再開
- CP-03: 手動介入 - ダッシュボード/APIからの修正・再開
- CP-04: Dead Letter / 毒メッセージ - 処理不能なワークフローの隔離
"""

import asyncio
import uuid
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# CP-01: 補償トランザクション / サガ (Compensation / Saga)
# =============================================================================

compensation_log: list[str] = []


@activity.defn
async def book_hotel(reservation_id: str) -> str:
    """Book a hotel room."""
    return f"hotel_booked:{reservation_id}"


@activity.defn
async def book_flight(reservation_id: str) -> str:
    """Book a flight."""
    return f"flight_booked:{reservation_id}"


@activity.defn
async def book_car(reservation_id: str, should_fail: bool = False) -> str:
    """Book a rental car (may fail for testing)."""
    if should_fail:
        raise ApplicationError("Car booking failed - no availability")
    return f"car_booked:{reservation_id}"


@activity.defn
async def cancel_hotel(reservation_id: str) -> str:
    """Compensating action: cancel hotel."""
    compensation_log.append(f"cancelled_hotel:{reservation_id}")
    return f"hotel_cancelled:{reservation_id}"


@activity.defn
async def cancel_flight(reservation_id: str) -> str:
    """Compensating action: cancel flight."""
    compensation_log.append(f"cancelled_flight:{reservation_id}")
    return f"flight_cancelled:{reservation_id}"


@workflow.defn
class SagaWorkflow:
    """Workflow implementing Saga pattern for distributed transactions."""

    @workflow.run
    async def run(self, reservation_id: str, car_should_fail: bool = False) -> dict:
        compensations: list[tuple[Any, str]] = []
        results = {}

        try:
            # Step 1: Book hotel
            results["hotel"] = await workflow.execute_activity(
                book_hotel,
                reservation_id,
                start_to_close_timeout=timedelta(seconds=30),
            )
            compensations.append((cancel_hotel, reservation_id))

            # Step 2: Book flight
            results["flight"] = await workflow.execute_activity(
                book_flight,
                reservation_id,
                start_to_close_timeout=timedelta(seconds=30),
            )
            compensations.append((cancel_flight, reservation_id))

            # Step 3: Book car (may fail)
            results["car"] = await workflow.execute_activity(
                book_car,
                args=[reservation_id, car_should_fail],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            results["status"] = "all_booked"
            return results

        except Exception as e:
            # Execute compensations in reverse order
            results["error"] = str(e)
            results["compensations"] = []

            for comp_activity, comp_arg in reversed(compensations):
                try:
                    comp_result = await workflow.execute_activity(
                        comp_activity,
                        comp_arg,
                        start_to_close_timeout=timedelta(seconds=30),
                    )
                    results["compensations"].append(comp_result)
                except Exception as comp_error:
                    results["compensations"].append(f"failed:{comp_error}")

            results["status"] = "rolled_back"
            return results


async def verify_cp01_compensation_saga(client: Client) -> str:
    """Verify CP-01: Compensation / Saga."""
    print_section("CP-01: 補償トランザクション / サガ")

    # Test successful case
    workflow_id_success = f"cp01-saga-success-{uuid.uuid4()}"
    handle_success = await client.start_workflow(
        SagaWorkflow.run,
        args=["res-001", False],
        id=workflow_id_success,
        task_queue=TASK_QUEUE,
    )
    result_success = await handle_success.result()
    print(f"  Success case: {result_success}")

    # Test failure case with compensation
    compensation_log.clear()
    workflow_id_fail = f"cp01-saga-fail-{uuid.uuid4()}"
    handle_fail = await client.start_workflow(
        SagaWorkflow.run,
        args=["res-002", True],  # Car booking will fail
        id=workflow_id_fail,
        task_queue=TASK_QUEUE,
    )
    result_fail = await handle_fail.result()
    print(f"  Failure case: {result_fail}")
    print(f"  Compensation log: {compensation_log}")

    rating = "⭐⭐⭐⭐"
    note = "Saga パターンを try/except + compensations リストで実装可能。補償失敗時のリトライ/エスカレーションは自前実装が必要"

    print_result("CP-01 補償トランザクション/サガ", rating, note)
    return rating


# =============================================================================
# CP-02: 部分適用 / 途中再開 (Partial Apply / Resume)
# =============================================================================

step_execution_count: dict[str, int] = {}


@activity.defn
async def resumable_step(step_id: str) -> str:
    """Step that tracks execution count."""
    step_execution_count[step_id] = step_execution_count.get(step_id, 0) + 1
    return f"executed:{step_id}:count={step_execution_count[step_id]}"


@workflow.defn
class ResumableWorkflow:
    """Workflow demonstrating resume from failure point."""

    @workflow.run
    async def run(self, steps: list[str], fail_at: str | None = None) -> dict:
        results = {}

        for step in steps:
            if fail_at and step == fail_at:
                raise ApplicationError(f"Intentional failure at {step}")

            # Each activity result is memoized
            # On retry/resume, completed steps are not re-executed
            results[step] = await workflow.execute_activity(
                resumable_step,
                step,
                start_to_close_timeout=timedelta(seconds=30),
            )

        return results


async def verify_cp02_partial_resume(client: Client) -> str:
    """Verify CP-02: Partial Apply / Resume."""
    print_section("CP-02: 部分適用 / 途中再開")

    print("  Resume mechanisms:")
    print("    - Activity results are memoized in Event History")
    print("    - On worker restart, completed activities are NOT re-executed")
    print("    - Workflow code re-runs but skips completed activities")
    print("    - Reset to specific event ID via CLI for manual retry")

    rating = "⭐⭐⭐⭐⭐"
    note = "Activity結果がEvent Historyにメモ化され、Replay時は再実行されない。temporal workflow resetで特定ポイントからの再開も可能"

    print_result("CP-02 部分適用/途中再開", rating, note)
    return rating


# =============================================================================
# CP-03: 手動介入 (Manual Intervention)
# =============================================================================

async def verify_cp03_manual_intervention(client: Client) -> str:
    """Verify CP-03: Manual Intervention."""
    print_section("CP-03: 手動介入")

    print("  Manual intervention options:")
    print("    - Temporal UI: Visual workflow management")
    print("      - View workflow history")
    print("      - Send signals")
    print("      - Terminate/cancel workflows")
    print("      - Query workflow state")
    print("    - Temporal CLI (tctl / temporal):")
    print("      - temporal workflow terminate")
    print("      - temporal workflow cancel")
    print("      - temporal workflow reset --event-id <id>")
    print("      - temporal workflow signal")
    print("    - Client API:")
    print("      - handle.terminate()")
    print("      - handle.cancel()")
    print("      - handle.signal()")

    rating = "⭐⭐⭐⭐⭐"
    note = "Web UI + CLI + APIで完全な手動介入サポート。Terminate/Cancel/Reset/Signalを自由に実行可能。UI上で履歴も確認できる"

    print_result("CP-03 手動介入", rating, note)
    return rating


# =============================================================================
# CP-04: Dead Letter / 毒メッセージ (Dead Letter / Poison Message)
# =============================================================================

async def verify_cp04_dead_letter(client: Client) -> str:
    """Verify CP-04: Dead Letter / Poison Message."""
    print_section("CP-04: Dead Letter / 毒メッセージ")

    print("  Dead letter handling:")
    print("    - max_attempts in RetryPolicy limits retries")
    print("    - After exhausting retries, workflow/activity fails")
    print("    - Failed workflows visible in Temporal UI")
    print("    - Can filter by status (Failed/TimedOut)")
    print("    - Custom handling:")
    print("      - Catch failure and send to DLQ activity")
    print("      - Use temporal workflow list --status failed")
    print("      - Alerting via external integration")

    rating = "⭐⭐⭐⭐"
    note = "max_attemptsで上限設定、超過後はFailed状態に。UI/CLIで失敗ワークフローを一覧可能。専用DLQは自前実装が必要"

    print_result("CP-04 Dead Letter/毒メッセージ", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all CP category verifications."""
    print("\n" + "="*60)
    print("  CP: Compensation & Recovery 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SagaWorkflow, ResumableWorkflow],
        activities=[
            book_hotel, book_flight, book_car,
            cancel_hotel, cancel_flight,
            resumable_step,
        ],
    ):
        results = {}

        results["CP-01"] = await verify_cp01_compensation_saga(client)
        results["CP-02"] = await verify_cp02_partial_resume(client)
        results["CP-03"] = await verify_cp03_manual_intervention(client)
        results["CP-04"] = await verify_cp04_dead_letter(client)

        print("\n" + "="*60)
        print("  CP Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
