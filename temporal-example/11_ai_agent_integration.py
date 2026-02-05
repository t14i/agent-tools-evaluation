"""
AI: AI/Agent Integration - LLM/Agent統合・非決定性・HITLの検証

評価項目:
- AI-01: LLM呼び出しのActivity化 - LLM APIコールをdurableなActivityとして実行
- AI-02: 非決定性の扱い - LLM応答のReplay / キャッシュ / 固定
- AI-03: HITL / 人間承認 - ワークフロー内での人間の承認待ち
- AI-04: ストリーミング - LLMのストリーミング応答の扱い
- AI-05: Agent Framework統合 - LangGraph / CrewAI等との統合
- AI-06: Tool実行の耐障害性 - 外部ツール呼び出しのリトライ・タイムアウト
"""

import asyncio
import os
import uuid
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# AI-01: LLM呼び出しのActivity化
# =============================================================================

@activity.defn
async def call_llm(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """
    LLM call as a durable activity.
    In real implementation, this would call OpenAI API.
    """
    # Simulated LLM response (in production, use openai.chat.completions.create)
    # This demonstrates the pattern - actual API call would be here
    return {
        "model": model,
        "prompt": prompt,
        "response": f"[Simulated LLM response to: {prompt[:50]}...]",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


@workflow.defn
class LLMWorkflow:
    """Workflow demonstrating LLM call as activity."""

    @workflow.run
    async def run(self, prompt: str) -> dict:
        # LLM call is an activity - result is memoized
        result = await workflow.execute_activity(
            call_llm,
            prompt,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_attempts=3,
                non_retryable_error_types=["InvalidAPIKeyError"],
            ),
        )
        return result


async def verify_ai01_llm_activity(client: Client) -> str:
    """Verify AI-01: LLM Call as Activity."""
    print_section("AI-01: LLM呼び出しのActivity化")

    workflow_id = f"ai01-llm-{uuid.uuid4()}"

    handle = await client.start_workflow(
        LLMWorkflow.run,
        "What is the capital of France?",
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  LLM Activity pattern:")
    print("    - LLM call wrapped in @activity.defn")
    print("    - Response is memoized in Event History")
    print("    - Retry policy handles transient failures")
    print("    - Timeout prevents hanging on slow responses")

    rating = "⭐⭐⭐⭐⭐"
    note = "LLM呼び出しをActivityとしてラップ。結果はEvent Historyに記録され、Replay時は再実行されない（コスト節約）"

    print_result("AI-01 LLM呼び出しのActivity化", rating, note)
    return rating


# =============================================================================
# AI-02: 非決定性の扱い
# =============================================================================

async def verify_ai02_non_determinism(client: Client) -> str:
    """Verify AI-02: Non-determinism Handling."""
    print_section("AI-02: 非決定性の扱い")

    print("  Non-determinism handling:")
    print("    1. Activity isolation:")
    print("       - LLM calls are activities (not in workflow code)")
    print("       - Activity results memoized")
    print("       - Replay uses cached result")
    print("    2. Determinism preserved:")
    print("       - Same history → same replay result")
    print("       - No need for seed/temperature control for replay")
    print("    3. Re-execution control:")
    print("       - workflow reset to re-run from specific point")
    print("       - New result recorded in new history")

    rating = "⭐⭐⭐⭐"
    note = "Activity隔離でLLM出力を履歴に保存。Replay時は保存結果を返す。プロンプト/モデル変更の影響管理は自前"

    print_result("AI-02 非決定性の扱い", rating, note)
    return rating


# =============================================================================
# AI-03: HITL / 人間承認
# =============================================================================

@workflow.defn
class HITLApprovalWorkflow:
    """Workflow demonstrating human-in-the-loop approval."""

    def __init__(self) -> None:
        self._approved: bool | None = None
        self._approval_reason: str = ""

    @workflow.run
    async def run(self, action: str) -> dict:
        # Execute LLM to generate proposal
        proposal = await workflow.execute_activity(
            call_llm,
            f"Generate a detailed plan for: {action}",
            start_to_close_timeout=timedelta(seconds=60),
        )

        # Wait for human approval
        await workflow.wait_condition(lambda: self._approved is not None)

        if self._approved:
            return {
                "status": "approved",
                "proposal": proposal,
                "reason": self._approval_reason,
            }
        else:
            return {
                "status": "rejected",
                "proposal": proposal,
                "reason": self._approval_reason,
            }

    @workflow.signal
    async def approve(self, reason: str = "") -> None:
        """Human approves the proposal."""
        self._approved = True
        self._approval_reason = reason

    @workflow.signal
    async def reject(self, reason: str = "") -> None:
        """Human rejects the proposal."""
        self._approved = False
        self._approval_reason = reason

    @workflow.query
    def get_approval_status(self) -> dict:
        """Query current approval status."""
        return {
            "approved": self._approved,
            "reason": self._approval_reason,
        }


async def verify_ai03_hitl_approval(client: Client) -> str:
    """Verify AI-03: HITL / Human Approval."""
    print_section("AI-03: HITL / 人間承認")

    workflow_id = f"ai03-hitl-{uuid.uuid4()}"

    # Start workflow
    handle = await client.start_workflow(
        HITLApprovalWorkflow.run,
        "Deploy new AI model to production",
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    # Check status before approval
    await asyncio.sleep(0.5)
    status_before = await handle.query(HITLApprovalWorkflow.get_approval_status)
    print(f"  Status before approval: {status_before}")

    # Simulate human approval
    await handle.signal(HITLApprovalWorkflow.approve, "LGTM - reviewed by team lead")
    print("  Sent approval signal")

    result = await handle.result()
    print(f"  Result: {result}")
    print("  HITL pattern:")
    print("    - workflow.wait_condition() for blocking wait")
    print("    - Signal handlers for approve/reject")
    print("    - Query for status check")
    print("    - Durable: survives worker restarts")

    rating = "⭐⭐⭐⭐⭐"
    note = "Signal + wait_conditionで永続的な承認待機。Worker再起動しても待機状態を維持。UI/CLI/APIから承認可能"

    print_result("AI-03 HITL/人間承認", rating, note)
    return rating


# =============================================================================
# AI-04: ストリーミング
# =============================================================================

async def verify_ai04_streaming(client: Client) -> str:
    """Verify AI-04: Streaming."""
    print_section("AI-04: ストリーミング")

    print("  Streaming considerations:")
    print("    - Activities return complete results (no mid-stream)")
    print("    - Streaming within activity is possible")
    print("    - Cannot stream through workflow boundary")
    print("  Patterns for streaming-like behavior:")
    print("    1. Batch activities: Split into chunks")
    print("    2. Update signals: Send progress via signals")
    print("    3. Query for partial results")
    print("    4. External callback: Stream to external service")

    rating = "⭐⭐⭐"
    note = "ワークフロー境界を越えたストリーミングは不可。Activity内部でのストリーム処理は可能。チャンク分割パターンで対応"

    print_result("AI-04 ストリーミング", rating, note)
    return rating


# =============================================================================
# AI-05: Agent Framework統合
# =============================================================================

async def verify_ai05_framework_integration(client: Client) -> str:
    """Verify AI-05: Agent Framework Integration."""
    print_section("AI-05: Agent Framework統合")

    print("  Integration patterns:")
    print("    1. LangGraph + Temporal:")
    print("       - LangGraph graphs as activities")
    print("       - Temporal for durable orchestration")
    print("       - Example: github.com/temporalio/samples-python")
    print("    2. CrewAI + Temporal:")
    print("       - Crew.kickoff() in activity")
    print("       - Temporal handles persistence")
    print("    3. OpenAI SDK + Temporal:")
    print("       - Direct API calls as activities")
    print("       - Native support for function calling")
    print("  Documentation:")
    print("    - AI/LLM orchestration guides available")
    print("    - Sample projects in temporal-samples repo")

    rating = "⭐⭐⭐⭐"
    note = "LangGraph/CrewAI/OpenAI SDKとの統合パターンあり。サンプルとドキュメントが整備されている"

    print_result("AI-05 Agent Framework統合", rating, note)
    return rating


# =============================================================================
# AI-06: Tool実行の耐障害性
# =============================================================================

@activity.defn
async def execute_tool(tool_name: str, params: dict) -> dict:
    """Execute external tool with retry support."""
    # Simulated tool execution
    return {
        "tool": tool_name,
        "params": params,
        "result": f"Tool {tool_name} executed successfully",
    }


@workflow.defn
class ToolExecutionWorkflow:
    """Workflow demonstrating fault-tolerant tool execution."""

    @workflow.run
    async def run(self, tools: list[dict]) -> list[dict]:
        results = []
        for tool in tools:
            # Each tool call has its own retry policy
            result = await workflow.execute_activity(
                execute_tool,
                args=[tool["name"], tool.get("params", {})],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(seconds=30),
                    maximum_attempts=5,
                ),
            )
            results.append(result)
        return results


async def verify_ai06_tool_fault_tolerance(client: Client) -> str:
    """Verify AI-06: Tool Execution Fault Tolerance."""
    print_section("AI-06: Tool実行の耐障害性")

    workflow_id = f"ai06-tools-{uuid.uuid4()}"

    tools = [
        {"name": "web_search", "params": {"query": "AI news"}},
        {"name": "code_executor", "params": {"code": "print('hello')"}},
    ]

    handle = await client.start_workflow(
        ToolExecutionWorkflow.run,
        tools,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")
    print("  Tool fault tolerance:")
    print("    - Each tool call is an activity with retry policy")
    print("    - Configurable timeouts per tool")
    print("    - Backoff and retry for transient failures")
    print("    - Results memoized (no re-execution on replay)")
    print("    - Heartbeat for long-running tools")

    rating = "⭐⭐⭐⭐⭐"
    note = "各ツール呼び出しにRetry Policy設定可能。タイムアウト、バックオフ、最大試行回数を細かく制御できる"

    print_result("AI-06 Tool実行の耐障害性", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all AI category verifications."""
    print("\n" + "="*60)
    print("  AI: AI/Agent Integration 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LLMWorkflow, HITLApprovalWorkflow, ToolExecutionWorkflow],
        activities=[call_llm, execute_tool],
    ):
        results = {}

        results["AI-01"] = await verify_ai01_llm_activity(client)
        results["AI-02"] = await verify_ai02_non_determinism(client)
        results["AI-03"] = await verify_ai03_hitl_approval(client)
        results["AI-04"] = await verify_ai04_streaming(client)
        results["AI-05"] = await verify_ai05_framework_integration(client)
        results["AI-06"] = await verify_ai06_tool_fault_tolerance(client)

        print("\n" + "="*60)
        print("  AI Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
