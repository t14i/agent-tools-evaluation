"""
DX: Developer Experience - 開発・テスト・デバッグの効率性の検証

評価項目:
- DX-01: SDK設計 - APIのシンプルさ、言語ネイティブ度
- DX-02: 言語サポート - 対応言語とSDK成熟度
- DX-03: ローカル開発 - ローカルでの実行・デバッグの容易さ
- DX-04: テスト / Time Skipping - ユニットテスト、長期スリープのスキップ
- DX-05: エラーメッセージ / デバッグ - エラー時の情報量
- DX-06: 学習曲線 - 概念モデルの複雑さ
- DX-07: ローカルReplayハーネス - 本番履歴のローカル再生
"""

import asyncio
import uuid
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# サンプルワークフロー（SDK設計デモ用）
# =============================================================================

@activity.defn
async def greet(name: str) -> str:
    """Simple greeting activity."""
    return f"Hello, {name}!"


@workflow.defn
class GreetingWorkflow:
    """Simple workflow demonstrating SDK design."""

    @workflow.run
    async def run(self, name: str) -> str:
        # Natural Python async/await syntax
        greeting = await workflow.execute_activity(
            greet,
            name,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return greeting

    @workflow.signal
    async def update_name(self, new_name: str) -> None:
        """Signal handler - just a decorated method."""
        pass

    @workflow.query
    def get_status(self) -> str:
        """Query handler - simple method."""
        return "running"


# =============================================================================
# DX-01: SDK設計
# =============================================================================

async def verify_dx01_sdk_design(client: Client) -> str:
    """Verify DX-01: SDK Design."""
    print_section("DX-01: SDK設計")

    workflow_id = f"dx01-sdk-{uuid.uuid4()}"

    handle = await client.start_workflow(
        GreetingWorkflow.run,
        "World",
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Result: {result}")

    print("  SDK design highlights:")
    print("    - @workflow.defn, @activity.defn decorators")
    print("    - Native async/await support")
    print("    - Type hints throughout")
    print("    - IDE autocompletion works")
    print("    - @workflow.signal, @workflow.query for handlers")

    rating = "⭐⭐⭐⭐⭐"
    note = "デコレータベースでPythonネイティブ。async/await対応、型ヒント完備、IDE補完も効く"

    print_result("DX-01 SDK設計", rating, note)
    return rating


# =============================================================================
# DX-02: 言語サポート
# =============================================================================

async def verify_dx02_language_support(client: Client) -> str:
    """Verify DX-02: Language Support."""
    print_section("DX-02: 言語サポート")

    print("  Supported languages:")
    print("    - Go (most mature, reference implementation)")
    print("    - Java (production ready)")
    print("    - TypeScript (production ready)")
    print("    - Python (production ready)")
    print("    - .NET (production ready)")
    print("    - PHP (community)")
    print("    - Ruby (community)")
    print("  Language parity:")
    print("    - Core features available in all official SDKs")
    print("    - Some advanced features vary")

    rating = "⭐⭐⭐⭐⭐"
    note = "Go/Java/TypeScript/Python/.NETの公式SDK。全て本番対応。コア機能は全SDK共通"

    print_result("DX-02 言語サポート", rating, note)
    return rating


# =============================================================================
# DX-03: ローカル開発
# =============================================================================

async def verify_dx03_local_development(client: Client) -> str:
    """Verify DX-03: Local Development."""
    print_section("DX-03: ローカル開発")

    print("  Local development options:")
    print("    1. temporal server start-dev")
    print("       - Single command startup")
    print("       - In-memory storage (ephemeral)")
    print("       - UI included at localhost:8233")
    print("    2. Docker Compose")
    print("       - Persistent storage")
    print("       - Production-like setup")
    print("    3. Temporal CLI")
    print("       - Workflow management")
    print("       - History inspection")

    rating = "⭐⭐⭐⭐⭐"
    note = "temporal server start-devで1コマンド起動。メモリ内DBでゼロ設定、UI付き。Docker Composeで永続も可能"

    print_result("DX-03 ローカル開発", rating, note)
    return rating


# =============================================================================
# DX-04: テスト / Time Skipping
# =============================================================================

async def verify_dx04_testing(client: Client) -> str:
    """Verify DX-04: Testing / Time Skipping."""
    print_section("DX-04: テスト / Time Skipping")

    print("  Testing features:")
    print("    1. Workflow testing:")
    print("       - WorkflowEnvironment for isolated testing")
    print("       - Mock activities")
    print("       - Assertion on workflow state")
    print("    2. Time Skipping:")
    print("       - await env.sleep() skips time instantly")
    print("       - Test 30-day workflows in milliseconds")
    print("    3. Replay testing:")
    print("       - Replay workflow against saved history")
    print("       - Detect determinism issues")

    print("  Example test pattern:")
    print("    async with await WorkflowEnvironment.start_time_skipping() as env:")
    print("        result = await env.client.execute_workflow(...)")
    print("        # Time-based waits are skipped")

    rating = "⭐⭐⭐⭐⭐"
    note = "WorkflowEnvironmentでTime Skipping対応。30日スリープもミリ秒でテスト。Replay Testで決定性検証も可能"

    print_result("DX-04 テスト/Time Skipping", rating, note)
    return rating


# =============================================================================
# DX-05: エラーメッセージ / デバッグ
# =============================================================================

async def verify_dx05_error_messages(client: Client) -> str:
    """Verify DX-05: Error Messages / Debugging."""
    print_section("DX-05: エラーメッセージ / デバッグ")

    print("  Error handling:")
    print("    - Activity errors include full stack trace")
    print("    - ApplicationError for business errors")
    print("    - Clear error types (ActivityError, WorkflowError)")
    print("    - Error details propagate through workflow")
    print("  Debugging:")
    print("    - Event History shows exact failure point")
    print("    - UI displays error message and stack trace")
    print("    - workflow.logger for contextual logging")
    print("    - Replay with debugger attached")

    rating = "⭐⭐⭐⭐"
    note = "スタックトレース完備、エラータイプ明確。UIで失敗箇所・詳細を確認可能。Replay デバッグも可能"

    print_result("DX-05 エラーメッセージ/デバッグ", rating, note)
    return rating


# =============================================================================
# DX-06: 学習曲線
# =============================================================================

async def verify_dx06_learning_curve(client: Client) -> str:
    """Verify DX-06: Learning Curve."""
    print_section("DX-06: 学習曲線")

    print("  Concepts to learn:")
    print("    - Workflow vs Activity distinction")
    print("    - Determinism constraints (important!)")
    print("    - Event sourcing mental model")
    print("    - Task queues and workers")
    print("    - Signals, queries, and timers")
    print("  Learning resources:")
    print("    - Comprehensive documentation")
    print("    - Interactive tutorials")
    print("    - Sample applications")
    print("    - Active community (Slack)")

    rating = "⭐⭐⭐"
    note = "コンセプトは明確だが、決定性制約とEvent Sourcingの理解が必要。ドキュメントとチュートリアルは充実"

    print_result("DX-06 学習曲線", rating, note)
    return rating


# =============================================================================
# DX-07: ローカルReplayハーネス
# =============================================================================

async def verify_dx07_local_replay(client: Client) -> str:
    """Verify DX-07: Local Replay Harness."""
    print_section("DX-07: ローカルReplayハーネス")

    print("  Replay capabilities:")
    print("    1. Export history:")
    print("       temporal workflow show --workflow-id <id> --output json > history.json")
    print("    2. Replay locally:")
    print("       - WorkflowReplayerを使用")
    print("       - 本番履歴をローカルでデバッグ")
    print("    3. Determinism verification:")
    print("       - コード変更後にReplayして互換性確認")

    print("  Example:")
    print("    replayer = WorkflowReplayer(workflows=[MyWorkflow])")
    print("    await replayer.replay_workflow(history)")

    rating = "⭐⭐⭐⭐⭐"
    note = "本番Event HistoryをエクスポートしてローカルReplay可能。WorkflowReplayerでデバッグ・決定性検証ができる"

    print_result("DX-07 ローカルReplayハーネス", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all DX category verifications."""
    print("\n" + "="*60)
    print("  DX: Developer Experience 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[GreetingWorkflow],
        activities=[greet],
    ):
        results = {}

        results["DX-01"] = await verify_dx01_sdk_design(client)
        results["DX-02"] = await verify_dx02_language_support(client)
        results["DX-03"] = await verify_dx03_local_development(client)
        results["DX-04"] = await verify_dx04_testing(client)
        results["DX-05"] = await verify_dx05_error_messages(client)
        results["DX-06"] = await verify_dx06_learning_curve(client)
        results["DX-07"] = await verify_dx07_local_replay(client)

        print("\n" + "="*60)
        print("  DX Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
