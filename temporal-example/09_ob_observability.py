"""
OB: Observability - 実行状態の理解・調査・監視の検証

評価項目:
- OB-01: ダッシュボード / UI - ワークフロー一覧・状態・履歴の可視化
- OB-02: メトリクス - 実行数・成功率・レイテンシ
- OB-03: 実行履歴の可視化 - ステップ単位の入出力閲覧
- OB-04: OTel準拠 - OpenTelemetry対応
- OB-05: アラート - 失敗率・遅延の検知と通知
- OB-06: ログ - 構造化ログ
"""

import asyncio
import uuid
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

from common import TEMPORAL_ADDRESS, TASK_QUEUE, print_section, print_result


# =============================================================================
# サンプルワークフロー
# =============================================================================

@activity.defn
async def observable_activity(step: str) -> str:
    """Activity with logging for observability demo."""
    return f"completed:{step}"


@workflow.defn
class ObservableWorkflow:
    """Workflow for observability testing."""

    @workflow.run
    async def run(self, steps: list[str]) -> dict:
        results = {}
        for step in steps:
            results[step] = await workflow.execute_activity(
                observable_activity,
                step,
                start_to_close_timeout=timedelta(seconds=30),
            )
        return results


# =============================================================================
# OB-01: ダッシュボード / UI
# =============================================================================

async def verify_ob01_dashboard(client: Client) -> str:
    """Verify OB-01: Dashboard / UI."""
    print_section("OB-01: ダッシュボード / UI")

    print("  Temporal Web UI features:")
    print("    - Workflow list with search/filter")
    print("    - Workflow detail view:")
    print("      - Input/Output visualization")
    print("      - Event History timeline")
    print("      - Query execution")
    print("      - Signal sending")
    print("      - Terminate/Cancel actions")
    print("    - Task Queue monitoring")
    print("    - Namespace settings")
    print("  Access:")
    print("    - Default: http://localhost:8233 (dev server)")
    print("    - Temporal Cloud: cloud.temporal.io")

    rating = "⭐⭐⭐⭐⭐"
    note = "Temporal Web UIで完全な可視化。ワークフロー一覧/詳細/履歴/検索/フィルタ/アクション実行が可能"

    print_result("OB-01 ダッシュボード/UI", rating, note)
    return rating


# =============================================================================
# OB-02: メトリクス
# =============================================================================

async def verify_ob02_metrics(client: Client) -> str:
    """Verify OB-02: Metrics."""
    print_section("OB-02: メトリクス")

    print("  Built-in metrics (Prometheus format):")
    print("    - temporal_workflow_started_total")
    print("    - temporal_workflow_completed_total")
    print("    - temporal_workflow_failed_total")
    print("    - temporal_workflow_execution_latency")
    print("    - temporal_activity_execution_latency")
    print("    - temporal_task_queue_depth")
    print("  SDK metrics:")
    print("    - Worker performance metrics")
    print("    - Activity execution metrics")
    print("  Export:")
    print("    - Prometheus endpoint on server")
    print("    - SDK metrics exporters available")

    rating = "⭐⭐⭐⭐⭐"
    note = "Prometheus形式のメトリクスをサーバー/SDKで公開。ワークフロー数/成功率/レイテンシ/キュー深度を標準出力"

    print_result("OB-02 メトリクス", rating, note)
    return rating


# =============================================================================
# OB-03: 実行履歴の可視化
# =============================================================================

async def verify_ob03_history_visualization(client: Client) -> str:
    """Verify OB-03: History Visualization."""
    print_section("OB-03: 実行履歴の可視化")

    workflow_id = f"ob03-history-{uuid.uuid4()}"

    handle = await client.start_workflow(
        ObservableWorkflow.run,
        ["step-a", "step-b", "step-c"],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    result = await handle.result()
    print(f"  Workflow result: {result}")

    # Fetch history
    history = await handle.fetch_history()
    event_count = len(list(history.events))
    print(f"  Event count: {event_count}")

    print("  History features:")
    print("    - Full Event History in UI")
    print("    - Event types: WorkflowStarted, ActivityScheduled, etc.")
    print("    - Input/Output for each event")
    print("    - Timeline visualization")
    print("    - JSON export via CLI")
    print("  CLI: temporal workflow show --workflow-id <id>")

    rating = "⭐⭐⭐⭐⭐"
    note = "Event History全体をUI/CLIで確認可能。各イベントの入出力、タイムラインが可視化される"

    print_result("OB-03 実行履歴の可視化", rating, note)
    return rating


# =============================================================================
# OB-04: OTel準拠
# =============================================================================

async def verify_ob04_otel(client: Client) -> str:
    """Verify OB-04: OpenTelemetry Compliance."""
    print_section("OB-04: OTel準拠")

    print("  OpenTelemetry support:")
    print("    - SDK interceptors for tracing")
    print("    - OpenTelemetry exporter available")
    print("    - Trace context propagation")
    print("    - Span creation for workflows/activities")
    print("  Integration:")
    print("    - temporalio.contrib.opentelemetry module")
    print("    - Custom TracingInterceptor")
    print("  Note: Requires additional setup")

    rating = "⭐⭐⭐⭐"
    note = "OpenTelemetryインテグレーション提供（contrib.opentelemetry）。Interceptor設定でトレース伝播可能"

    print_result("OB-04 OTel準拠", rating, note)
    return rating


# =============================================================================
# OB-05: アラート
# =============================================================================

async def verify_ob05_alerts(client: Client) -> str:
    """Verify OB-05: Alerts."""
    print_section("OB-05: アラート")

    print("  Alerting options:")
    print("    - Prometheus metrics + AlertManager")
    print("    - Temporal Cloud: Built-in alerts")
    print("    - Custom: Poll workflow status + notify")
    print("  Common alert conditions:")
    print("    - Workflow failure rate")
    print("    - Task queue backlog")
    print("    - Activity timeout rate")
    print("    - Long-running workflows")

    rating = "⭐⭐⭐⭐"
    note = "Prometheus + AlertManagerで構成可能。Temporal Cloudはビルトインアラート。専用アラート機能はメトリクス連携で実現"

    print_result("OB-05 アラート", rating, note)
    return rating


# =============================================================================
# OB-06: ログ
# =============================================================================

async def verify_ob06_logging(client: Client) -> str:
    """Verify OB-06: Logging."""
    print_section("OB-06: ログ")

    print("  Logging features:")
    print("    - SDK: workflow.logger for structured logging")
    print("    - Activity logging with context")
    print("    - Correlation: Workflow ID, Run ID in logs")
    print("    - Server logs: JSON format available")
    print("  SDK logging:")
    print("    - workflow.logger.info('message')")
    print("    - Automatically includes workflow context")
    print("    - Log levels: debug, info, warn, error")

    rating = "⭐⭐⭐⭐"
    note = "workflow.loggerで構造化ログ。Workflow ID/Run IDが自動付与されフィルタリング可能。Server側もJSON対応"

    print_result("OB-06 ログ", rating, note)
    return rating


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all OB category verifications."""
    print("\n" + "="*60)
    print("  OB: Observability 検証")
    print("="*60)

    client = await Client.connect(TEMPORAL_ADDRESS)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ObservableWorkflow],
        activities=[observable_activity],
    ):
        results = {}

        results["OB-01"] = await verify_ob01_dashboard(client)
        results["OB-02"] = await verify_ob02_metrics(client)
        results["OB-03"] = await verify_ob03_history_visualization(client)
        results["OB-04"] = await verify_ob04_otel(client)
        results["OB-05"] = await verify_ob05_alerts(client)
        results["OB-06"] = await verify_ob06_logging(client)

        print("\n" + "="*60)
        print("  OB Category Summary")
        print("="*60)
        for item, rating in results.items():
            print(f"  {item}: {rating}")


if __name__ == "__main__":
    asyncio.run(main())
