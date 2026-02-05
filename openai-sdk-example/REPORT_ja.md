# OpenAI Agents SDK 検証レポート

## 概要

本レポートは、OpenAI Agents SDK (openai-agents-python v0.8.0) の本番運用適合性を、エージェントフレームワーク評価基準（NIST AI RMF、WEF AI Agents in Action、IMDA Model AI Governance、OTel GenAI Semantic Conventions）に基づいて検証した結果をまとめたものである。

## テスト環境

- Python: 3.13
- OpenAI Agents SDK: 0.8.0
- OpenAI: 1.60.x
- Pydantic: 2.0.x

---

## 総評

### 建て付け・アーキテクチャ

OpenAI Agents SDKは、**LLMを中心としたエージェント構築のための高レベル抽象化フレームワーク**である。設計思想は「シンプルさ優先」で、LangGraphのような明示的なグラフ定義ではなく、**LLMに判断を委ねる宣言的アプローチ**を採用している。

```python
# 基本構造: Agent → Tool → Runner の3層
agent = Agent(
    name="MyAgent",
    instructions="...",
    tools=[tool1, tool2],
    handoffs=[handoff(other_agent)]  # マルチエージェントはこれだけ
)
result = Runner.run_sync(agent, "Hello")
```

**特徴的な設計判断:**
- **handoff()による委譲**: エージェント間の連携をLLMが自律判断。明示的なルーティングロジック不要
- **Pydanticネイティブ**: ツール引数は型ヒントから自動でJSONスキーマ生成
- **トレーシング標準装備**: デバッグ用ダッシュボードがデフォルトで有効

### カバー範囲

| 領域 | カバー度 | 詳細 |
|------|---------|------|
| ツール定義・実行 | ◎ | @function_tool、最大128ツール/エージェント、並列実行 |
| マルチエージェント | ◎ | ネイティブhandoff()、階層型・ルーティング型 |
| 状態永続化 | ○ | Sessions API（SQLite/SQLAlchemy/Dapr/Hosted） |
| メモリ | ○ | 会話メモリ、File Search（RAG）組み込み |
| HITL | ◎ | ネイティブneeds_approval API（v0.8.0）、承認/拒否/状態永続化 |
| 観測性 | ○ | トレーシング標準、Datadog/Langfuse連携 |
| ガバナンス | △ | Guardrails基盤のみ、ポリシーエンジンは自作 |
| 決定性・リプレイ | × | seedパラメータのみ、本格的なリプレイは自作 |
| 運用ガード | × | SLO/コストガード/キルスイッチは自作 |

### 使い勝手

**良い点:**

- **学習曲線が緩やか**: LangGraphの「ノード・エッジ・グラフコンパイル」に比べ、Agent/Tool/Runnerの3概念だけで始められる
- **マルチエージェントが簡単**: `handoff(other_agent)` の1行で委譲設定完了
- **型安全**: Pydantic統合でツール引数の型チェック・バリデーションが自動
- **デバッグしやすい**: トレーシングが標準で有効、OpenAIダッシュボードで実行履歴を確認可能

**悪い点:**

- **柔軟性の限界**: 複雑な条件分岐やサブグラフを作りたいときに表現力が足りない
- **LLM依存度が高い**: handoffの判断をLLMに委ねるため、意図しないルーティングが起きうる
- **エンタープライズ機能不足**: 監査証跡、PII除去、テナント分離、冪等性などは全て自作
- **OpenAIロックイン**: 他のLLMプロバイダへの切り替えが困難

---

## 星評価基準

| 星 | ラベル | 定義 | 判断基準 |
|----|--------|------|----------|
| ⭐ | 未対応 | 機能なし or 動作しない | ドキュメントなし、動かない、完全自作が必要 |
| ⭐⭐ | 実験的 | 動くが制約大、PoCでも苦労 | 動くが落とし穴多数、ドキュメント不足、API不安定 |
| ⭐⭐⭐ | PoC可 | 基本機能OK、デモには使えるが本番には追加作業必要 | 主要ケースは動く、エッジケースに弱い、監視・ログは自作 |
| ⭐⭐⭐⭐ | 本番可 | 実用的、軽微なカスタマイズで本番投入可 | 安定、ドキュメント充実、本番事例あり |
| ⭐⭐⭐⭐⭐ | 本番推奨 | そのまま本番利用可、ベストプラクティス確立 | 大規模本番事例あり、エコシステム成熟 |

---

## 前提パラメータ

| パラメータ | 値 |
|-----------|-----|
| 自律性 | 承認必須 |
| 権限 | 制限付き書き込み |
| 予測可能性 | LLM判断を含む |
| コンテキスト | 内部データ |

---

## カテゴリ別評価サマリー

### カバレッジサマリー（57項目）

| カテゴリ | 項目数 | Good (⭐⭐⭐+) | Not Good (⭐⭐-) | 備考 |
|----------|--------|---------------|-----------------|------|
| TC: ツール呼び出し | 5 | 5 | 0 | ツール定義優秀、Pydanticネイティブ |
| HI: 人間介入 | 5 | 4 | 1 | ネイティブneeds_approval API（v0.8.0）、タイムアウト/通知は自作 |
| DU: 永続的実行 | 6 | 4 | 2 | Sessions API堅牢、クリーンアップ/並行性は自作 |
| ME: メモリ | 8 | 4 | 4 | 基本メモリOK、エージェント自律管理なし |
| MA: マルチエージェント | 5 | 4 | 1 | ネイティブhandoffs優秀 |
| GV: ガバナンス | 6 | 2 | 4 | Guardrails利用可、ポリシー/監査は自作 |
| DR: 決定性・リプレイ | 6 | 1 | 5 | トレーシング部分的リプレイ、冪等性自作 |
| CX: コネクタ・運用 | 4 | 2 | 2 | Responses API良好、レート制限自作 |
| OB: 観測性 | 7 | 4 | 3 | 組み込みトレーシング優秀、OTel/SLO自作 |
| TE: テスト・評価 | 5 | 3 | 2 | モック注入可、シミュレーション自作 |
| **合計** | **57** | **33** | **24** | |

### フェイルクローズ項目ステータス

| 項目 | 評価 | 影響 | 適用対象 |
|------|------|------|----------|
| TE-01 ユニットテスト/モッキング | ⭐⭐⭐⭐ | **PASS** - model_settingsで注入可 | 全権限レベル |
| GV-01 破壊的操作ゲート | ⭐⭐⭐⭐ | **PASS** - Guardrails + 承認コールバック | 制限付き書き込み以上 |
| DR-01 リプレイ | ⭐⭐ | **BORDERLINE** - トレーシング部分的、LLMキャッシュなし | 制限付き書き込み以上 |
| DR-04 冪等性 | ⭐ | **FAIL** - ネイティブサポートなし | 完全書き込み |
| CX-02 レート制限/リトライ | ⭐⭐ | **BORDERLINE** - 自動リトライ一部あり、レート制限なし | 制限付き書き込み以上 |
| OB-01 トレース | ⭐⭐⭐⭐⭐ | **PASS** - 組み込みトレーシングがデフォルト有効 | 完全書き込み |
| OB-06 SLO/アラート | ⭐ | **FAIL** - ネイティブSLO管理なし | 完全書き込み |

> **フェイルクローズルール**: これらの項目のいずれかが⭐⭐以下の場合、他カテゴリに関わらず総合評価は⭐⭐が上限となる。
> TE-01は全権限レベルで必須。その他は書き込み権限に応じて適用。

---

## Good項目（評価⭐⭐⭐以上）

| カテゴリ | ID | 項目 | 評価 | 備考 |
|----------|-----|------|------|------|
| ツール呼び出し | TC-01 | ツール定義 | ⭐⭐⭐⭐⭐ | @function_toolデコレータ、自動Pydanticスキーマ |
| ツール呼び出し | TC-02 | 制御可能な自動化 | ⭐⭐⭐⭐ | human_input_callback、ツールラッピング |
| ツール呼び出し | TC-03 | 並列実行 | ⭐⭐⭐⭐⭐ | エージェントあたり最大128ツール |
| ツール呼び出し | TC-04 | エラーハンドリング | ⭐⭐⭐⭐ | 自動エラーキャッチ、LLMリカバリ |
| ツール呼び出し | TC-05 | 引数バリデーション | ⭐⭐⭐⭐⭐ | ネイティブPydantic統合 |
| 人間介入 | HI-01 | 中断API | ⭐⭐⭐⭐⭐ | ネイティブneeds_approval=True、result.interruptions |
| 人間介入 | HI-02 | 状態操作 | ⭐⭐⭐⭐ | RunState.to_json()/from_json()、完全な状態アクセス |
| 人間介入 | HI-03 | 再開制御 | ⭐⭐⭐⭐⭐ | state.approve()/reject()、選択的判断 |
| 永続的実行 | DU-01 | 状態永続化 | ⭐⭐⭐⭐ | Sessions API |
| 永続的実行 | DU-02 | プロセス再開 | ⭐⭐⭐⭐ | セッション復元 |
| 永続的実行 | DU-03 | HITL永続化 | ⭐⭐⭐ | Sessions + 状態シリアライズ |
| 永続的実行 | DU-04 | ストレージ選択肢 | ⭐⭐⭐⭐⭐ | SQLite/SQLAlchemy/Dapr/Hosted |
| メモリ | ME-01 | 短期メモリ | ⭐⭐⭐⭐ | Conversations API |
| メモリ | ME-02 | 長期メモリ | ⭐⭐⭐⭐ | Sessions + Storage |
| メモリ | ME-03 | セマンティック検索 | ⭐⭐⭐⭐ | File Search (RAG) 組み込み |
| メモリ | ME-06 | 自動抽出 | ⭐⭐⭐ | Context Summarization |
| マルチエージェント | MA-01 | 複数エージェント定義 | ⭐⭐⭐⭐⭐ | Agentクラス、クリーンAPI |
| マルチエージェント | MA-02 | 委譲 | ⭐⭐⭐⭐⭐ | ネイティブhandoff()関数 |
| マルチエージェント | MA-03 | 階層プロセス | ⭐⭐⭐⭐ | Agent-as-toolパターン |
| マルチエージェント | MA-04 | ルーティング | ⭐⭐⭐⭐ | Handoff条件、柔軟 |
| ガバナンス | GV-01 | 破壊的操作ゲート | ⭐⭐⭐⭐ | Guardrails + 承認 |
| ガバナンス | GV-03 | Policy as Code | ⭐⭐⭐ | Guardrailクラス |
| コネクタ・運用 | CX-03 | 非同期ジョブ | ⭐⭐⭐⭐ | Responses API background=true |
| 観測性 | OB-01 | トレース | ⭐⭐⭐⭐⭐ | 組み込み、デフォルト有効 |
| 観測性 | OB-02 | トークン消費 | ⭐⭐⭐⭐⭐ | request_usage_entries |
| 観測性 | OB-03 | ログ出力 | ⭐⭐⭐⭐ | トレーススパン |
| 観測性 | OB-04 | 外部連携 | ⭐⭐⭐⭐ | Datadog/Langfuse/Agenta |
| テスト・評価 | TE-01 | ユニットテスト/モッキング | ⭐⭐⭐⭐ | model_settings注入 |
| テスト・評価 | TE-02 | 状態注入 | ⭐⭐⭐ | セッション復元 |
| テスト・評価 | TE-05 | 評価フック | ⭐⭐⭐ | OpenAI Evals統合 |

---

## Not Good項目（評価⭐⭐以下）

| カテゴリ | ID | 項目 | 評価 | 備考 | 検証スクリプト |
|----------|-----|------|------|------|----------------|
| 人間介入 | HI-04 | タイムアウト | ⭐⭐ | カスタム実装が必要 | 06_hitl_state.py |
| 人間介入 | HI-05 | 通知 | ⭐ | 組み込み通知なし | 06_hitl_state.py |
| 永続的実行 | DU-05 | クリーンアップ（TTL） | ⭐ | 自動クリーンアップなし | 09_session_production.py |
| 永続的実行 | DU-06 | 並行アクセス | ⭐⭐ | カスタムロックが必要 | 09_session_production.py |
| メモリ | ME-04 | メモリAPI | ⭐⭐ | 限定的なCRUD API | 12_memory_context.py |
| メモリ | ME-05 | エージェント自律管理 | ⭐ | LangMem相当なし | 12_memory_context.py |
| メモリ | ME-07 | メモリクリーンアップ（TTL） | ⭐ | ネイティブTTLなし | 12_memory_context.py |
| メモリ | ME-08 | Embeddingコスト | ⭐⭐⭐ | token_usage利用可 | 12_memory_context.py |
| マルチエージェント | MA-05 | 共有メモリ | ⭐⭐⭐ | カスタム実装が必要 | 14_multiagent_orchestration.py |
| ガバナンス | GV-02 | 最小権限/スコープ | ⭐⭐ | ネイティブ権限システムなし | 15_governance_guardrails.py |
| ガバナンス | GV-04 | PII/除去 | ⭐ | ネイティブ除去なし | 16_governance_audit.py |
| ガバナンス | GV-05 | テナント/目的拘束 | ⭐ | ネイティブバインディングなし | 16_governance_audit.py |
| ガバナンス | GV-06 | 監査証跡完全性 | ⭐⭐⭐ | トレーシング部分的 | 16_governance_audit.py |
| 決定性・リプレイ | DR-01 | リプレイ | ⭐⭐ | トレーシング部分的、LLMキャッシュなし | 17_determinism_replay.py |
| 決定性・リプレイ | DR-02 | エビデンス参照 | ⭐⭐⭐ | トレーススパン利用可 | 17_determinism_replay.py |
| 決定性・リプレイ | DR-03 | 非決定性の隔離 | ⭐⭐ | seedパラメータ限定的 | 17_determinism_replay.py |
| 決定性・リプレイ | DR-04 | 冪等性 | ⭐ | ネイティブサポートなし | 18_determinism_recovery.py |
| 決定性・リプレイ | DR-05 | プラン差分 | ⭐ | ネイティブ差分なし | 18_determinism_recovery.py |
| 決定性・リプレイ | DR-06 | 障害復旧 | ⭐⭐ | Sessionsで部分的復旧可 | 18_determinism_recovery.py |
| コネクタ・運用 | CX-01 | 認証/クレデンシャル管理 | ⭐⭐⭐ | APIキーのみ | 19_connectors_streaming.py |
| コネクタ・運用 | CX-02 | レート制限/リトライ | ⭐⭐ | 自動リトライ一部、レート制限なし | 19_connectors_streaming.py |
| コネクタ・運用 | CX-04 | 状態マイグレーション | ⭐⭐ | マイグレーションサポートなし | 20_connectors_responses.py |
| 観測性 | OB-05 | OTel準拠 | ⭐⭐ | ネイティブOpenTelemetryなし | 22_observability_integration.py |
| 観測性 | OB-06 | SLO/アラート | ⭐ | ネイティブSLO管理なし | 23_observability_guard.py |
| 観測性 | OB-07 | コストガード | ⭐ | ネイティブ予算/キルスイッチなし | 23_observability_guard.py |
| テスト・評価 | TE-03 | シミュレーション/ユーザーエミュレーション | ⭐⭐ | ネイティブシミュレーションなし | 25_testing_evaluation.py |
| テスト・評価 | TE-04 | ドライラン/サンドボックスモード | ⭐⭐ | ネイティブドライランなし | 25_testing_evaluation.py |

---

## 検証スクリプト

| スクリプト | カテゴリ | 主要検証項目 |
|------------|----------|--------------|
| 01_quickstart.py | - | 基本的なAgent SDK構造 |
| 02_tool_definition.py | ツール呼び出し | TC-01: @function_tool、Pydantic |
| 03_tool_execution.py | ツール呼び出し | TC-03, TC-04, TC-05: 並列、エラー、バリデーション |
| 04_tool_control.py | ツール呼び出し | TC-02: 制御可能な自動化 |
| 05_hitl_approval.py | 人間介入 | HI-01, HI-03: 承認フロー |
| 06_hitl_state.py | 人間介入 | HI-02, HI-04, HI-05: 状態、タイムアウト、通知 |
| 07_session_basic.py | 永続的実行 | DU-01, DU-02: Sessions |
| 08_session_backends.py | 永続的実行 | DU-03, DU-04: ストレージバックエンド |
| 09_session_production.py | 永続的実行 | DU-05, DU-06: クリーンアップ、並行性 |
| 10_memory_conversation.py | メモリ | ME-01, ME-02: 会話メモリ |
| 11_memory_filesearch.py | メモリ | ME-03: File Search / RAG |
| 12_memory_context.py | メモリ | ME-04 ~ ME-08: コンテキスト管理 |
| 13_multiagent_handoff.py | マルチエージェント | MA-01, MA-02: ネイティブhandoffs |
| 14_multiagent_orchestration.py | マルチエージェント | MA-03, MA-04, MA-05: オーケストレーション |
| 15_governance_guardrails.py | ガバナンス | GV-01, GV-02, GV-03: Guardrails |
| 16_governance_audit.py | ガバナンス | GV-04, GV-05, GV-06: 監査証跡 |
| 17_determinism_replay.py | 決定性・リプレイ | DR-01, DR-02, DR-03: リプレイ |
| 18_determinism_recovery.py | 決定性・リプレイ | DR-04, DR-05, DR-06: 復旧 |
| 19_connectors_streaming.py | コネクタ・運用 | CX-01, CX-02: 認証、レート制限 |
| 20_connectors_responses.py | コネクタ・運用 | CX-03, CX-04: Responses API |
| 21_observability_tracing.py | 観測性 | OB-01, OB-02, OB-03: トレーシング |
| 22_observability_integration.py | 観測性 | OB-04, OB-05: 外部連携 |
| 23_observability_guard.py | 観測性 | OB-06, OB-07: SLO、コストガード |
| 24_testing_mock.py | テスト・評価 | TE-01, TE-02: モッキング |
| 25_testing_evaluation.py | テスト・評価 | TE-03, TE-04, TE-05: 評価 |

---

# Part 1: クイックスタート

## 1.1 最小構成 (01_quickstart.py)

**目的**: OpenAI Agents SDKの基本を理解する

### コアコンセプト

```python
from agents import Agent, Runner, function_tool

# 1. ツール定義
@function_tool
def get_weather(city: str) -> str:
    """都市の天気を取得する"""
    return f"{city}の天気: 晴れ、22°C"

# 2. エージェント作成
agent = Agent(
    name="WeatherBot",
    instructions="あなたは親切な天気アシスタントです。",
    tools=[get_weather],
)

# 3. 実行
result = Runner.run_sync(agent, "東京の天気は？")
print(result.final_output)
```

### 主要要素

| 要素 | 説明 |
|------|------|
| `@function_tool` | ツール定義用デコレータ |
| `Agent` | name、instructions、toolsを持つコアエージェントクラス |
| `Runner.run_sync` | 同期実行 |
| `result.final_output` | エージェントの最終応答 |

### LangGraphとの比較

| 観点 | OpenAI SDK | LangGraph |
|------|------------|-----------|
| セットアップ | Agentクラス | StateGraph + ノード + エッジ |
| 実行 | Runner.run_sync | graph.invoke |
| 複雑さ | シンプル、宣言的 | 複雑、明示的グラフ |

---

# Part 2: ツール呼び出し

## 2.1 ツール定義 (02_tool_definition.py)

**評価**: ⭐⭐⭐⭐⭐ (本番推奨)

### 定義方法

```python
# 方法1: シンプル
@function_tool
def get_weather(city: str) -> str:
    """都市の天気を取得する"""
    return f"天気: {city}"

# 方法2: Annotated使用
@function_tool
def get_weather_typed(
    city: Annotated[str, "都市名"],
    unit: Annotated[str, "温度単位"] = "celsius"
) -> str:
    """オプション付き天気取得"""
    return f"天気: {city}"

# 方法3: strictモード
@function_tool(strict_mode=True)
def get_weather_strict(city: str) -> str:
    """厳密なJSONスキーマバリデーション"""
    return f"天気: {city}"
```

### 比較

| 方法 | 利点 | 欠点 |
|------|------|------|
| シンプル | 最小コード | 引数説明なし |
| Annotated | 説明あり | 冗長 |
| strict_mode | JSON準拠 | 柔軟性低下 |

---

## 2.2 ツール実行 (03_tool_execution.py)

**評価**:
- TC-03 (並列): ⭐⭐⭐⭐⭐
- TC-04 (エラーハンドリング): ⭐⭐⭐⭐
- TC-05 (バリデーション): ⭐⭐⭐⭐⭐

### 主要な発見

| 機能 | サポート | 備考 |
|------|----------|------|
| 並列実行 | ✅ 完全 | エージェントあたり最大128ツール |
| エラーハンドリング | ✅ 良好 | 自動キャッチ、LLMリカバリ |
| Pydanticバリデーション | ✅ ネイティブ | 自動スキーマ生成 |

---

# Part 3: Human-in-the-Loop (HITL)

## 3.1 ネイティブ承認フロー (05_hitl_approval.py)

**評価** (v0.8.0で更新):
- HI-01 (中断API): ⭐⭐⭐⭐⭐
- HI-02 (状態操作): ⭐⭐⭐⭐
- HI-03 (再開制御): ⭐⭐⭐⭐⭐

### ネイティブHITL API (v0.8.0+)

```python
# 承認が必要なツールを定義
@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """ファイルを削除。人間の承認が必要。"""
    return f"削除完了: {path}"

# 条件付き承認（callable）
async def needs_approval_check(ctx, params, call_id) -> bool:
    return "/etc" in params.get("path", "")

@function_tool(needs_approval=needs_approval_check)
def read_file(path: str) -> str:
    """ファイル読み取り。機密パスには承認が必要。"""
    return f"内容: {path}"

# 実行と中断処理
result = await Runner.run(agent, "Delete /tmp/test.txt")

if result.interruptions:
    state = result.to_state()
    for interruption in result.interruptions:
        state.approve(interruption)  # or state.reject(interruption)
    result = await Runner.run(agent, state)  # 再開
```

### LangGraphとの比較

| 機能 | OpenAI SDK (v0.8.0+) | LangGraph |
|------|---------------------|-----------|
| 中断 | needs_approval=True | interrupt() |
| 再開 | Runner.run(agent, state) | Command(resume=...) |
| 状態 | RunState.to_json()/from_json() | Checkpointer |
| 承認/却下 | state.approve()/reject() | 状態更新 |
| **評価** | **LangGraphと同等** | ネイティブサポート |

---

# Part 4: 永続的実行

## 4.1 Sessions API (07_session_basic.py, 08_session_backends.py)

**評価**:
- DU-01 (永続化): ⭐⭐⭐⭐
- DU-04 (ストレージ選択肢): ⭐⭐⭐⭐⭐

### ストレージバックエンド

| バックエンド | ユースケース | 備考 |
|--------------|--------------|------|
| In-memory | 開発 | 非永続 |
| SQLite | シングルノード | ローカルファイル |
| SQLAlchemy | マルチデータベース | PostgreSQL、MySQL |
| Dapr | クラウドネイティブ | Kubernetes |
| Hosted | マネージド | OpenAIホスト |

---

# Part 5: メモリ

## 5.1 メモリ機能 (10-12スクリプト)

| 機能 | 評価 | 備考 |
|------|------|------|
| 短期 | ⭐⭐⭐⭐ | Conversations API |
| 長期 | ⭐⭐⭐⭐ | Sessions + Storage |
| File Search (RAG) | ⭐⭐⭐⭐ | 組み込み |
| エージェント自律 | ⭐ | LangMem相当なし |
| クリーンアップ | ⭐ | TTLなし |

### File Search (RAG)

OpenAI SDKにはFile Searchが組み込まれている:
- 自動チャンキングとEmbedding
- ベクトルストア管理
- セマンティック検索

---

# Part 6: マルチエージェント (MA)

## 6.1 ネイティブHandoffs (13_multiagent_handoff.py)

**評価**: ⭐⭐⭐⭐⭐ (本番推奨)

### Handoffパターン

```python
from agents import Agent, handoff

triage_agent = Agent(
    name="TriageAgent",
    instructions="適切な専門家にルーティングする。",
    handoffs=[
        handoff(flight_agent, tool_description_override="フライト関連の問い合わせ用"),
        handoff(hotel_agent, tool_description_override="ホテル関連の問い合わせ用"),
    ],
)
```

### 主要な利点

- ネイティブ `handoff()` 関数
- LLMが委譲タイミングを判断
- 説明がルーティング判断をガイド
- 循環handoffサポート

### LangGraph/CrewAIとの比較

| 機能 | OpenAI SDK | LangGraph | CrewAI |
|------|------------|-----------|--------|
| 委譲 | handoff()ネイティブ | 手動ツール | allow_delegation=True |
| ルーティング | LLM判断 | 条件付きエッジ | Processタイプ |
| コード | シンプル | 複雑 | シンプル |

---

# Part 7: ガバナンス (GV)

## 7.1 Guardrails (15_governance_guardrails.py)

**評価**:
- GV-01 (操作ゲート): ⭐⭐⭐⭐
- GV-03 (Policy as Code): ⭐⭐⭐

### Guardrailパターン

```python
from agents.guardrail import input_guardrail, InputGuardrailResult

@input_guardrail
async def validate_input(ctx, agent, input_text):
    if "blocked_pattern" in input_text:
        return InputGuardrailResult(
            output_info=None,
            tripwire_triggered=True
        )
    return InputGuardrailResult(output_info=None, tripwire_triggered=False)
```

### ガバナンスのギャップ

| 項目 | 状態 | 備考 |
|------|------|------|
| GV-02 最小権限 | ⭐⭐ | ネイティブ権限なし |
| GV-04 PII除去 | ⭐ | カスタム実装 |
| GV-05 テナント拘束 | ⭐ | カスタム実装 |
| GV-06 監査証跡 | ⭐⭐⭐ | トレーシング部分的 |

---

# Part 8: 決定性とリプレイ (DR)

## 8.1 リプレイの限界 (17_determinism_replay.py)

**評価**: ⭐⭐ (実験的)

### 主要な発見

- トレーシングが実行履歴を提供
- 組み込みLLMレスポンスキャッシュなし
- `seed`パラメータは限定的な再現性
- カスタムリプレイ実装が必要

### seedパラメータ

```python
# OpenAI APIはseedで再現性をサポート
model_params = {
    "temperature": 0,
    "seed": 42  # 同じフィンガープリント、同一出力ではない
}
```

---

# Part 9: コネクタ (CX)

## 9.1 Responses API (20_connectors_responses.py)

**評価**: ⭐⭐⭐⭐ (本番可)

### バックグラウンド実行

```python
# Responses API で background=true
result = api.create_response(
    agent_name="MyAgent",
    input_text="長時間タスク",
    background=True  # 即座にリターン
)
# 後で結果をポーリング
final = api.get_response(result["job_id"])
```

---

# Part 10: 観測性 (OB)

## 10.1 組み込みトレーシング (21_observability_tracing.py)

**評価**: ⭐⭐⭐⭐⭐ (本番推奨)

### 主要機能

- トレーシングがデフォルト有効
- エージェント、ツール、handoffスパン
- トークン使用量追跡 (request_usage_entries)
- ダッシュボード可視化

### 外部連携

| 連携先 | サポート | 備考 |
|--------|----------|------|
| Datadog | ✅ ネイティブ | 組み込み |
| Langfuse | ✅ フック | トレーシング経由 |
| Agenta | ✅ フック | トレーシング経由 |
| OpenTelemetry | ⚠️ カスタム | ブリッジが必要 |

---

# Part 11: テスト (TE)

## 11.1 モッキング (24_testing_mock.py)

**評価**: ⭐⭐⭐⭐ (本番可)

### テストパターン

```python
# model_settings経由でモックLLMを注入
mock_llm = MockLLM(responses=["固定レスポンス"])

# モックツール
mock_tool = MockTool("search", return_value={"results": []})

# テスト実行
result = agent_fn(input_data, mock_llm, {"search": mock_tool})

# アサーション
assert "expected" in result["output"]
assert mock_tool.was_called_with(query="test")
```

---

# フレームワーク比較

## LangGraphとの比較

| 観点 | OpenAI SDK | LangGraph |
|------|------------|-----------|
| 学習曲線 | 緩やか | 急 |
| 抽象化レベル | 高 | 低 |
| ツール定義 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| マルチエージェント | ⭐⭐⭐⭐⭐ handoffs | ⭐⭐⭐ 手動 |
| HITL | ⭐⭐⭐⭐ callbacks | ⭐⭐⭐⭐⭐ interrupt() |
| メモリ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ LangMem |
| ガバナンス | ⭐⭐⭐ guardrails | ⭐⭐ カスタム |
| 決定性 | ⭐⭐ | ⭐⭐ |
| 観測性 | ⭐⭐⭐⭐⭐ 組み込み | ⭐⭐⭐ LangSmith |
| 柔軟性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## CrewAIとの比較

| 観点 | OpenAI SDK | CrewAI |
|------|------------|--------|
| 委譲 | handoff() 明示的 | allow_delegation=True |
| エージェント定義 | Agentクラス | role/goal/backstoryを持つAgent |
| オーケストレーション | Handoffベース | Processタイプ |
| メモリ | Sessions | ネイティブエージェントメモリ |
| エンタープライズ重視 | 高 | 低 |

---

# 推奨事項

## OpenAI Agents SDKを使うべきケース

- シンプルから中程度のマルチエージェントワークフロー
- ネイティブOpenAIエコシステム統合
- 組み込みトレーシング要件
- 軽微なカスタマイズで本番投入
- シンプルなAPIを好むチーム

## 使うべきでないケース（カスタム実装なしで）

- 複雑なグラフトポロジー（LangGraph推奨）
- 高度なエージェント管理メモリ（LangGraph + LangMem推奨）
- 厳密なリプレイ/決定性要件
- コンプライアンスグレードの監査証跡
- カスタムLLMプロバイダ

## 本番デプロイ要件

外部書き込みを伴う本番環境では、以下を実装:

1. **ガバナンスレイヤー** (スクリプト15-16)
   - カスタムPolicyEngine
   - PIIRedactor
   - ハッシュチェーン付きAuditTrail

2. **決定性インフラ** (スクリプト17-18)
   - LLMレスポンスキャッシュ
   - IdempotencyManager
   - RecoveryManager

3. **コネクタ抽象化** (スクリプト19-20)
   - RateLimiter + ExponentialBackoff
   - CredentialManager

4. **観測性スタック** (スクリプト21-23)
   - OTelブリッジ
   - SLOManager + CostGuard

---

# 結論

## 総合評価: ⭐⭐⭐ (PoC可 - フェイルクローズ考慮)

OpenAI Agents SDKは、ネイティブhandoffsと組み込みトレーシングにより優れた開発者体験を提供するが、エンタープライズグレードの安全機能が不足している。

### 強み (⭐⭐⭐⭐+)

- **ツール呼び出し**: ネイティブPydantic統合、最大128ツール
- **マルチエージェント**: handoff()がエレガントで強力
- **トレーシング**: 組み込み、デフォルト有効
- **Sessions**: 複数のストレージバックエンド
- **シンプルさ**: クリーンで宣言的なAPI

### 弱み (⭐ から ⭐⭐)

- **ガバナンス**: ポリシーエンジンなし、監査限定的
- **決定性**: 部分的リプレイ、冪等性なし
- **メモリ**: エージェント自律管理なし
- **OTel**: ネイティブサポートなし
- **ガード**: SLO/コスト制御なし

### フェイルクローズの影響

以下の⭐評価により:
- DR-04 (冪等性) - 完全書き込み
- OB-06 (SLO/アラート) - 完全書き込み

**外部書き込みを伴う本番システムでは、これらの重要なギャップにより総合評価が制限される。**

TE-01 (ユニットテスト/モッキング) と GV-01 (操作ゲート) はパスしており、開発ワークフローをサポートする。

---

# ファイル構成

```
openai-sdk-example/
├── 01_quickstart.py              # クイックスタート
├── 02_tool_definition.py         # ツール定義比較
├── 03_tool_execution.py          # ツール実行（並列、エラー）
├── 04_tool_control.py            # 制御可能な自動化
├── 05_hitl_approval.py           # HITL承認フロー
├── 06_hitl_state.py              # HITL状態、タイムアウト、通知
├── 07_session_basic.py           # Sessions基本
├── 08_session_backends.py        # ストレージバックエンド
├── 09_session_production.py      # 本番考慮事項
├── 10_memory_conversation.py     # 会話メモリ
├── 11_memory_filesearch.py       # File Search / RAG
├── 12_memory_context.py          # コンテキスト管理
├── 13_multiagent_handoff.py      # ネイティブhandoffs
├── 14_multiagent_orchestration.py # オーケストレーションパターン
├── 15_governance_guardrails.py   # Guardrails
├── 16_governance_audit.py        # 監査証跡 & PII
├── 17_determinism_replay.py      # リプレイメカニズム
├── 18_determinism_recovery.py    # 復旧パターン
├── 19_connectors_streaming.py    # 認証 & レート制限
├── 20_connectors_responses.py    # Responses API
├── 21_observability_tracing.py   # 組み込みトレーシング
├── 22_observability_integration.py # 外部連携
├── 23_observability_guard.py     # SLO & コストガード
├── 24_testing_mock.py            # モッキング
├── 25_testing_evaluation.py      # 評価
├── REPORT.md                     # 英語レポート
├── REPORT_ja.md                  # 本レポート（日本語）
├── .env.example                  # 環境変数テンプレート
├── pyproject.toml
└── uv.lock
```
