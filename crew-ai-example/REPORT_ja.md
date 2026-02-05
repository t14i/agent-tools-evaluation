# CrewAI 検証レポート

## 概要

本レポートは、Agent Framework評価基準（NIST AI RMF、WEF AI Agents in Action、IMDA Model AI Governance、OTel GenAI Semantic Conventions）に基づき、CrewAIの本番環境適用可能性を検証した結果をまとめたものです。

## 検証環境

- Python: 3.13
- CrewAI: 1.9.3
- crewai-tools: 0.20.0+

---

## 星評価の基準

| 星 | ラベル | 定義 | 判定基準 |
|----|--------|------|----------|
| ⭐ | 未対応 | 機能がない、または壊れている | ドキュメントに記載なし、実装しても動かない、完全に自前実装が必要 |
| ⭐⭐ | 実験的 | 動くが制約が大きく、PoCでも苦労する | 動作はするがハマりどころ多い、ドキュメント不足、APIが不安定 |
| ⭐⭐⭐ | PoC可 | 基本動作OK、デモには使えるが本番には追加実装が必要 | 主要ケースは動く、エッジケースに弱い、監視・ログ等は自前 |
| ⭐⭐⭐⭐ | 本番可 | 実用的、多少のカスタマイズで本番投入できる | 安定動作、ドキュメント充実、本番事例あり、軽微なカスタマイズで済む |
| ⭐⭐⭐⭐⭐ | 本番推奨 | そのまま本番で使える、ベストプラクティスが確立 | 大規模本番事例あり、エコシステム成熟、運用ノウハウが蓄積 |

---

## 前提パラメータ

| パラメータ | 値 |
|-----------|-----|
| Autonomy | 要承認 |
| Authority | 制限付き書き込み |
| Predictability | LLM判断あり |
| Context | 社内データ |

---

## カテゴリ別評価サマリー

### カバレッジサマリー（52項目）

| カテゴリ | 項目数 | Good (⭐⭐⭐+) | Not Good (⭐⭐-) | 備考 |
|---------|--------|---------------|-----------------|------|
| TC (5) | 5 | 3 | 2 | Tool定義は強い、制御が弱い |
| HI (5) | 5 | 2 | 3 | 基本HITLのみ、タイムアウト/通知なし |
| DU (6) | 6 | 4 | 2 | @persistは良い、TTL/並行はカスタム |
| ME (8) | 8 | 2 | 6 | 基本メモリは良い、高度機能欠如 |
| MA (5) | 5 | 5 | 0 | **最強カテゴリ** |
| GV (6) | 6 | 0 | 6 | ネイティブなガバナンスなし |
| DR (6) | 6 | 0 | 6 | ネイティブな決定論性/リプレイなし |
| CX (4) | 4 | 1 | 3 | スキーマ検証のみ利用可 |
| OB (7) | 7 | 3 | 4 | 基本ログは良い、OTel/SLOなし |
| **合計** | **52** | **20** | **32** | |

### フェイルクローズ項目の状態

| 項目 | 評価 | 影響 |
|------|------|------|
| GV-01 破壊的操作のゲート | ⭐ | **FAIL** - ネイティブなゲートなし |
| DR-01 Replay（再現実行） | ⭐ | **FAIL** - ネイティブなリプレイなし |
| DR-04 冪等 / Exactly-once | ⭐ | **FAIL** - ネイティブな冪等性なし |
| CX-02 レート制限・リトライ | ⭐ | **FAIL** - ネイティブなレート制限なし |

> **フェイルクローズルール**: これらの項目が⭐⭐以下の場合、他カテゴリの評価に関係なく総合評価の上限は⭐⭐となる。

---

## Good項目（⭐⭐⭐以上）

| カテゴリ | ID | 項目 | 評価 | 備考 |
|---------|-----|------|------|------|
| TC | TC-01 | Tool定義 | ⭐⭐⭐⭐ | @toolデコレータ + BaseToolクラス、ドキュメント充実 |
| TC | TC-04 | エラーハンドリング | ⭐⭐⭐ | max_retry_limitあり、基本的なリトライ |
| TC | TC-05 | 引数バリデーション | ⭐⭐⭐⭐ | Pydantic args_schemaネイティブサポート |
| HI | HI-01 | 中断API | ⭐⭐⭐ | human_input=Trueで実行停止 |
| HI | HI-03 | 再開制御 | ⭐⭐⭐ | Flowベースで承認/却下可能 |
| DU | DU-01 | 状態永続化 | ⭐⭐⭐⭐ | @persist + SQLiteネイティブ |
| DU | DU-02 | プロセス再開 | ⭐⭐⭐⭐ | kickoff(inputs={'id': ...})で動作 |
| DU | DU-03 | HITL永続化 | ⭐⭐⭐ | @persistと連携して動作 |
| DU | DU-04 | ストレージ選択肢 | ⭐⭐⭐ | SQLite組み込み、他はカスタム |
| ME | ME-01 | 短期メモリ | ⭐⭐⭐⭐ | memory=True、設定不要 |
| ME | ME-02 | 長期メモリ | ⭐⭐⭐⭐ | セッション跨ぎで永続化 |
| MA | MA-01 | 複数Agent定義 | ⭐⭐⭐⭐⭐ | Crew/Agent抽象化、直感的 |
| MA | MA-02 | 委譲 | ⭐⭐⭐⭐⭐ | allow_delegation=True、独自の強み |
| MA | MA-03 | 階層プロセス | ⭐⭐⭐⭐ | Process.hierarchical + manager_agent |
| MA | MA-04 | ルーティング | ⭐⭐⭐⭐ | Flowの@router |
| MA | MA-05 | 共有メモリ | ⭐⭐⭐⭐ | エージェント間メモリ共有ネイティブ |
| OB | OB-01 | トレース | ⭐⭐⭐ | verbose=Trueで実行パス表示 |
| OB | OB-02 | トークン消費 | ⭐⭐⭐⭐ | result.token_usageネイティブ |
| OB | OB-03 | ログ出力 | ⭐⭐⭐ | output_log_fileあり、構造化は未対応 |
| CX | CX-04 | スキーマ/コントラクト | ⭐⭐⭐ | Pydanticでバリデーション可能 |

---

## Not Good項目（⭐⭐以下）

| カテゴリ | ID | 項目 | 評価 | 備考 | 検証スクリプト |
|---------|-----|------|------|------|---------------|
| TC | TC-02 | 制御可能な自動化 | ⭐ | ネイティブなポリシー制御なし、エージェント自動実行 | 14_governance_gate.py |
| TC | TC-03 | 並列実行 | ⭐⭐ | ネイティブな並列Tool呼び出しなし | - |
| HI | HI-02 | 状態操作 | ⭐⭐ | 中断中の状態アクセス制限あり | 06_hitl_flow_feedback.py |
| HI | HI-04 | タイムアウト | ⭐ | ネイティブなタイムアウト管理なし | - |
| HI | HI-05 | 通知 | ⭐ | ネイティブなWebhook/メール通知なし | - |
| DU | DU-05 | クリーンアップ（TTL） | ⭐ | ネイティブなTTLなし、カスタムStateTTLManager必要 | 07_durable_basic.py |
| DU | DU-06 | 並行アクセス | ⭐ | ネイティブなロックなし、カスタムConcurrencyManager必要 | 07_durable_basic.py |
| ME | ME-03 | セマンティック検索 | ⭐ | ネイティブな検索APIなし、カスタム実装必要 | 11_memory_basic.py |
| ME | ME-04 | メモリAPI | ⭐ | CRUD APIなし、カスタム実装必要 | 11_memory_basic.py |
| ME | ME-05 | Agent自律管理 | ⭐ | ネイティブサポートなし、カスタムAutonomousMemoryAgent必要 | 11_memory_basic.py |
| ME | ME-06 | 自動抽出 | ⭐ | ネイティブなファクト抽出なし | 11_memory_basic.py |
| ME | ME-07 | メモリクリーンアップ（TTL） | ⭐ | ネイティブなメモリTTLなし | 11_memory_basic.py |
| ME | ME-08 | Embeddingコスト | ⭐ | ネイティブなコスト追跡なし | 11_memory_basic.py |
| GV | GV-01 | 破壊的操作のゲート | ⭐ | ネイティブなゲート機構なし | 14_governance_gate.py |
| GV | GV-02 | 最小権限 / スコープ | ⭐ | ネイティブな権限システムなし | 16_governance_policy.py |
| GV | GV-03 | Policy as Code | ⭐ | ネイティブなポリシーエンジンなし | 16_governance_policy.py |
| GV | GV-04 | PII / 秘匿 | ⭐ | ネイティブなRedactionなし | 17_governance_audit.py |
| GV | GV-05 | テナント / 目的拘束 | ⭐ | ネイティブな目的拘束なし | - |
| GV | GV-06 | 監査証跡の完全性 | ⭐ | ネイティブな監査ログなし | 17_governance_audit.py |
| DR | DR-01 | Replay（再現実行） | ⭐ | ネイティブなリプレイ機構なし | 15_determinism_replay.py |
| DR | DR-02 | Evidence（根拠参照） | ⭐ | ネイティブな根拠収集なし | 18_determinism_evidence.py |
| DR | DR-03 | 非決定性の隔離 | ⭐ | ネイティブなLLM隔離モードなし | 18_determinism_evidence.py |
| DR | DR-04 | 冪等 / Exactly-once | ⭐ | ネイティブな冪等性キーなし | 15_determinism_replay.py |
| DR | DR-05 | Plan差分 | ⭐ | ネイティブな差分表示なし | 19_determinism_recovery.py |
| DR | DR-06 | 障害復旧 | ⭐ | ネイティブな復旧機構なし | 19_determinism_recovery.py |
| CX | CX-01 | 認証・資格情報管理 | ⭐ | ネイティブなOAuth/トークン管理なし | 20_connectors_auth.py |
| CX | CX-02 | レート制限・リトライ | ⭐ | ネイティブなレート制限なし | 04_tool_error_handling.py |
| CX | CX-03 | Async Job Pattern | ⭐ | ネイティブなジョブ追跡なし | 21_connectors_async.py |
| OB | OB-04 | 外部連携 | ⭐⭐ | ネイティブなLangSmith/Langfuse連携なし | - |
| OB | OB-05 | OTel準拠 | ⭐ | ネイティブなOpenTelemetryサポートなし | 22_observability_otel.py |
| OB | OB-06 | SLO / アラート | ⭐ | ネイティブなSLO管理なし | 23_observability_guard.py |
| OB | OB-07 | コストガード | ⭐ | ネイティブな予算/kill switchなし | 23_observability_guard.py |

---

## 検証スクリプト一覧

| スクリプト | カテゴリ | 主な検証項目 |
|-----------|----------|--------------|
| 01_quickstart.py | - | CrewAI基本構造 |
| 02_tool_definition.py | TC | TC-01: @tool, BaseTool, args_schema |
| 03_tool_execution.py | TC | TC-01, TC-04: Tool実行、キャッシュ |
| 04_tool_error_handling.py | TC, CX | TC-04, CX-02: エラーハンドリング、レート制限 |
| 05_hitl_task_input.py | HI | HI-01: human_input=True |
| 06_hitl_flow_feedback.py | HI | HI-01, HI-02, HI-03: FlowベースHITL |
| 07_durable_basic.py | DU | DU-01, DU-05, DU-06: Flow、TTL、並行制御 |
| 08_durable_resume.py | DU | DU-01, DU-02, DU-03: @persist、再開 |
| 09_collaboration_delegation.py | MA | MA-02: 委譲 |
| 10_collaboration_hierarchical.py | MA | MA-03: 階層プロセス |
| 11_memory_basic.py | ME | ME-01〜ME-08: メモリ機能 |
| 12_memory_longterm.py | ME | ME-02: 長期永続化 |
| 13_production_concerns.py | OB | OB-01〜OB-03: ログ、トークン |
| 14_governance_gate.py | GV, TC | GV-01, TC-02: 破壊的操作ゲート |
| 15_determinism_replay.py | DR | DR-01, DR-04: リプレイ、冪等性 |
| 16_governance_policy.py | GV | GV-02, GV-03: 最小権限、Policy as Code |
| 17_governance_audit.py | GV | GV-04, GV-06: PII秘匿、監査証跡 |
| 18_determinism_evidence.py | DR | DR-02, DR-03: 根拠参照、決定論モード |
| 19_determinism_recovery.py | DR | DR-05, DR-06: Plan差分、障害復旧 |
| 20_connectors_auth.py | CX | CX-01: OAuth、シークレット管理 |
| 21_connectors_async.py | CX | CX-03, CX-04: 非同期ジョブ、スキーマ検証 |
| 22_observability_otel.py | OB | OB-05: OpenTelemetry統合 |
| 23_observability_guard.py | OB | OB-06, OB-07: SLO、コストガード |

---

## 主な発見事項

### 強み（⭐⭐⭐⭐以上）

1. **マルチエージェント協調** (MA: 平均⭐⭐⭐⭐☆)
   - allow_delegation=Trueによるネイティブな委譲
   - 直感的なCrew/Agent/Task抽象化
   - role/goal/backstoryデザインパターン
   - manager_agentによる階層プロセス

2. **基本メモリ** (ME-01, ME-02: ⭐⭐⭐⭐)
   - memory=Trueで自動メモリ有効化
   - セッション跨ぎの長期永続化
   - エンティティメモリサポート

3. **耐久的実行コア** (DU-01〜DU-04: ⭐⭐⭐⭐)
   - @persistデコレータ + SQLite
   - IDによる状態再開
   - Flowベースのワークフロー

4. **Tool定義** (TC-01, TC-05: ⭐⭐⭐⭐)
   - @toolデコレータが直感的
   - Pydantic args_schemaバリデーション

### 弱み（⭐〜⭐⭐）

1. **ガバナンス** (GV: 全て⭐)
   - 破壊的操作のゲートなし
   - ポリシーエンジンなし
   - PII Redactionなし
   - 監査証跡なし
   - **全てカスタム実装が必要**

2. **決定論性とリプレイ** (DR: 全て⭐)
   - リプレイ機能なし
   - 根拠収集なし
   - 冪等性サポートなし
   - 復旧機構なし
   - **全てカスタム実装が必要**

3. **コネクタ** (CX: 4項目中3つが⭐)
   - OAuth/資格情報管理なし
   - レート制限なし
   - 非同期ジョブパターンなし

4. **高度な可観測性** (OB-05〜OB-07: ⭐)
   - OpenTelemetryサポートなし
   - SLO管理なし
   - コストガード/kill switchなし

---

## LangGraph比較

| 観点 | CrewAI | LangGraph |
|------|--------|-----------|
| 学習曲線 | 緩やか | 急 |
| 抽象度 | 高い | 低い |
| マルチエージェント (MA) | ⭐⭐⭐⭐⭐ ネイティブ | ⭐⭐⭐ カスタム |
| 委譲 | ⭐⭐⭐⭐⭐ ビルトイン | ⭐⭐ 手動 |
| HITL (HI) | ⭐⭐⭐ 基本 | ⭐⭐⭐⭐ 高度 |
| 耐久実行 (DU) | ⭐⭐⭐⭐ @persist | ⭐⭐⭐⭐ Checkpointer |
| ガバナンス (GV) | ⭐ カスタム | ⭐ カスタム |
| 決定論性 (DR) | ⭐ カスタム | ⭐⭐ 部分的 |
| 可観測性 (OB) | ⭐⭐ 限定的 | ⭐⭐⭐⭐ LangSmith |
| 本番実績 | 成長中 | 確立済み |

---

## 推奨事項

### CrewAIを使うべき場合

- マルチエージェント協調が主要要件
- 委譲/階層プロセスが必要
- 直感的な抽象化で素早いプロトタイピング
- role/goal/backstoryエージェント設計が好ましい
- **読み取り専用またはリスクの低い書き込み操作**

### CrewAIを使うべきでない場合（カスタム実装なしで）

- 破壊的操作のある本番システム
- 規制準拠要件（監査証跡、PII）
- リプレイ/デバッグ機能が必要なシステム
- 大量API連携（レート制限が必要）

### 本番デプロイ要件

外部書き込みを伴う本番デプロイには以下を実装:

1. **ガバナンス層**（スクリプト14-17）
   - 破壊的操作用ApprovalFlow
   - 最小権限用PolicyEngine
   - 機密データ用PIIRedactor
   - ハッシュチェーン付きAuditTrail

2. **決定論性インフラ**（スクリプト15, 18, 19）
   - インシデント調査用ReplayLogger
   - Exactly-once用IdempotencyKeyManager
   - 障害処理用RecoveryManager

3. **コネクタ抽象化**（スクリプト04, 20, 21）
   - TokenBucket + ExponentialBackoff
   - OAuthClient + SecretManager
   - AsyncJobExecutor

4. **可観測性スタック**（スクリプト22, 23）
   - OTel用CrewAIInstrumentor
   - SLOManager + CostGuard

---

## 結論

### 総合評価: ⭐⭐（実験的 - フェイルクローズ適用後）

CrewAIはマルチエージェント協調に優れるが、エンタープライズグレードの安全機能が欠如:

✅ **本番可（⭐⭐⭐⭐+）:**
- マルチエージェントシステム (MA)
- 基本メモリ (ME-01, ME-02)
- 状態永続化 (DU-01〜DU-04)
- Tool定義 (TC-01, TC-05)

⚠️ **PoC可（⭐⭐⭐）:**
- 基本HITL (HI-01, HI-03)
- 基本可観測性 (OB-01〜OB-03)

❌ **未対応（⭐）:**
- ガバナンス (GV-01〜GV-06)
- 決定論性とリプレイ (DR-01〜DR-06)
- 高度なコネクタ (CX-01〜CX-03)
- 高度な可観測性 (OB-05〜OB-07)

### フェイルクローズの影響

以下の⭐評価により:
- GV-01（破壊的操作のゲート）
- DR-01（Replay）
- DR-04（冪等性）
- CX-02（レート制限・リトライ）

**外部書き込みを伴う本番システムの総合評価は⭐⭐が上限となる。**

### 推奨

検証スクリプト（01-23）を不足機能実装のテンプレートとして活用。本番デプロイ時:
1. まずフェイルクローズ項目を全て実装
2. ガバナンス層を追加
3. 可観測性スタックを構築
4. 高度なHITLや可観測性が重要ならLangGraphも検討
