# Inngest Durable Execution 検証レポート

## 概要

本レポートは、Inngest（Python SDK）をDurable Execution評価基準（ExoFlow OSDI'23、Flux OSDI'23、Temporal 4 Properties、Restate、NIST AI RMF、OTel Semantic Conventions）に基づいて評価した結果をまとめたものです。

## テスト環境

- Python: 3.13
- Inngest SDK: 0.5.x
- Inngest Dev Server: latest
- OS: macOS (Darwin)

---

## 星評価の基準

| 星 | ラベル | 定義 | 判定基準 |
|-------|-------|------------|-------------------|
| ⭐ | 未サポート | 機能なし or 壊れている | ドキュメントなし、動作しない、完全なカスタム実装が必要 |
| ⭐⭐ | 実験的 | 動作するが大きな制約、PoCでも苦労 | 動作するが落とし穴多数、ドキュメント不足、API不安定 |
| ⭐⭐⭐ | PoC対応 | 基本機能OK、デモ用途なら使えるが本番には追加作業要 | メインケースは動作、エッジケースに弱い、監視/ログはカスタム |
| ⭐⭐⭐⭐ | 本番対応 | 実用的、軽微なカスタマイズで本番デプロイ可 | 安定、ドキュメント充実、本番事例あり |
| ⭐⭐⭐⭐⭐ | 本番推奨 | そのまま本番利用可、ベストプラクティス確立 | 大規模本番事例、成熟したエコシステム |

---

## 前提パラメータ

| パラメータ | 値 |
|-----------|-------|
| 実行時間 | 分〜時間 |
| 副作用 | 制限付き書き込み |
| デプロイ | サーバーレス / クラウド |
| スケール | 〜100 |
| 決定性 | LLM判断を含む |
| 言語 | Python |

---

## カテゴリ別評価サマリ

### カバレッジサマリ（58項目）

| カテゴリ | 項目数 | 良好 (⭐⭐⭐+) | 要注意 (⭐⭐-) | 備考 |
|----------|-------|---------------|-----------------|-------|
| EX: 実行セマンティクス | 7 | 6 | 1 | Journal-based memoizationは優秀、決定性検出は弱い |
| RT: リトライ＆タイムアウト | 4 | 2 | 2 | リトライは良好、タイムアウト/Heartbeatは限定的 |
| WF: ワークフロープリミティブ | 6 | 6 | 0 | 完全なワークフロー構成要素 |
| SG: シグナル＆イベント | 4 | 3 | 1 | イベント駆動はネイティブ、クエリは限定的 |
| VR: バージョニング＆マイグレーション | 4 | 2 | 2 | 基本的なバージョニング、検出は弱い |
| CP: 補償＆リカバリ | 4 | 4 | 0 | 手動Saga、良好なリカバリ |
| PF: パフォーマンス＆オーバーヘッド | 3 | 2 | 1 | HTTPオーバーヘッド、スループットは良好 |
| OP: 運用 | 7 | 7 | 0 | 優秀なサーバーレスデプロイ |
| OB: 可観測性 | 6 | 5 | 1 | 良好なUI、OTelは限定的 |
| DX: 開発者体験 | 7 | 5 | 2 | シンプルなAPI、Time Skippingなし |
| AI: AI/Agent統合 | 6 | 5 | 1 | AgentKitは優秀、ストリーミングは限定的 |
| **合計** | **58** | **47** | **11** | |

### Fail-Close項目のステータス

| 項目 | 評価 | 影響 | 適用条件 |
|------|--------|--------|------------|
| EX-01 進行保証 | ⭐⭐⭐⭐⭐ | **PASS** - Journal-basedステップmemoization | 常に必須 |
| EX-03 冪等性 / 重複排除 | ⭐⭐⭐⭐⭐ | **PASS** - Event-level idempotency key | 常に必須 |
| RT-01 リトライ戦略 | ⭐⭐⭐⭐⭐ | **PASS** - Function retries + RetryAfterError | 常に必須 |
| VR-01 ワークフローバージョニング | ⭐⭐⭐ | **PASS** - function_idバージョニング（基本） | Duration >= 分 |
| OP-02 ワークフロー管理API | ⭐⭐⭐⭐ | **PASS** - List/Get/Replay/Cancel | 常に必須 |

> **Fail-Closeルール**: 全てのFail-Close項目が⭐⭐⭐以上でパス。総合評価への上限適用なし。

---

## 良好な項目（評価⭐⭐⭐以上）

| カテゴリ | ID | 項目 | 評価 | 備考 |
|----------|-----|------|--------|-------|
| 実行セマンティクス | EX-01 | 進行保証 | ⭐⭐⭐⭐⭐ | Journal-basedステップmemoizationで進行を保証 |
| 実行セマンティクス | EX-02 | 副作用保証 | ⭐⭐⭐⭐ | step.run()で結果をmemoize、外部冪等性はユーザー責任 |
| 実行セマンティクス | EX-03 | 冪等性 / 重複排除 | ⭐⭐⭐⭐⭐ | Event-level idempotency keyをネイティブサポート |
| 実行セマンティクス | EX-04 | 状態永続化 | ⭐⭐⭐⭐⭐ | Journal-based永続化、Event Sourcingよりシンプル |
| 実行セマンティクス | EX-05 | 決定性制約 | ⭐⭐⭐⭐ | 緩い制約 - ステップ境界のみが重要、ただしステップID変更で問題が発生する可能性あり |
| 実行セマンティクス | EX-07 | Replay正確性 | ⭐⭐⭐⭐ | ステップmemoizationは正確、Temporalより粒度は粗い |
| リトライ＆タイムアウト | RT-01 | リトライ戦略 | ⭐⭐⭐⭐⭐ | 指数バックオフ、ジッター、RetryAfterError、NonRetriableError |
| リトライ＆タイムアウト | RT-03 | サーキットブレーカー | ⭐⭐⭐ | max_attempts + NonRetriableErrorで同等の結果を達成 |
| ワークフロープリミティブ | WF-01 | ステップ定義 | ⭐⭐⭐⭐⭐ | step.run()はシンプルで効果的 |
| ワークフロープリミティブ | WF-02 | 子ワークフロー | ⭐⭐⭐⭐ | step.invoke()で子関数呼び出し |
| ワークフロープリミティブ | WF-03 | 並列実行 / Fan-out | ⭐⭐⭐⭐⭐ | step.parallel()で自然な並列実行 |
| ワークフロープリミティブ | WF-04 | 条件分岐 / ループ | ⭐⭐⭐⭐⭐ | 標準のPython制御フロー |
| ワークフロープリミティブ | WF-05 | スリープ / タイマー | ⭐⭐⭐⭐⭐ | 永続的スリープ、Cronトリガー |
| ワークフロープリミティブ | WF-06 | キュー / 流量制御 | ⭐⭐⭐⭐ | 同時実行制限、レート制限 |
| シグナル＆イベント | SG-01 | 外部シグナル | ⭐⭐⭐⭐⭐ | step.send_event()でイベント駆動通信 |
| シグナル＆イベント | SG-02 | ウェイト / Awaitables | ⭐⭐⭐⭐⭐ | step.wait_for_event()と式マッチング |
| シグナル＆イベント | SG-03 | イベントトリガー | ⭐⭐⭐⭐⭐ | イベント駆動がネイティブアーキテクチャ |
| バージョニング＆マイグレーション | VR-01 | ワークフローバージョニング | ⭐⭐⭐ | function_idバージョニング、実行中ランの手動drain待機が必要 |
| バージョニング＆マイグレーション | VR-04 | スキーマ進化 | ⭐⭐⭐⭐ | JSONの柔軟性、Pydanticバリデーション |
| 補償＆リカバリ | CP-01 | 補償 / Saga | ⭐⭐⭐⭐ | 手動try/except + 補償パターン |
| 補償＆リカバリ | CP-02 | 部分再開 | ⭐⭐⭐⭐⭐ | ステップmemoizationによる自動部分再開 |
| 補償＆リカバリ | CP-03 | 手動介入 | ⭐⭐⭐⭐ | Dashboard/API Replay/Cancel |
| 補償＆リカバリ | CP-04 | Dead Letter | ⭐⭐⭐⭐ | Failedステータスが可視、NonRetriableError |
| パフォーマンス＆オーバーヘッド | PF-02 | Fan-outスループット | ⭐⭐⭐⭐ | step.parallel()で良好なスループット |
| パフォーマンス＆オーバーヘッド | PF-03 | ペイロードサイズ制限 | ⭐⭐⭐⭐ | 512KB制限、外部ストレージパターン |
| 運用 | OP-01 | デプロイモデル | ⭐⭐⭐⭐⭐ | サーバーレスネイティブ、Cloud/Self-hostedオプション |
| 運用 | OP-02 | ワークフロー管理API | ⭐⭐⭐⭐ | List/Get/Replay/Cancel |
| 運用 | OP-03 | ストレージバックエンド | ⭐⭐⭐⭐ | PostgreSQL + Redis（Temporalよりシンプル） |
| 運用 | OP-04 | スケーラビリティ | ⭐⭐⭐⭐⭐ | サーバーレス自動スケーリング、同時実行制限 |
| 運用 | OP-05 | データ保持 / クリーンアップ | ⭐⭐⭐⭐ | プランベースの保持、自動クリーンアップ |
| 運用 | OP-06 | マルチリージョン / HA | ⭐⭐⭐⭐ | Cloud HA、Self-hostedは外部セットアップが必要 |
| 運用 | OP-07 | マルチテナント分離 | ⭐⭐⭐⭐ | App ID + concurrency key分離 |
| 可観測性 | OB-01 | ダッシュボード / UI | ⭐⭐⭐⭐⭐ | 優秀なdev serverとcloud UI |
| 可観測性 | OB-02 | メトリクス | ⭐⭐⭐⭐ | ダッシュボードメトリクス、Self-hosted用Prometheus |
| 可観測性 | OB-03 | 履歴の可視化 | ⭐⭐⭐⭐⭐ | ステップタイムライン、入出力検査 |
| 可観測性 | OB-05 | アラート | ⭐⭐⭐⭐ | Cloudアラート、webhook連携 |
| 可観測性 | OB-06 | ログ | ⭐⭐⭐⭐ | 標準Pythonロギング（コンテキスト付き） |
| 開発者体験 | DX-01 | SDK設計 | ⭐⭐⭐⭐⭐ | シンプルなデコレーターAPI、async/awaitネイティブ |
| 開発者体験 | DX-02 | 言語サポート | ⭐⭐⭐⭐ | TypeScript/Python/Goで良好な互換性 |
| 開発者体験 | DX-03 | ローカル開発 | ⭐⭐⭐⭐⭐ | npx inngest-cli dev、ホットリロード |
| 開発者体験 | DX-05 | エラーメッセージ / デバッグ | ⭐⭐⭐⭐ | 明確なエラー、スタックトレース、UIデバッグ |
| 開発者体験 | DX-06 | 学習曲線 | ⭐⭐⭐⭐⭐ | シンプルなメンタルモデル、緩い決定性 |
| AI/Agent統合 | AI-01 | LLM呼び出しのActivity化 | ⭐⭐⭐⭐⭐ | step.run()でLLM、結果はmemoized |
| AI/Agent統合 | AI-02 | 非決定性の扱い | ⭐⭐⭐⭐ | ステップ分離で非決定性に対応 |
| AI/Agent統合 | AI-03 | HITL / 人間承認 | ⭐⭐⭐⭐⭐ | step.wait_for_event()（タイムアウト付き） |
| AI/Agent統合 | AI-05 | Agent Framework統合 | ⭐⭐⭐⭐⭐ | AgentKitネイティブ（実験的、APIは変更の可能性あり）、LangGraph/CrewAIパターン |
| AI/Agent統合 | AI-06 | Tool実行の耐障害性 | ⭐⭐⭐⭐⭐ | Tool毎のステップ分離（リトライ付き） |

---

## 要注意項目（評価⭐⭐以下）

| カテゴリ | ID | 項目 | 評価 | 備考 | 検証スクリプト |
|----------|-----|------|--------|-------|---------------------|
| 実行セマンティクス | EX-06 | 決定性違反ハンドリング | ⭐⭐⭐ | 自動検出なし、ステップIDミスマッチでランタイムエラー | 01_ex_execution_semantics.py |
| リトライ＆タイムアウト | RT-02 | タイムアウト体系 | ⭐⭐⭐ | シンプルなモデル、Temporalのような細かいタイムアウトなし | 02_rt_retry_timeout.py |
| リトライ＆タイムアウト | RT-04 | Heartbeat | ⭐⭐ | Heartbeat機構なし、ステップ分割で対応 | 02_rt_retry_timeout.py |
| シグナル＆イベント | SG-04 | クエリ | ⭐⭐⭐ | ネイティブクエリなし、Run APIか外部状態を使用 | 04_sg_signals_events.py |
| バージョニング＆マイグレーション | VR-02 | 非互換変更の検出 | ⭐⭐ | 自動検出やreplayテストなし | 05_vr_versioning.py |
| バージョニング＆マイグレーション | VR-03 | マイグレーション戦略 | ⭐⭐⭐ | 手動blue-green、自動drainなし | 05_vr_versioning.py |
| パフォーマンス＆オーバーヘッド | PF-01 | ステップレイテンシ | ⭐⭐⭐ | HTTPオーバーヘッド（ステップ毎に〜10-50ms） | 07_pf_performance.py |
| 可観測性 | OB-04 | OTel準拠 | ⭐⭐⭐ | OTel利用可能だが手動セットアップが必要 | 09_ob_observability.py |
| 開発者体験 | DX-04 | テスト / Time Skipping | ⭐⭐ | Time Skippingなし、長いスリープは実時間が必要 | 10_dx_developer_experience.py |
| 開発者体験 | DX-07 | ローカルReplayハーネス | ⭐⭐ | 本番履歴をローカルでreplayできない | 10_dx_developer_experience.py |
| AI/Agent統合 | AI-04 | ストリーミング | ⭐⭐⭐ | Inngest境界を越えてストリーミングできない | 11_ai_agent_integration.py |

---

## 検証スクリプト

| スクリプト | カテゴリ | 主な検証項目 |
|--------|------------|------------------------|
| 01_ex_execution_semantics.py | 実行セマンティクス | EX-01〜EX-07: 進行、冪等性、replay |
| 02_rt_retry_timeout.py | リトライ＆タイムアウト | RT-01〜RT-04: リトライ、タイムアウト、heartbeat |
| 03_wf_workflow_primitives.py | ワークフロープリミティブ | WF-01〜WF-06: ステップ、子WF、並列、スリープ |
| 04_sg_signals_events.py | シグナル＆イベント | SG-01〜SG-04: シグナル、ウェイト、クエリ |
| 05_vr_versioning.py | バージョニング＆マイグレーション | VR-01〜VR-04: バージョニング、マイグレーション |
| 06_cp_compensation.py | 補償＆リカバリ | CP-01〜CP-04: Saga、再開、介入 |
| 07_pf_performance.py | パフォーマンス＆オーバーヘッド | PF-01〜PF-03: レイテンシ、スループット、ペイロード |
| 08_op_operations.py | 運用 | OP-01〜OP-07: デプロイ、管理、スケール |
| 09_ob_observability.py | 可観測性 | OB-01〜OB-06: UI、メトリクス、OTel、ログ |
| 10_dx_developer_experience.py | 開発者体験 | DX-01〜DX-07: SDK、テスト、デバッグ |
| 11_ai_agent_integration.py | AI/Agent統合 | AI-01〜AI-06: LLM、HITL、ツール |

---

## 主な知見

### 強み（⭐⭐⭐⭐+）

1. **開発者体験**（DX: 平均 ⭐⭐⭐⭐）
   - Temporalよりシンプルなメンタルモデル
   - 緩い決定性制約
   - 「普通のコード」+ ステップ境界
   - dev serverによる優秀なローカル開発

2. **運用**（OP: 平均 ⭐⭐⭐⭐⭐）
   - サーバーレスネイティブアーキテクチャ
   - シンプルなデプロイモデル
   - PostgreSQL + Redis（Temporalよりシンプル）
   - 組み込みcloudオプション

3. **AI/Agent統合**（AI: 平均 ⭐⭐⭐⭐⭐）
   - AgentKitでネイティブエージェントサポート（ツール管理、LLM呼び出し間の状態永続化、ツール失敗時の自動リトライを提供）
   - step.run()でLLM memoization
   - wait_for_eventによる優秀なHITL
   - LangGraph/CrewAI統合パターン

4. **ワークフロープリミティブ**（WF: 平均 ⭐⭐⭐⭐⭐）
   - step.run/parallel/sleep/invokeが包括的
   - 自然なPython制御フロー
   - 永続的タイマーとCron

5. **可観測性**（OB: 平均 ⭐⭐⭐⭐）
   - 優秀なdev server UI
   - ステップタイムライン可視化
   - 監視用cloudダッシュボード

### 検討事項

1. **Time Skipping**（DX-04: ⭐⭐）
   - 長時間ワークフローのテストにおける大きなギャップ
   - 回避策あり:
     - 環境変数によるスリープ時間のオーバーライド
     - テスト用のモック/スタブパターン
     - 統合テスト用の短い時間パラメータ
   - Temporalはこの点で大幅に優れている

2. **Heartbeat**（RT-04: ⭐⭐）
   - 長時間ステップ用のネイティブheartbeatなし
   - 複数ステップへの分割が必要
   - Temporalのheartbeat + タイムアウトの方が優れている

3. **バージョニング**（VR: 平均 ⭐⭐⭐）
   - 基本的なfunction_idバージョニング
   - 非互換変更の自動検出なし
   - TemporalのようなBuild IDルーティングなし

4. **ローカルReplay**（DX-07: ⭐⭐）
   - 本番履歴をローカルでreplayできない
   - 手動でのイベントコピーが必要
   - TemporalのWorkflowReplayerの方がはるかに優れている

5. **ステップレイテンシ**（PF-01: ⭐⭐⭐）
   - HTTPベースモデルでレイテンシが追加される
   - ステップ毎に〜10-50ms vs Temporalの〜4ms
   - 多くのユースケースでは許容範囲

---

## Temporal vs Inngest 比較

| 観点 | Temporal | Inngest |
|--------|----------|---------|
| 実行モデル | Event Sourcing + Replay | Journal-based + ステップmemoization |
| 決定性 | 厳密（ワークフロー全体） | 緩い（ステップ境界のみ） |
| 学習曲線 | 急（概念多い） | 緩やか（シンプルなモデル） |
| デプロイ | Worker + Server | サーバーレスHTTP |
| Time Skipping | ⭐⭐⭐⭐⭐ 組み込み | ⭐⭐ 利用不可 |
| Heartbeat | ⭐⭐⭐⭐⭐ ネイティブ | ⭐⭐ なし |
| バージョニング | ⭐⭐⭐⭐⭐ Build ID | ⭐⭐⭐ function_id |
| ローカルReplay | ⭐⭐⭐⭐⭐ WorkflowReplayer | ⭐⭐ 手動コピー |
| Agent統合 | ⭐⭐⭐⭐ Activityパターン | ⭐⭐⭐⭐⭐ AgentKit |
| インフラ | 重い（4サービス） | 軽い（Postgres + Redis） |
| ローカル開発 | ⭐⭐⭐⭐⭐ start-dev | ⭐⭐⭐⭐⭐ inngest-cli dev |
| コスト | サーバー + ストレージコスト | 従量課金（Cloud）またはセルフホスト |

### Inngestを使うべき場合

- シンプルなデプロイ要件
- サーバーレス/FaaSアーキテクチャ
- 緩い決定性制約でOK
- AgentKitを使ったAI/Agentワークロード
- 迅速なプロトタイピング
- 短時間のワークフロー

### 代わりにTemporalを使うべき場合

- 厳密なバージョニング要件
- テストにTime Skippingが必要
- heartbeatを使った長時間Activity
- 本番履歴のデバッグ（replay）
- Event Sourcing監査要件
- 多言語ワークフロー

---

## 推奨事項

### 本番デプロイチェックリスト

Inngestを本番デプロイする際のチェックリスト:

1. **インフラ**
   - [ ] Inngest Cloudアカウントまたはself-hostedセットアップ
   - [ ] 永続化用PostgreSQL（self-hosted）
   - [ ] キャッシュ用Redis（self-hosted）
   - [ ] 関数デプロイ（Vercel/AWS/コンテナ）

2. **監視**
   - [ ] ダッシュボードアクセス設定
   - [ ] アラートwebhook設定
   - [ ] Prometheusスクレイピング（self-hosted）

3. **開発プラクティス**
   - [ ] ステップIDが安定して意味のあるものになっている
   - [ ] dev serverで統合テスト
   - [ ] バージョニング用blue-greenデプロイ

---

## 結論

### 総合評価: ⭐⭐⭐⭐（本番対応）

Inngestは、シンプルさとAI/Agent統合に優れた本番対応のdurable executionプラットフォームです:

**強み:**
- durable executionプラットフォーム中で最もシンプルな学習曲線
- サーバーレスネイティブアーキテクチャ
- AIワークロード向けの優秀なAgentKit
- dev serverによる良好な開発者体験
- 緩い決定性で「普通のコード」が動作

**弱み:**
- Time Skippingなし（長時間ワークフローテストに重大な影響）
- Heartbeat機構なし
- 自動検出のない基本的なバージョニング
- 本番履歴をローカルでreplayできない

### 結論

Inngestは以下に**推奨**:
- durable executionを初めて使うチーム
- サーバーレス/FaaSアーキテクチャ
- HITLを使ったAI/Agentアプリケーション
- 短時間のワークフロー（分〜時間）

以下の場合は代わりにTemporalを検討:
- テストにTime Skippingが必要
- 厳密なバージョニングとreplayデバッグ
- heartbeatを使った長時間Activity
- 多言語サポート

58項目の評価で**47項目が⭐⭐⭐以上**、**全5つのFail-Close項目がパス**。Inngestは現代のサーバーレスアプリケーションに対してシンプルさと耐久性の優れたバランスを提供します。

---

# ファイル構成

```
inngest-example/
├── 01_ex_execution_semantics.py   # EXカテゴリ検証
├── 02_rt_retry_timeout.py         # RTカテゴリ検証
├── 03_wf_workflow_primitives.py   # WFカテゴリ検証
├── 04_sg_signals_events.py        # SGカテゴリ検証
├── 05_vr_versioning.py            # VRカテゴリ検証
├── 06_cp_compensation.py          # CPカテゴリ検証
├── 07_pf_performance.py           # PFカテゴリ検証
├── 08_op_operations.py            # OPカテゴリ検証
├── 09_ob_observability.py         # OBカテゴリ検証
├── 10_dx_developer_experience.py  # DXカテゴリ検証
├── 11_ai_agent_integration.py     # AIカテゴリ検証
├── serve.py                       # Inngest関数用FastAPIサーバー
├── common.py                      # 共有ユーティリティ
├── REPORT.md                      # 本レポート（英語版）
├── REPORT_ja.md                   # 本レポート（日本語版）
├── README.md                      # クイックスタートガイド
└── pyproject.toml
```
