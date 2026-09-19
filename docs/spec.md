# Jev reranker 仕様

## 目的と公開 API

HAKARI の `TypeSafeRerankerAdapter` を、評価基盤・GPU・NumPy に依存しない Python 3.11+ ライブラリへ移植する。初期公開済みの 0.0.1 は scaffolding のみであり、本仕様は次回リリース向けの実装を扱う。

`JevReranker(...).rank(query, documents, **kwargs)` と `rerank(...)` は同じ API。ともに `raw_rerank(...)` の `results` を返す薄い wrapper とする。入力は query 文字列と文書文字列の sequence。結果は `corpus_id`（入力の0始まり index）、`score`、既定で元の `text` を持つ辞書のリストで、スコア降順、同点は入力順。重複文書も別の候補として保持する。`top_k=None` は全件、0 は空、正整数は上位件数。top_k は出力のみを制限し、採点対象を削らない。`return_documents=False` で本文を省略する。

`raw_rerank(..., detail=True)` は JSON 化可能な `results` と `detail` を返す。各 result にも文書別 detail を付与する。raw は HTTP 応答そのものではなく、複数リクエストを統合した実行記録。`detail=False` では detail を保存しない。入力不正・不明な kwargs は API 呼び出し前に例外とし、黙って無視しない。同期 API と context manager / `close()` を提供する。呼び出し単位の統計を分離し、同時実行でも他の呼び出しの情報を混ぜない。

## 採点方式と存在理由

- **listwise（既定）**: query と候補群を同じ state に含め、文書ごとに独立した Noul（回答に役立つ確率）を質問する。文書比較の文脈を共有し、query の重複送信を減らす。API が直接順列を返す意味ではない。候補が文脈上限に収まらなければ分割する。
- **pointwise**: query と1文書の state ごとに Noul を採点する。他候補の追加・削除が質問の文脈を変えず、大きな候補集合でも評価しやすい。候補数ぶんリクエストと query の重複が生じる。
- **pairwise**: 全ての非順序ペアについて「A は B より回答に役立つか」と逆方向を同じ state で質問する。A の勝率は `(p(A>B) + 1-p(B>A))/2`、B はその補数。各文書の平均勝率で順位を作る。位置バイアスの緩和と相対比較の実験用。O(n²) の質問・計算を要し、既定ではない。1文書は比較不要なので score=0.5。pairwise score は絶対的な関連確率ではなく、その候補集合での平均勝率である。

通常の関連判定は移植元と同じ英語の narrow question と true/false criteria を使う。日本語などの入力を翻訳せず JSON の query/documents として送る。`instructions` と `criteria` を constructor で差し替え可能。pointwise/listwise template は `{document}`、pairwise は `{left}` と `{right}` を必須とする。model は既定 `jev-latest`、固定バージョンも任意指定可能。

## Tokenizer、長文、分割

既定は `google/embeddinggemma-300m` の tokenizer.json を Hugging Face Hub から遅延取得する。モデルの重みや PyTorch は不要。Gemma 利用条件の承諾と HF 認証が必要な場合は利用者の環境/キャッシュを使う。既定 revision は `57c266a740f537b4dc058e1b0cda161fd15afa75` に固定する。`split_tokenizer_revision="main"` で追従も可能。リビジョン、ローカル tokenizer.json パス、encode/decode を持つ tokenizer オブジェクトを指定可能。

tokenizer の長さ設定は 65536、truncation/padding は無効にして計数する。EmbeddingGemma 本体の 2048-token embedding context を拡張する処理ではなく、Jev に送る文字列を測定する tokenizer-only 設定である。65536 より長い入力も切らずに計数し、Jev の内部 tokenizer とは一致を保証しない。設定値・取得 revision を detail に残す。

`document_max_tokens=4000` が既定。長い文書の末尾を落とし、decode→encode 後も制限を満たすことを検証する。短文は原文を保持する。None で切り詰め無効。query は切り詰めない。tokenizer と予算を変更可能で、元テキスト・入力 index と採点に使った prefix の区別を維持する。

listwise の既定予算は state + 最大 question が26000、request 全体が48000の推定 token。query、JSON、question を含めて検査する。文書数の差が最大1になるように長い文書から token load を均す。分割した候補の順番は query hash と元 index による決定的 shuffle とする。全候補をちょうど1回成功採点して raw Noul を統合する。同点判定には分割順を使わない。別 chunk では共有文脈が異なるため同じ較正は保証しない。

HTTP 400/422 の `detail.error_type=max_tokens_exceeded` では失敗 chunk の予算を半減して再分割する。成功済み chunk は再送しない。単一候補、pointwise の1文書、pairwise の1ペアが収まらなければ明示的な ContextLimitError とし、さらなる無断切り詰めや別モードへの変更はしない。

## 認証・設定

API キーの優先順位は明示 `api_key` → 実際の環境変数 `api_key_env` → 指定 `.env` の同名値。`api_key_env='TYPESAFE_API_KEY'` を既定にして既存 TypeSafe との互換性を保つ。dotenv はプロセス全体の環境を書き換えずに読む。`dotenv_path=None` で無効。空の環境変数は未設定として扱う。キーがなければ ConfigurationError。dotenv と detail に OpenAI 用変数は不要。

モデル指定は明示 model → JEV_MODEL（環境/.env）→ jev-latest。endpoint は明示 endpoint → TYPESAFE_ENDPOINT（環境/.env）→ https://api.typesafe.ai/v1/systemone。endpoint は完全な HTTPS URL（ローカル試験のみ HTTP localhost 可）、認証情報や query/fragment を URL に含めない。redirect を追従しない。

既定 max_concurrency は listwise=4、pointwise/pairwise=20。instance 内の同時 HTTP 呼び出し数を制限する。listwise chunk は逐次処理。timeout=180秒は各 HTTP I/O のタイムアウトで呼び出し全体の deadline ではない。

## エラーと retry

429/500/502/503/504/529 と httpx transport error のみ bounded retry。`max_retries=8` は初回に加えた最大再試行回数。指数 backoff + jitter、Retry-After（秒/HTTP date）を尊重し、1回の待機は最大60秒。401/403、その他 validation error、壊れた JSON、欠落/過剰 answer、非有限値・範囲外 score、欠落/不正 usage/model は即座に失敗する。失敗を0点に置き換えない。通信が失われた request も課金された可能性があり、usage は請求台帳ではない。

公開例外は JevError を基底として ConfigurationError、APIError（status_code）、ResponseValidationError、ContextLimitError。失敗時 detail を要求していれば例外の detail に途中までの実行記録を添付する。応答エラー本文や Authorization は例外・ログに出さない。

## detail と再現性

detail は schema_version、UTC 開始時刻、経過秒、package/Python/platform/dependency versions、endpoint、要求/解決 model、mode、実効 instructions/criteria、tokenizer/revision/予算、timeout/retry/concurrency、query/document hashes、切り詰めの元/送信 token 数、split trace、リクエスト別質問・state・応答・試行回数・待機・usage、合算 usage を記録する。pairwise は相手 index と方向別判定も残す。api_key、HTTP 認証ヘッダー、任意の環境変数やマシンの hostname は収集しない。detail には query/document 本文が含まれるため、保存先と共有範囲は呼び出し側が管理する。自動でファイルを書かず、`json.dump(raw, ...)` で保存できる。

## テスト・配布

TDD で HTTP mock と小さい tokenizer を用いた offline テストを先に追加する。順位、同点、重複、kwargs、環境優先順位、分割/復旧、retry、validation、detail、並行呼び出しを検証する。

live E2E は pytest の `--live` 明示指定のみで実行し通常 CI では skip。実際の .env と Gemma tokenizer を使い、4文書の関連度順序を日英・中国語・スペイン語・混在言語で確認する。返却スコアが整列していることだけでなく、期待 index 順と strict なスコア差を検査する。実サービスは変動し得るため fixture を緩めて成功扱いにせず、失敗時は detail を診断する。

uv cooldown と lock を維持し、tox、clean build、twine strict、隔離 wheel install、sdist/wheel 内容検査を通す。公開済み0.0.1の version/tag は再利用せず、公開作業は docs/release.md の reviewed PR フローで別途行う。

## 参照

- 移植元: hakari-bench `hakari_bench/models.py` の TypeSafeRerankerAdapter（MIT）。
- https://docs.typesafe.ai/api
- https://docs.typesafe.ai/models
- https://docs.typesafe.ai/cookbooks/rerank_typesafe
- https://huggingface.co/google/embeddinggemma-300m
