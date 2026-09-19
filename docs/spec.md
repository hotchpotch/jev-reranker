# Jev reranker 仕様

## 目的と公開 API

HAKARI の `TypeSafeRerankerAdapter` を、評価基盤・GPU・NumPy に依存しない Python 3.11+ ライブラリへ移植する。初期公開済みの 0.0.1 は scaffolding のみであり、本仕様は次回リリース向けの実装を扱う。

`JevReranker(...).rank(query, documents, **kwargs)` と `rerank(...)` は同じ API。ともに `raw_rerank(...)` の `results` を返す薄い wrapper とする。入力は query 文字列と文書文字列の sequence。結果は `document_index`（入力の0始まり index）、`score`、既定で元の `text` を持つ辞書のリストで、スコア降順、同点は入力順。重複文書も別の候補として保持する。`top_k=None` は全件、0 は空、正整数は上位件数。top_k は出力のみを制限し、採点対象を削らない。`return_documents=False` で本文を省略する。

`raw_rerank(..., detail=True)` は JSON 化可能な `results` と `detail` を返す。各 result にも文書別 detail を付与する。raw は HTTP 応答そのものではなく、複数リクエストを統合した実行記録。`detail=False` では detail を保存しない。入力不正・不明な kwargs は API 呼び出し前に例外とし、黙って無視しない。非同期の `a_rank()` / `a_raw_rank()` を中心として、同期 wrapper と両方の context manager を提供する。`a_rerank()` / `a_raw_rerank()`、同期 `raw_rank()` は命名互換用 alias。呼び出し単位の統計を分離し、同時実行でも他の呼び出しの情報を混ぜない。

## asyncio 実行とライフサイクル

HTTP 通信は `httpx.AsyncClient`、同時実行制限は `asyncio.Semaphore`、retry 待機は `asyncio.sleep` を使う。HTTP 呼び出しごとの ThreadPoolExecutor は作らない。既存 `rank/rerank/raw_rerank` は同じ async 実装を呼ぶ blocking wrapper とし、採点・分割・検証のロジックを重複させない。

`await a_rank(query, documents, *, top_k=None, return_documents=True, detail=False)` は results リスト、`await a_raw_rank(...)` は results と全体 detail の辞書を返す。引数・同点順序・例外型は同期 API と共通。async API は `async with JevReranker(...)` または `await aclose()` で終了する。

instance は最初の async 利用時のイベントループに固定し、同じ AsyncClient/接続プール/Semaphore を再利用する。async instance を別ループで使う、または同期 API から使う場合は ConfigurationError。同期利用では instance ごとに1つの専用イベントループスレッドを遅延起動し、`run_coroutine_threadsafe` 経由で既存の複数スレッドから安全に共有する。同期 instance を別の async ループから使うことも拒否する。イベントループ内の同期メソッド呼び出しは実行前に拒否し、a_rank/a_raw_rank/aclose を案内する。

HTTP client は初回 request 時に作る。空入力や top_k=0 は HTTP client や tokenizer をロードしない。同期 API の loop は繰り返し呼び出しても作り直さず close 時に停止する。明示 `client` は httpx.AsyncClient のみを受け入れて所有権を移さない。`transport` は AsyncBaseTransport を受け入れて所有し、request 未実行でも終了時に閉じる。両方の同時指定は拒否する。外部 client は呼び出し側が同じループで管理する。

pointwise は最大 max_concurrency 個の worker task が入力を順に取り出し、pairwise は最大 max_concurrency ペアずつ処理する。大量候補から無制限の task を作らない。共有 Semaphore による上限は同一 instance の全 query に効き、retry 待機中も request 枠を占有する。listwise の分割 chunk は引き続き逐次処理する。Tokenizer の遅延ロード・前処理・計数は同期処理なので asyncio.to_thread に退避し、tokenizer lock を維持する。

各呼び出しの統計は独立した _Run に記録する。HTTP 側の状態更新は所属ループで行い、await を挟まない小さな更新に mutex は使わない。キャンセル時は子タスクを cancel・回収してから CancelledError をそのまま伝える。失敗時も子タスクを回収し、公開 API の JevError を ExceptionGroup に変換しない。キャンセルした HTTP trace の status は cancelled とする（兄弟 request の失敗で返る例外 detail から確認できる）。

aclose/close 開始後は新しい採点を拒否し、実行中の採点を待ってから所有する client/transport を閉じる。aclose 待機側がキャンセルされても cleanup task は shield し、再度 aclose で終了を待てる。利用者は loop を閉じる前に aclose を完了させる。close は同期ブリッジのループ・補助スレッドも終了させる。設定は使用中に変更しない。

## 採点方式と存在理由

- **listwise（既定）**: query と候補群を同じ state に含め、文書ごとに独立した Noul（回答に役立つ確率）を質問する。文書比較の文脈を共有し、query の重複送信を減らす。API が直接順列を返す意味ではない。候補が文脈上限に収まらなければ分割する。
- **pointwise**: query と1文書の state ごとに Noul を採点する。他候補の追加・削除が質問の文脈を変えず、大きな候補集合でも評価しやすい。候補数ぶんリクエストと query の重複が生じる。
- **pairwise**: 全ての非順序ペアについて「A は B より回答に役立つか」と逆方向を同じ state で質問する。A の勝率は `(p(A>B) + 1-p(B>A))/2`、B はその補数。各文書の平均勝率で順位を作る。位置バイアスの緩和と相対比較の実験用。O(n²) の質問・計算を要し、既定ではない。1文書は比較不要なので score=0.5。pairwise score は絶対的な関連確率ではなく、その候補集合での平均勝率である。

通常の関連判定は移植元と同じ英語の narrow question と true/false criteria を使う。日本語などの入力を翻訳せず JSON の query/documents として送る。`instructions` と `criteria` を constructor で差し替え可能。pointwise/listwise template は `{document}`、pairwise は `{left}` と `{right}` を必須とする。model は既定 `jev-latest`、固定バージョンも任意指定可能。

## 長さ計測、Tokenizer、長文、分割

既定は Python の `len(text)` による Unicode code point 数。tokenizer は optional とし、通常のインストールは httpx と python-dotenv のみを必要とする。`length_fn: Callable[[str], int]` で独自計測に差し替えられる。関数は決定的な非負整数を返し、bool や負数などの不正値・実行エラーは ConfigurationError。同期関数を worker thread で実行し、instance 内の計数 lock で保護する。文書と JSON 化した state/question/request に同じ方法を使う。tokenizer と length_fn の併用は拒否する。

`tokenizer="google/embeddinggemma-300m"` で tokenizer.json を Hub から遅延取得し token 数で計測する。名前指定時は `jev-reranker[tokenizer]` 相当の extra（checkout では `.[tokenizer]`）を必要とする。任意の encode/decode オブジェクトも受け取れる。モデルの重みや PyTorch は不要。Gemma 利用条件の承諾と HF 認証が必要な場合は利用者の環境/キャッシュを使う。Gemma の既定 revision は `57c266a740f537b4dc058e1b0cda161fd15afa75` に固定。`split_tokenizer_revision="main"` で追従可能。別 Hub repo、ローカル tokenizer.json も指定可能。`split_tokenizer_name` は名前指定の別名。

実環境では tokenizer を推奨する。特に英語では文字数が token 数より大きくなりやすく、文字基準は早すぎる切り詰め・分割につながる場合がある。ただし Gemma と Jev の内部 token 数は同一とは限らない。オブジェクト指定では model_max_length が tokenizer_max_length（既定65536）未満なら元オブジェクトの属性を自動で引き上げ、大きな既存値は維持する。属性がない独自 tokenizer は変更しない。必要な引き上げができない読み取り専用属性は ConfigurationError とする。名前指定の tokenizer の長さ設定は65536、truncation/padding は無効で、これを超えても全 token を数える。EmbeddingGemma 本体の embedding context を拡張する処理ではない。

`document_max_length=4000` が既定。len なら文字 prefix、tokenizer なら token prefix を decode→encode して上限内を確認する。独自関数では文字 prefix を二分探索して上限内の結果を再検査し、空 prefix すら収まらなければ例外。非単調な関数では最長 prefix を保証しない。短文は原文を保持する。None で切り詰め無効。query は切り詰めない。元テキスト・入力 index と採点に使った prefix の区別を維持する。

listwise の既定予算は state + 最大 question が `split_state_budget=26000`、request 全体が `split_request_budget=48000`。単位は選択した計測方法に従う。query、JSON、question を含めて検査する。文書数の差が最大1になるように長い文書から計測値の負荷を均す。候補順は query hash と元 index による決定的 shuffle とする。全候補をちょうど1回成功採点して raw Noul を統合する。同点判定には分割順を使わない。別 chunk では共有文脈が異なるため同じ較正は保証しない。旧 token 固有名の設定3つは互換 alias とし、計測単位を強制しない。

HTTP 400/422 の `detail.error_type=max_tokens_exceeded` では失敗 chunk の予算を半減して再分割する。成功済み chunk は再送しない。単一候補、pointwise の1文書、pairwise の1ペアが収まらなければ明示的な ContextLimitError とし、さらなる無断切り詰めや別モードへの変更はしない。

## 認証・設定

API キーの優先順位は明示 `api_key` → 実際の環境変数 `api_key_env` → 指定 `.env` の同名値。`api_key_env='TYPESAFE_API_KEY'` を既定にして既存 TypeSafe との互換性を保つ。dotenv はプロセス全体の環境を書き換えずに読む。`dotenv_path=None` で無効。空の環境変数は未設定として扱う。キーがなければ ConfigurationError。dotenv と detail に OpenAI 用変数は不要。

モデル指定は明示 model → JEV_MODEL（環境/.env）→ jev-latest。endpoint は明示 endpoint → TYPESAFE_ENDPOINT（環境/.env）→ https://api.typesafe.ai/v1/systemone。endpoint は完全な HTTPS URL（ローカル試験のみ HTTP localhost 可）、認証情報や query/fragment を URL に含めない。redirect を追従しない。

既定 max_concurrency は listwise=4、pointwise/pairwise=20。instance 内の同時 HTTP 呼び出し数を制限する。listwise chunk は逐次処理。timeout=180秒は各 HTTP I/O のタイムアウトで呼び出し全体の deadline ではない。

## エラーと retry

429/500/502/503/504/529 と httpx transport error のみ bounded retry。`max_retries=8` は初回に加えた最大再試行回数。指数 backoff + jitter、Retry-After（秒/HTTP date）を尊重し、1回の待機は最大60秒。401/403、その他 validation error、壊れた JSON、欠落/過剰 answer、非有限値・範囲外 score、欠落/不正 usage/model は即座に失敗する。失敗を0点に置き換えない。通信が失われた request も課金された可能性があり、usage は請求台帳ではない。

公開例外は JevError を基底として ConfigurationError、APIError（status_code）、ResponseValidationError、ContextLimitError。失敗時 detail を要求していれば例外の detail に途中までの実行記録を添付する。応答エラー本文や Authorization は例外・ログに出さない。

## detail と再現性

detail は schema_version=2、UTC 開始時刻、経過秒、package/Python/platform/dependency versions、endpoint、要求/解決 model、mode、実効 instructions/criteria、tokenizer/revision/予算、timeout/retry/concurrency、query/document hashes、切り詰めの original_length/sent_length/length_unit、split trace、リクエスト別質問・state・応答・試行回数・待機・usage、合算 usage を記録する。configuration に length_function と length_unit を、request に estimated_length を残す。usage の token 数は API の実測値でありローカル計測とは区別する。pairwise は相手 index と方向別判定も残す。api_key、HTTP 認証ヘッダー、任意の環境変数やマシンの hostname は収集しない。detail には query/document 本文が含まれるため、保存先と共有範囲は呼び出し側が管理する。自動でファイルを書かず、`json.dump(raw, ...)` で保存できる。

## テスト・配布

TDD で async HTTP mock と小さい tokenizer を用いた offline テストを先に追加する。順位、同点、重複、kwargs、環境優先順位、分割/復旧、retry、validation、detail、並行呼び出し、同期/非同期の一致、loop 所有権、キャンセル、client の終了処理を検証する。

live E2E は pytest の `--live` 明示指定のみで実行し通常 CI では skip。実際の .env と Gemma tokenizer を使い、同期・非同期の双方で4文書の関連度順序を日英・中国語・スペイン語・混在言語で確認する。返却スコアが整列していることだけでなく、期待 index 順と strict なスコア差を検査する。実サービスは変動し得るため fixture を緩めて成功扱いにせず、失敗時は detail を診断する。

uv cooldown と lock を維持し、tox、clean build、twine strict、隔離 wheel install、sdist/wheel 内容検査を通す。公開済み0.0.1の version/tag は再利用せず、公開作業は docs/release.md の reviewed PR フローで別途行う。

## 参照

- 移植元: hakari-bench `hakari_bench/models.py` の TypeSafeRerankerAdapter（MIT）。
- https://docs.typesafe.ai/api
- https://docs.typesafe.ai/models
- https://docs.typesafe.ai/cookbooks/rerank_typesafe
- https://huggingface.co/google/embeddinggemma-300m
