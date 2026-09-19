# jev-reranker

TypeSafe Jev による、多言語対応の Python reranker。`listwise`、`pointwise`、`pairwise` を選べ、長さに応じた分割、retry、実行内容を記録する detail に対応します。Python 3.11 以上が必要です。

以下は次回リリース向けの開発版の説明です。公開済み `0.0.1` はパッケージ構成のみです。

## インストールと認証

この checkout からインストールします。

```sh
uv pip install .
cp .env.sample .env
```

`.env` の `TYPESAFE_API_KEY` を設定するか、同名の環境変数を設定してください。
明示的な `api_key=` が最優先で、次に環境変数、最後に `.env` の値を使用します。
`.env` は Git 対象外です。`dotenv_path=None` でファイル読み込みを無効にできます。

既定では Python の `len(text)` で文字数を数えます。tokenizer の依存パッケージやダウンロードは不要です。

## 使い方

```python
from jev_reranker import JevReranker

with JevReranker() as reranker:
    results = reranker.rank(
        "赤い惑星と呼ばれるのは？",
        ["パンは小麦粉から作ります。", "Mars is known as the Red Planet."],
        top_k=1,
    )
    print(results)
    # [{"document_index": 1, "score": ..., "text": "Mars is known as the Red Planet."}]
```

`rank()` と `rerank()` は同じ API で、`raw_rerank()` の `results` を返す wrapper です。
`document_index` は元の入力リストの0始まり index。スコア降順、同点は入力順で返します。
重複テキストも独立した候補として扱い、`text` は切り詰め前の原文です。
`return_documents=False` で本文を省略できます。`top_k` は結果の件数を制限し、採点対象は変更しません。
空の文書リストと `top_k=0` は通信なしで空結果を返します。

## 長さの数え方と上限

`document_max_length` の既定値は **4000** です。標準では `len(text)`、つまり Unicode code point 数で数えます（バイト数や画面上の文字幅ではありません）。上限を超える文書は先頭部分を送信し、返却する `text` は原文のままです。

```python
with JevReranker(document_max_length=8000) as reranker:
    results = reranker.rank("query", ["document"])
# document_max_length=None なら文書の切り詰めを無効化
```

**実環境では tokenizer の利用を推奨します。** 特に英語では文字数が token 数より大きくなりやすく、文字数基準だと必要以上に早く文書を切り詰めたり、リクエストを分割したりする可能性があります。
Gemma tokenizer を使う場合は、この checkout の追加依存をインストールします。

```sh
uv pip install '.[tokenizer]'
```

```python
with JevReranker(
    tokenizer="google/embeddinggemma-300m",
    document_max_length=8000,   # この場合は8000 token。省略時は4000 token
    split_state_budget=26000,
    split_request_budget=48000,
) as reranker:
    results = reranker.rank("query", ["document"])
```

初回の採点時に tokenizer だけを取得します。Gemma の利用条件への同意と Hugging Face 認証（`HF_TOKEN` など）が必要になる場合があります。重み・PyTorch・GPU は不要です。既定 Gemma revision は固定され、`split_tokenizer_revision="main"` などで変更できます。別の Hub repo、ローカル tokenizer.json、`encode(text)` / `decode(ids)` を持つオブジェクトにも差し替えられます。独自オブジェクトを使う場合、ライブラリ側の tokenizer extra は不要です。

任意の同期 callable `length_fn: Callable[[str], int]` も受け取れます。例えば UTF-8 のバイト数を基準にできます。

```python
def utf8_length(text: str) -> int:
    return len(text.encode("utf-8"))

with JevReranker(length_fn=utf8_length, document_max_length=8000) as reranker:
    results = reranker.rank("query", ["document"])  # 最大8000 bytes の prefix
```

関数は決定的な非負整数を返すものとし、`tokenizer` と併用しません。worker thread から呼ばれることがあります。文書と JSON 化した state/questions/request の予算判定に同じ関数を使うため、通常の文書以外の文字列も受け取ります。独自関数では文字列の prefix を二分探索して上限内に収めます。長さが単調増加しない関数では、最長の prefix は保証しません。

`split_state_budget`（既定26000）と `split_request_budget`（既定48000）も選んだ計測方法の単位です。query は切り詰めません。文字数・独自関数・Gemma token 数のいずれも Jev の内部 token 数との一致は保証せず、API の context 超過は別途処理します。
旧名 `document_max_tokens` / `split_state_token_budget` / `split_request_token_budget` は互換 alias として受け付けますが、単位は同じく選択した計測方法に従います。

## 非同期 API

内部の HTTP 通信は `httpx.AsyncClient` と `asyncio` で実行します。
async アプリケーションからは次の API を直接 await できます。

```python
import asyncio
from jev_reranker import JevReranker

async def main():
    async with JevReranker(mode="pointwise", max_concurrency=8) as reranker:
        results = await reranker.a_rank("赤い惑星は？", ["火星です。", "金星です。"])
        raw = await reranker.a_raw_rank(
            "赤い惑星は？", ["火星です。", "金星です。"], detail=True,
        )
        # 同じ instance で複数 query を並行処理できる
        batches = await asyncio.gather(
            reranker.a_rank("赤い惑星は？", ["火星です。", "金星です。"]),
            reranker.a_rank("地球の衛星は？", ["月です。", "太陽です。"]),
        )
        print(results, raw["detail"]["usage"], batches)

asyncio.run(main())
```

| 同期 API | 非同期 API | 返り値 |
| --- | --- | --- |
| `rank()` / `rerank()` | `a_rank()` / `a_rerank()` | 順位付き results のリスト |
| `raw_rank()` / `raw_rerank()` | `a_raw_rank()` / `a_raw_rerank()` | results と省略可能な全体 detail |
| `close()` / `with` | `await aclose()` / `async with` | HTTP 接続などの解放 |

同期・非同期で引数と結果の構造は同じです。`top_k`、`return_documents`、`detail` も共通です。
`a_rank()` は `a_raw_rank()` の wrapper、既存の同期 API も同じ非同期採点処理を利用します。

1つの instance は1つのイベントループに所属します。非同期 API は最初の利用時のループに結び付き、
接続プールと Semaphore を再利用します。別の `asyncio.run()` へ持ち回らず、同じループ内で使って閉じてください。
同期 API は instance ごとに1つの専用イベントループスレッドを遅延起動し、複数の同期呼び出し元スレッドから共有できます。
同期用 instance と非同期用 instance は分けてください。稼働中のイベントループ内で同期 API を呼ぶと、
ループをブロックさせずに `ConfigurationError` で非同期 API の利用を案内します。

並行 HTTP 数は instance 全体で制限します。pointwise は worker task、pairwise は上限件数ごとの task、
listwise の分割 chunk は逐次処理です。retry の待機は `asyncio.sleep` を使います。
同期的な tokenizer のロード・計数は `asyncio.to_thread` に退避するため、HTTP 待機用スレッドは作りません。

呼び出しをキャンセルすると、その呼び出し内の子タスクを回収して `asyncio.CancelledError` を伝播します。
HTTP エラー時も子タスクを停止し、従来の `JevError` 系例外を保持します（ExceptionGroup に包みません）。
`aclose()` / `close()` は新しい処理を拒否し、実行中の採点が終了してから所有する接続を閉じます。
先に止めたい処理は、呼び出し側でタスクを cancel してから閉じてください。

## モードと設定

| モード | 処理 | スコア |
| --- | --- | --- |
| `listwise`（既定） | 候補を同じ state で質問し、上限を超える場合は分割 | 文書ごとの Noul 関連確率 |
| `pointwise` | query と1文書ごとに独立した質問 | 文書ごとの Noul 関連確率 |
| `pairwise` | 全ペアを両方向で比較 | 相手に勝つ平均確率。絶対的な関連度ではない |

```python
reranker = JevReranker(
    model="jev-1.13.0",       # 既定 jev-latest。比較実験では固定バージョンを推奨
    mode="listwise",
    document_max_length=4000,  # None で文書の prefix 切り詰めを無効化
    split_state_budget=26000,
    split_request_budget=48000,
    max_concurrency=4,
    timeout=180.0,
    max_retries=8,
)
# 使い終わったら reranker.close()
```

| constructor 引数 | 既定値・意味 |
| --- | --- |
| `model` | 明示値 → `JEV_MODEL`（環境/.env）→ `jev-latest` |
| `api_key`, `api_key_env` | キーの明示値、検索する環境変数名（`TYPESAFE_API_KEY`） |
| `dotenv_path` | `.env`。プロセスの環境変数は変更しない |
| `endpoint` | 明示値 → `TYPESAFE_ENDPOINT` → `https://api.typesafe.ai/v1/systemone`。完全な endpoint URL |
| `max_concurrency` | listwise=4、pointwise/pairwise=20。instance 全体の HTTP 同時実行上限 |
| `timeout`, `max_retries` | 各 HTTP I/O 180秒、初回を除いて最大8 retry |
| `split_tokenizer_name` | 既定 None。`tokenizer` の文字列指定と同等の別名 |
| `split_tokenizer_revision` | 既定 Gemma は固定 commit。他の tokenizer は指定なしなら Hub 既定 revision |
| `tokenizer_max_length` | 65536。渡された tokenizer の `model_max_length` が小さければ自動で引き上げる。既存の大きな値は維持 |
| `tokenizer` | 既定 None。Hub repo、ローカル tokenizer.json または encode/decode オブジェクト |
| `length_fn` | 既定 None（tokenizer 未指定なら `len`）。独自の同期計数関数 |
| `document_max_length` | 4000。None で文書切り詰め無効 |
| `split_state_budget`, `split_request_budget` | 26000 / 48000。選択した計測方法による上限 |
| `instructions`, `criteria` | 質問 template と true/false の判定基準 |
| `client` | 借用する `httpx.AsyncClient`。同じループで使う。ranker は閉じない。同期版 `httpx.Client` は不可 |
| `transport` | `httpx.AsyncBaseTransport`（例: `httpx.MockTransport`）。ranker が所有して閉じる。client との併用不可 |

Gemma の長さ設定は **Jev の入力長を推定するため** のものです。EmbeddingGemma モデル本体の context を拡張しません。
Gemma と Jev の token 数は同一とは限らず、query・question も予算を消費します。
分割後の listwise は異なる候補の文脈で採点するため、分割前と同じスコアになる保証はありません。
pairwise は n(n−1)/2 リクエストを使うため、小さな候補集合での相対比較に向きます。

名前で指定した tokenizer は truncation/padding を自動で無効にし、65536 を超える入力も全体を計数します。オブジェクト指定では書き換え可能な `model_max_length` を `tokenizer_max_length` 以上へ自動調整します（渡したオブジェクト自体を変更）。上限属性のない独自 tokenizer はそのまま使います。これは tokenizer 側の設定であり、文書の送信上限 `document_max_length` は変更しません。

独自 tokenizer は特殊 token を加えず、入力を切り詰めない encoder を用意してください。
Transformers の tokenizer を渡す場合も、このインターフェースに合わせた wrapper で
`add_special_tokens=False, truncation=False` を指定します。

質問を変更する場合、listwise/pointwise は `{document}` を含めます。
pairwise は `{left}` と `{right}` を含めます。これ以外の template フィールドや不明な kwargs はエラーになります。

```python
reranker = JevReranker(
    instructions="Does {document} directly answer `query`?",
    criteria={"true": "Provides the requested facts", "false": "Does not provide the requested facts"},
)
```

## 詳細ログとエラー

```python
import json
from jev_reranker import JevError, JevReranker

with JevReranker() as reranker:
    try:
        raw = reranker.raw_rerank("赤い惑星は？", ["火星です。", "金星です。"], detail=True)
    except JevError as exc:
        # APIError.status_code や、失敗までの exc.detail を調査できる
        raise

with open("rerank-log.json", "w", encoding="utf-8") as file:
    json.dump(raw, file, ensure_ascii=False, indent=2)
```

`raw["detail"]` に実効設定、Python/依存バージョン、要求/解決モデル、tokenizer revision、
usage、retry、分割、送信した state/questions と応答を記録します。
`raw["results"][i]["detail"]` には `original_length` / `sent_length` / `length_unit`、対応 request ID、pairwise の比較結果が入ります。
`rank(..., detail=True)` でも文書別 detail を取得できます。実行全体のログは `raw_rerank` を使ってください。
detail の `schema_version` は2です。`usage.input_tokens` / `output_tokens` は API が返す実 token 数で、ローカルの長さ計測とは別です。
ログには本文が含まれます。API キー・認証ヘッダー・任意の環境変数は保存しません。

429/500/502/503/504/529 と transport error は指数 backoff と jitter で retry し、
`Retry-After` を最大60秒まで尊重します。認証エラーや壊れた成功応答は retry しません。
listwise の context 超過は失敗した候補群だけを再分割します。単一文書/ペアでも収まらなければ
`ContextLimitError` になります。失敗を0点に置き換えることはありません。
その他の公開例外は `ConfigurationError`、`APIError`、`ResponseValidationError`（共通基底 `JevError`）です。

constructor の設定は利用中に書き換えず、別設定には別 instance を作ってください。

## 開発と live E2E

```sh
uv sync --locked --dev
uv run --locked tox
# 明示指定した場合だけ、本物の .env / Jev / Gemma tokenizer で実行
uv run --locked pytest tests/test_live.py --live -q
```

通常の pytest / CI は live テストを skip し、外部通信や API キーを必要としません。
live E2E は日英・中国語・スペイン語・混在言語の各4文書を3モードで採点し、
同期・非同期 API の両方で期待する全順位と厳密なスコア順序を検査します。接続再利用の live 検査も含みます。実行ログは Git 対象外の `.live-results/` に保存します。
成功結果・制約は [live 検証記録](docs/live-validation.md)、詳しい設計は [仕様](docs/spec.md) を参照してください。

uv の依存解決は公開後1週間の cooldown を維持しています。

```sh
uv build --no-sources --clear
uv run --locked twine check --strict dist/*
```

公開は [リリース手順](docs/release.md) に従い、新バージョンを付けた reviewed PR とタグから行います。
[CHANGELOG](CHANGELOG.md) / [次回リリースの変更](docs/releases/HEAD.md)。

## ライセンス

MIT。[LICENSE](LICENSE)、移植元の [権利表示](THIRD_PARTY_NOTICES.md) を参照してください。
