# jev-reranker

TypeSafe Jev による、多言語対応の Python reranker。`listwise`、`pointwise`、`pairwise` を選べ、token 分割、retry、実行内容を記録する detail に対応します。Python 3.11 以上が必要です。

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

初回の採点時に `google/embeddinggemma-300m` の **tokenizer だけ** をダウンロードします。
Gemma の利用条件への同意と、Hugging Face のログイン（`hf auth login` / `HF_TOKEN`）が必要になる場合があります。
モデルの重み、PyTorch、GPU は不要です。tokenizer をキャッシュしておけば以後の Hub 通信を避けられます。

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
    # [{"corpus_id": 1, "score": ..., "text": "Mars is known as the Red Planet."}]
```

`rank()` と `rerank()` は同じ API で、`raw_rerank()` の `results` を返す wrapper です。
`corpus_id` は元の入力リストの0始まり index。スコア降順、同点は入力順で返します。
重複テキストも独立した候補として扱い、`text` は切り詰め前の原文です。
`return_documents=False` で本文を省略できます。`top_k` は結果の件数を制限し、採点対象は変更しません。
空の文書リストと `top_k=0` は通信なしで空結果を返します。

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
    document_max_tokens=4000,  # None で文書の prefix 切り詰めを無効化
    split_state_token_budget=26000,
    split_request_token_budget=48000,
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
| `split_tokenizer_name` | `google/embeddinggemma-300m`。Hub repo またはローカル tokenizer.json |
| `split_tokenizer_revision` | 既定 Gemma は固定 commit。他の tokenizer は指定なしなら Hub 既定 revision |
| `tokenizer_max_length` | 65536。計数時は truncation を無効にし、これを超えても全 token を数える |
| `tokenizer` | 任意の `encode(text) -> list[int]` / `decode(tokens) -> str` を持つオブジェクト |
| `instructions`, `criteria` | 質問 template と true/false の判定基準 |
| `client` | テストや独自 transport 用の `httpx.Client`。所有権は呼び出し側に残る |

Gemma の長さ設定は **Jev の入力長を推定するため** のものです。EmbeddingGemma モデル本体の context を拡張しません。
Gemma と Jev の token 数は同一とは限らず、query・question も予算を消費します。
分割後の listwise は異なる候補の文脈で採点するため、分割前と同じスコアになる保証はありません。
pairwise は n(n−1)/2 リクエストを使うため、小さな候補集合での相対比較に向きます。

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
`raw["results"][i]["detail"]` には切り詰め token 数、対応 request ID、pairwise の比較結果が入ります。
`rank(..., detail=True)` でも文書別 detail を取得できます。実行全体のログは `raw_rerank` を使ってください。
ログには本文が含まれます。API キー・認証ヘッダー・任意の環境変数は保存しません。

429/500/502/503/504/529 と transport error は指数 backoff と jitter で retry し、
`Retry-After` を最大60秒まで尊重します。認証エラーや壊れた成功応答は retry しません。
listwise の context 超過は失敗した候補群だけを再分割します。単一文書/ペアでも収まらなければ
`ContextLimitError` になります。失敗を0点に置き換えることはありません。
その他の公開例外は `ConfigurationError`、`APIError`、`ResponseValidationError`（共通基底 `JevError`）です。

API は同期式です。async アプリケーションでは `asyncio.to_thread` 等で呼び出してください。
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
期待する全順位と厳密なスコア順序を検査します。実行ログは Git 対象外の `.live-results/` に保存します。
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
