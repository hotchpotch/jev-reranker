# HEAD

- 開発・検査・リリースの運用を `AGENTS.md` とリリース手順書に整理。
- 開発中の変更を `HEAD.md`、公開済みの変更をバージョン別のリリースログで管理。
- GitHub Release の本文をリリースログから生成し、PyPI と同じ wheel・ソース配布物を添付するフローを整備。

- `JevReranker.rank()` / `rerank()` / `raw_rerank()` を実装。listwise・pointwise・pairwise の採点、同点での入力順維持に対応。
- Gemma tokenizer による token 計数・文書 prefix 切り詰め、listwise の均等分割と API context 上限超過時の再分割を追加。tokenizer・長さ・モデル・質問は変更可能。
- 環境変数/.env の認証、bounded retry、厳密な応答検証、実行詳細と失敗時の診断を追加。
- 通常実行しない多言語 live E2E と offline テストを追加。次回リリースの仕様と使用例を文書化。

- HTTP 通信を httpx.AsyncClient + asyncio に移行。`a_rank()` / `a_raw_rank()` と async context manager / `aclose()` を追加し、既存の同期 API は同じ非同期実装を利用する wrapper として維持。
- 同時実行数の制限、非同期 retry、キャンセル時の子タスク回収、実行中の呼び出しを待つ終了処理に対応。instance のイベントループ所有権を明示的に検査。
- `client=` は httpx.AsyncClient 用に変更。独自非同期 transport を指定する `transport=` を追加。

- 長さ計測の既定を `len(text)`、文書上限を `document_max_length=4000` に変更。独自 `length_fn` と optional tokenizer extra、Gemma の明示指定に対応。分割予算も同じ単位で計測。
- 結果の入力位置を `corpus_id` から `document_index` に改名。detail schema 2 は original_length/sent_length/length_unit を記録し、API の usage token 数と区別。

- tokenizer オブジェクトの `model_max_length` を自動で `tokenizer_max_length`（既定65536）以上に引き上げる。既存のより大きな値は維持し、文書の送信上限とは分離。

- NanoBEIR-en HotPotQA の50 query × 10文書を評価する example を追加。hybrid 順と正解保証による候補選択、seed 42のシャッフル、nDCG@10 と詳細ログに対応。pyarrow は examples dependency group のみに追加。

- HotPotQA example の候補選択を全正解保証に修正。全 qrels 文書を確保し、残りを hybrid 上位で埋める。hybrid 外の正解は corpus から補完。

- HotPotQA example に `--target en/ja` と `--top-k` を追加。既定10件、none/all/値なしで元 hybrid 全候補を補完せず評価。dataset・split 別キャッシュと候補内の正解欠落の記録に対応。

- ターゲット切り替えに合わせ、評価スクリプトを `examples/eval.py` に改名。

- `examples/eval.py` に Sentence Transformers CrossEncoder backend を追加。候補選択・nDCG を共通化し、モデル/device/dtype/最大長を指定可能にした。`sentence-transformers` extra と、tokenizer・CrossEncoder・評価用 pyarrow を含む `all` extra を追加。

- `docs/eval.md` に英語の評価ガイドを追加し、`eval.py --help` に両 backend の実行方法・候補選択・指標の説明を掲載。

- 評価スクリプトの候補数は既定で制限なし（元 hybrid 全候補、正解追加なし）。10件評価は `--top-k 10` で明示指定。
