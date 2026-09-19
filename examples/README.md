# NanoBEIR-en / HotPotQA の50 query × 10 docs 評価

このリポジトリの root から実行します。`.env` の `TYPESAFE_API_KEY` を使用します（既存の環境変数が優先）。実 API を呼ぶため利用料金が発生します。

```sh
uv run --locked --group examples --extra tokenizer python examples/hotpotqa.py
```

`pyarrow` は examples dependency group のみに入り、ライブラリの runtime 依存には追加しません。スクリプトは自身の位置から `src/` を import path の先頭に置き、実際の import 先も検査します。公開済みの別バージョンや HAKARI のライブラリは使いません。

既定は listwise、Gemma tokenizer、文書上限4000 token、同時実行数4、seed 42。モデルは `JEV_MODEL`（環境変数/.env）、なければ `jev-latest` です。変更例:

```sh
uv run --locked --group examples --extra tokenizer python examples/hotpotqa.py \
  --model jev-1.13.0 --mode pointwise --document-max-length 8000 --concurrency 4
# tokenizer を使わず文字数で計測する場合: --tokenizer none
```

## データ選択と指標

移植元 HAKARI の `scripts/build_reranking_hybrid_nano_dataset.py` と `hakari_bench/metrics.py` を確認して実装しました。
[公開データ](https://huggingface.co/datasets/hakari-bench/NanoBEIR-en/tree/d3962aa8efe48ed79044c5e155b848982667b4ba) の `NanoHotpotQA` split を使用します。既定 revision は `d3962aa8efe48ed79044c5e155b848982667b4ba`。`--revision` で変更可能ですが、再現性のため commit SHA を指定してください。

1. queries 50件、corpus、qrels、reranking_hybrid を読み込みます。parquet は `.cache/hotpotqa/` に保存します。
2. hybrid は BM25 と dense（Harrier）を RRF で統合した上位100件、必要なら正解を末尾に追加した101件です。保存形式は順位付き文書IDの配列であり数値スコアはありません。この保存順位をスコア順として利用します。
3. qrels の全正解を必ず含め、残りの枠は hybrid 上位の非正解で埋めて10件にします。選択後も hybrid 順を維持し、hybrid 外の正解は corpus から補完して文書ID順で末尾へ追加します。正解が0件または10件超・候補重複・件数不足・本文欠落はエラーとし、query を黙って除外しません。
4. query ID をソートし、1つの `random.Random(42)` で各 query の10件を順次シャッフルします。正解ラベルは Jev に送りません。
5. `a_raw_rank(..., detail=True)` の `document_index` を入力文書IDへ戻し、query ごとの nDCG@10 を計算して50件の算術平均を取ります。

qrels は二値です。DCG は関連文書について `1/log2(rank+1)` を合計します。主指標 **ndcg_at_10** の IDCG は、その query の全 qrels を使います（HAKARI と同じ定義）。全正解を必ず10件に含めるため、正解2件を1・2位にすれば1.0です。

補助指標 **candidate_ndcg_at_10** は選択した10件内の正解のみで正規化します。全正解保証により主指標と必ず一致し、oracle は1.0になります。hybrid/shuffled 比較値と oracle（選択10件を完全に並べ替えた上限）は全 qrels 基準です。正解保証のため qrels を使って候補を選ぶ評価であり、元の100件 rerank や通常の検索ベンチマークとは直接比較しません。

`.live-results/hotpotqa-<UTC>/` に選択ID・シャッフル順・データハッシュの `manifest.json`、全50件の採点 detail、集計 `summary.json` を保存します。出力先は `--output` で変更でき、既存ディレクトリへの上書きは拒否します。失敗時は成功済みの詳細を残し、成功扱いの summary は作りません。ログには本文を含み、Git 対象外です。

## テスト

ファイル内に offline テストを同梱しています。実 API やデータ取得なしで、正解補完・順位維持・seed 42の順序・nDCG の手計算値・50 query の HTTP mock 実行とID対応を確認できます。通常の tox にも組み込んでいます。

```sh
uv run --locked pytest examples/hotpotqa.py -q
```

## 旧方式の実測（2026-09-19 JST、最低1件保証）

以下は修正前の参考記録です。現在は全正解保証に変更しており、現在の結果は次節に記載します。

上記既定コマンド相当で、このプロジェクトの `.env` を使用。Gemma tokenizer、listwise、解決モデル `jev-1.13.0`。全50 query、各10文書を完走しました。

| 指標 | 値 |
| --- | ---: |
| Jev nDCG@10（全 qrels） | **0.922629** |
| 選択10件の hybrid 順 nDCG@10 | 0.839609 |
| シャッフル順 nDCG@10 | 0.467036 |
| 選択10件の oracle nDCG@10 | 0.930366 |
| Jev nDCG@10（10件内正解で正規化） | **0.989407** |

正解補完は2 query。API request 50回、retry 0回、入力98,835 token、出力9,200 token。
実行部は4.85秒（データ取得・候補準備を除き、tokenizer の遅延ロード・ログ保存を含む）。最長文書は Gemma 計数で341 token、文書の切り詰めはありませんでした。
生ログ: `.live-results/hotpotqa-20260919T090729680568Z/`。
同じ候補・seed でも API の結果や速度は変動し得ます。


## 全正解保証での再計測（2026-09-19 JST）

`.env` を使い、同じ dataset revision・seed 42・listwise・Gemma で再実行。
全50 query で **正解2件を両方含む10 docs** になっていることを manifest で検査しました。

| 指標 | 値 |
| --- | ---: |
| Jev nDCG@10 | **0.995183** |
| 選択10件の hybrid 順 nDCG@10 | 0.871805 |
| シャッフル順 nDCG@10 | 0.516081 |
| oracle nDCG@10 | 1.000000 |

全正解基準と10件内基準は全 query で一致。47/50 query で nDCG=1.0、残る3件は0.919721。
正解補完が必要だった query は9件。実行部4.18秒。
生ログ: `.live-results/hotpotqa-20260919T094249681443Z/`。前節とは候補集合が異なるため、同一条件のスコア改善とは解釈しません。
