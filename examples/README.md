# NanoBEIR-en/ja / HotPotQA の50 query 評価

English guide: [Evaluation and example script usage](../docs/eval.md).

このリポジトリの root から実行します。`.env` の `TYPESAFE_API_KEY` を使用します（既存の環境変数が優先）。実 API を呼ぶため利用料金が発生します。

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py
```

`pyarrow` は examples dependency group のみに入り、ライブラリの runtime 依存には追加しません。スクリプトは自身の位置から `src/` を import path の先頭に置き、実際の import 先も検査します。公開済みの別バージョンや HAKARI のライブラリは使いません。

既定は候補数制限なし（hybrid 全100/101件）、listwise、Gemma tokenizer、文書上限4000 token、同時実行数4、seed 42。モデルは `JEV_MODEL`（環境変数/.env）、なければ `jev-latest` です。変更例:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --model jev-1.13.0 --mode pointwise --document-max-length 8000 --concurrency 4
# tokenizer を使わず文字数で計測する場合: --tokenizer none
```

## Sentence Transformers での評価

同じスクリプトで `--backend sentence-transformers` を指定します。候補選択・seed 42のシャッフル・nDCG は Jev と共通です。Jev API や認証キーは使いません。

```sh
uv run --locked --extra all python examples/eval.py --backend sentence-transformers \
  --model BAAI/bge-reranker-v2-m3 \
  --model-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --device cuda:1 --dtype float16 --target en --top-k 10
```

`all` extra は tokenizer 関連・Sentence Transformers/PyTorch・pyarrow を含みます。次版公開後の導入は `uv add 'jev-reranker[all]'`。通常インストールには追加しません。個別の `sentence-transformers` extra もあります。

[CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html) の `predict()` で query-document ペアを採点し、raw logit 降順（同点は入力順）に並べます。1ペアに1スコアを返す reranker が対象です。各 query の処理は直列、query 内は `--batch-size`（既定16）で batch 化します。`--device` 未指定ならライブラリが選択、`--dtype` は既定 float32。`--max-length` は query と document を合わせた token 上限で、未指定ならモデルの上限を使います。超過時はモデル tokenizer が切り詰め、元のペア長と切り詰め件数を detail に残します。

`--mode` / `--tokenizer` / `--document-max-length` は Jev 専用で、CrossEncoder には適用しません。CrossEncoder はモデル自身の tokenizer を使います。`--model-revision` はモデル、`--revision` は評価データの revision です。モデル・device・dtype・各依存バージョン・解決モデル revision をログに記録します。

## ターゲットと候補数の変更

```sh
# 日本語、全正解を含む10件
uv run --locked --group examples --extra tokenizer python examples/eval.py --target ja --top-k 10

# 日本語、hybrid 全候補（100/101件）。正解の追加も行わない
uv run --locked --group examples --extra tokenizer python examples/eval.py --target ja --top-k none

# 英語、全正解を含む20件
uv run --locked --group examples --extra tokenizer python examples/eval.py --target en --top-k 20
```

`--top-k` の省略時は件数無制限です。10件に絞る場合は `--top-k 10` を指定します。`--top-k none` / `--top-k all` / 値を付けない `--top-k` は件数無制限です。ここでの top-k は **採点する候補数** で、評価の cutoff は常に nDCG@10です。

`--target en`（既定）/ `ja` は各 dataset の固定 commit を選びます。日本語は `hakari-bench/NanoBEIR-ja`、revision `5c1d5564643f9ca7a8c275688acf09fd940aa5f2`。両方とも既定 split は `NanoHotpotQA`。`--split` で変更可能です。独自の `--dataset` を指定する場合は `--revision` も指定してください。対応する入力形式は queries 50件、hybrid 100/101件、二値 qrels の parquet です。キャッシュは dataset・split・revision ごとに分離します。

件数制限なしでは、元の hybrid 候補に正解が欠けていても **追加・削除しません**。シャッフルのみ行います。したがって全正解保証の件数指定モードとは候補の作り方が異なり、全 qrels 基準と候補内基準の nDCG は一致しない場合があります。`queries_missing_positives`、query ごとの `missing_positive_ids`、候補数の最小・最大を記録します。以下の全正解保証の説明は件数指定モードに適用します。

BGE/TEI 比較スクリプトも manifest の dataset・split・候補数に追従します。同じ候補と順序で比較するには:

```sh
uv run --locked --group examples python tmp/hotpotqa_bge_tei.py \
  --jev-run .live-results/hotpotqa-20260919T095218446825Z
```

`tmp/` の比較スクリプトと結果はローカル実験用で Git 対象外です。

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
uv run --locked pytest examples/eval.py -q
```

## 旧方式の実測（2026-09-19 JST、最低1件保証）

以下は修正前の参考記録です。現在は全正解保証に変更しており、現在の結果は次節に記載します。

当時の既定（10件）で、このプロジェクトの `.env` を使用。Gemma tokenizer、listwise、解決モデル `jev-1.13.0`。全50 query、各10文書を完走しました。

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


## NanoBEIR-ja の実測（2026-09-19 JST）

各条件50 query、seed 42。Jev は listwise＋Gemma、解決モデル jev-1.13.0。
BGE は GPU 0 の TEI にある BAAI/bge-reranker-v2-m3（float16、revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e）。同じ manifest の文書とシャッフル順を使用。

| 候補条件 | Jev nDCG@10 | BGE nDCG@10 |
| --- | ---: | ---: |
| 全正解保証10件 | 0.972851 | 0.904432 |
| 元 hybrid 全100/101件 | 0.898227 | 0.826311 |

全候補では4 query に一部正解の欠落があり、oracle nDCG@10 は0.969052。
10件モードは全 query に全正解を含み oracle=1.0。

- Jev 10件: `.live-results/hotpotqa-20260919T095137416694Z/`
- Jev 全候補: `.live-results/hotpotqa-20260919T095218446825Z/`
- BGE 10件: `tmp/hotpotqa-bge-20260919T095234230606Z.json`
- BGE 全候補: `tmp/hotpotqa-bge-20260919T095234754751Z.json`


## Sentence Transformers / BGE 英語10件の実測（2026-09-19 JST）

全50 query、各10文書、全正解2件保証、seed 42。過去の Jev 実測と全 query の候補・入力順が同じことを manifest で確認しました。

| 実装 | nDCG@10 | 正解2件が上位2位に入った query |
| --- | ---: | ---: |
| Jev listwise（既存結果） | 0.995183 | 47/50 |
| Sentence Transformers / BGE | 0.978154 | 42/50 |
| TEI / BGE（既存結果） | 0.978154 | 42/50 |

BGE revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`、GPU 1、float16、batch size 16、最大長8192。実際の最大ペア長399 token、切り詰め0件。Sentence Transformers 5.7.0 / PyTorch 2.14.0 / Transformers 5.17.0。
全体9.92秒（モデルロード・前処理・ログ保存を含み、データ取得は除外）。モデルロード3.61秒。TEI の常駐サーバー計測と時間の範囲が異なるため、速度を直接比較しません。
生ログ: `.live-results/hotpotqa-20260919T095830815933Z/`。
