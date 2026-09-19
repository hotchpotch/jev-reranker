# Live E2E 検証記録

2026-09-19 JST に開発環境の `.env` を使用して実行。
認証値と本文を含む生ログは Git 対象外の `.live-results/` に保存し、この文書には採点結果のみ記録する。

- コマンド: `uv run --locked pytest tests/test_live.py --live -q`
- 結果: **15 passed**（5言語構成 × 3モード）。実行時間19.09秒。
- 要求 model: `jev-latest`、API 解決 model: `jev-1.13.0`。
- tokenizer: `google/embeddinggemma-300m`、revision `57c266a740f537b4dc058e1b0cda161fd15afa75`。
- 4文書を `[部分回答, 無関係, 完全回答, 関連概要]` の順で入力し、全ケースで
  `corpus_id=[2, 0, 3, 1]` かつ4つのスコアに厳密な大小関係があることを確認した。
- mixed は日本語 query に対し、英語の完全回答、中国語の部分回答、スペイン語の概要、日本語の無関係文書を使用。
- listwise/pointwise は Noul、pairwise は両方向比較を対称化した平均勝率。異なる方式間でスコアの絶対値は比較しない。

| 言語構成 | mode | 完全回答 → 部分回答 → 関連概要 → 無関係 | input tokens |
| --- | --- | --- | ---: |
| en | listwise | 0.9800, 0.8200, 0.1100, 0.0100 | 692 |
| en | pointwise | 0.9800, 0.5900, 0.0700, 0.0000 | 1518 |
| en | pairwise | 0.9400, 0.6450, 0.3467, 0.0683 | 2958 |
| ja | listwise | 0.9800, 0.8300, 0.0900, 0.0100 | 812 |
| ja | pointwise | 0.9700, 0.6400, 0.0800, 0.0000 | 1710 |
| ja | pairwise | 0.9117, 0.6583, 0.3400, 0.0900 | 3390 |
| zh | listwise | 0.9800, 0.8200, 0.1000, 0.0100 | 740 |
| zh | pointwise | 0.9700, 0.5100, 0.0700, 0.0000 | 1590 |
| zh | pairwise | 0.9167, 0.6567, 0.3500, 0.0767 | 3126 |
| es | listwise | 0.9800, 0.8500, 0.1100, 0.0100 | 719 |
| es | pointwise | 0.9800, 0.6400, 0.0800, 0.0000 | 1566 |
| es | pairwise | 0.9283, 0.6550, 0.3333, 0.0833 | 3060 |
| mixed | listwise | 0.9800, 0.8200, 0.1100, 0.0100 | 747 |
| mixed | pointwise | 0.9800, 0.5900, 0.0900, 0.0000 | 1645 |
| mixed | pairwise | 0.9217, 0.6550, 0.3367, 0.0867 | 3195 |

これは明確な関連度を持つ4文書の smoke 検証であり、多言語全体の品質保証や benchmark ではない。
実サービスの変更、固定モデルの実行揺れ、候補集合によって結果が変わる可能性がある。
`jev-latest` を使うテストは将来の変更を検出する意図があり、失敗時に期待順位を自動で緩めない。
分割・長文切り詰め・retry・並行実行・壊れた応答の検証は、別途 offline テストで行う。

## asyncio・optional tokenizer 変更後の再検証

2026-09-19 JST、同じ `.env` で上記コマンドを再実行し、**32 passed（35.24秒）**。
5言語構成 × 3モード × 同期/非同期の30ケースは Gemma tokenizer を明示指定し、
全ケースで `document_index=[2, 0, 3, 1]` と厳密なスコア降順を確認した。
残る2ケースは既定の `len(text)` を使い、同期・非同期それぞれで同じ instance の接続再利用を確認した。
新しい detail は schema_version=2、長さを original_length/sent_length/length_unit で記録する。
上の表は変更前の検証記録であり、当時のフィールド名を保持している。

通常検証は121 passed、live 32 skipped。tox の lint/type 検査、clean build、twine strict を通過。
隔離した base wheel 環境で tokenizer 依存が存在しないこと、非空入力の同期・非同期 mock 採点、
明示的な tokenizer 利用時の追加依存案内も確認した。
