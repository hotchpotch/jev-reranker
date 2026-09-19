# HEAD

- 開発・検査・リリースの運用を `AGENTS.md` とリリース手順書に整理。
- 開発中の変更を `HEAD.md`、公開済みの変更をバージョン別のリリースログで管理。
- GitHub Release の本文をリリースログから生成し、PyPI と同じ wheel・ソース配布物を添付するフローを整備。

- `JevReranker.rank()` / `rerank()` / `raw_rerank()` を実装。listwise・pointwise・pairwise の採点、同点での入力順維持に対応。
- Gemma tokenizer による token 計数・文書 prefix 切り詰め、listwise の均等分割と API context 上限超過時の再分割を追加。tokenizer・長さ・モデル・質問は変更可能。
- 環境変数/.env の認証、bounded retry、厳密な応答検証、実行詳細と失敗時の診断を追加。
- 通常実行しない多言語 live E2E と offline テストを追加。次回リリースの仕様と使用例を文書化。
