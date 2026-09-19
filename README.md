# jev-reranker

JEV reranker の Python パッケージ。現在は開発環境とパッケージ構成のみで、reranker の機能は未実装です。

Python 3.11 以上と uv を使用します。PyPI の配布名は `jev-reranker`、Python の import 名は `jev_reranker` です。

## 開発

```sh
uv sync --locked
uv run tox
```

uv は公開後1週間を経過した依存パッケージを解決対象にします（`exclude-newer = "1 week"`）。

tox で Python 3.11 の pytest、ruff、ty を実行します。

## ビルド

```sh
uv build
```

`dist/` に wheel とソース配布物を生成します。

## CI とリリース

GitHub Actions で検査と配布物のビルドを行います。`vX.Y.Z` タグの push で PyPI に公開します。
初回設定と公開手順は [リリース手順](docs/release.md) を参照してください。

## ライセンス

MIT。詳細は [LICENSE](LICENSE) を参照してください。
