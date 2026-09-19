# リリース手順

`vX.Y.Z` タグの push で `.github/workflows/release.yml` を起動します。
通常 CI と同じ pytest・ruff・ty、ビルド、メタデータ検査、wheel のインストール検査を通過後、
PyPI へ公開し、同じ配布物を添付した GitHub Release を作成します。
現在の `0.0.1` はパッケージ構成のみの初期リリースで、reranker の機能は未実装です。

## 初回設定

GitHub リポジトリに `pypi` Environment を用意します。

PyPI にログインし、[Publishing](https://pypi.org/manage/account/publishing/) で
GitHub の pending publisher を登録します。

| フィールド | 値 |
| --- | --- |
| PyPI Project Name | `jev-reranker` |
| Repository owner | `hotchpotch` |
| Repository name | `jev-reranker` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

API トークンや GitHub Secrets の登録は不要です。
pending publisher の登録だけでは名前は予約されず、初回公開の成功時にプロジェクトが作られます。
詳しくは [PyPI の公式手順](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) を参照してください。

## 公開

1. `pyproject.toml` のバージョンと `uv.lock` を更新します（初回は `0.0.1` 設定済み）。
2. 次回以降は `origin/main` から `release/vX.Y.Z` ブランチを作り、レビューと CI を通過した PR をマージします。タグ対象のマージコミット SHA を確認します。
3. 初回は pending publisher の登録完了を確認します。
4. 確認したコミットにタグを付けて push します。

```sh
git tag -a v0.0.1 <CIが成功したコミットSHA> -m 'Release v0.0.1'
git push origin v0.0.1
```

タグと `pyproject.toml` のバージョンが異なる場合は公開前に失敗します。
一度公開したバージョンやタグは再利用せず、次のバージョンに進めます。
PyPI 公開後に GitHub Release 作成だけが失敗した場合は、失敗したジョブだけを再実行します。

[Actions](https://github.com/hotchpotch/jev-reranker/actions) と
[PyPI](https://pypi.org/project/jev-reranker/) で結果を確認します。
公開直後の検証では、このプロジェクトの1週間 cooldown の対象外として明示的に指定します。

```sh
uv run --isolated --no-project --exclude-newer-package jev-reranker=false --with jev-reranker==0.0.1 python -c 'import jev_reranker; from importlib.metadata import version; print(version("jev-reranker"))'
```
