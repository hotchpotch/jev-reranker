# Release guide

Pushing a `vX.Y.Z` tag triggers `.github/workflows/release.yml`. After the shared CI checks (pytest, ruff, ty, builds, metadata validation, and wheel installation) pass, the workflow publishes to PyPI and creates a GitHub Release with the same distributions attached.

CI installs the wheel and runtime dependencies in isolation and checks synchronous and asynchronous reranking, relevance prompts, and execution details with a mock HTTP transport and no network requests.

Run local validation before publishing:

```sh
uv sync --locked --dev
uv run --locked tox
uv build --no-sources --clear
uv run --locked twine check --strict dist/*
```

`--clear` prevents older distributions from being included. Run live validation when needed with `uv run --locked pytest tests/test_live.py --live -q`; normal CI excludes it.

## Initial setup

Create a GitHub Environment named `pypi` in the repository. Sign in to PyPI and register a GitHub pending publisher under [Publishing](https://pypi.org/manage/account/publishing/).

| Field | Value |
| --- | --- |
| PyPI Project Name | `jev-reranker` |
| Repository owner | `hotchpotch` |
| Repository name | `jev-reranker` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

No API token or GitHub secret is required. Registering a pending publisher does not reserve the project name; the first successful publication creates the project. See the [PyPI instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Changelog and release notes

During development, record user-visible changes in `docs/releases/HEAD.md`. When preparing a release, move them to `docs/releases/vX.Y.Z.md`, add the release date, and reset `HEAD.md` to only `# HEAD`. Prepend the new version to `CHANGELOG.md`. Include the version, lockfile, and release notes in the same release PR.

```sh
python scripts/release-notes.py vX.Y.Z
```

This command prefers a nonempty `HEAD.md`, then the requested version's notes, then `Release vX.Y.Z` if neither has a body. Before tagging, ensure the final notes exist, `HEAD.md` is reset, and the generated body matches the release. GitHub Releases use this body and the same wheel and source distribution published to PyPI. GitHub Packages does not provide a Python package registry; GitHub-side distributions belong in Releases.

## Publishing

1. Create `release/vX.Y.Z` from the current `origin/main`.
2. Update `pyproject.toml` and `uv.lock` with `uv version X.Y.Z`, finalize release notes, and merge a reviewed PR after CI passes. Confirm the exact merged commit SHA to tag.
3. For the initial release, confirm pending-publisher registration is complete.
4. Create an annotated tag on the verified commit and push it. Substitute the new version and actual SHA in this template:

```sh
git tag -a vX.Y.Z <commit-sha-with-passing-CI> -m 'Release vX.Y.Z'
git push origin vX.Y.Z
```

A mismatch between the tag and `pyproject.toml` fails before publication. Never reuse a published version or tag. If PyPI succeeds but GitHub Release creation fails, rerun only the failed job.

Check [Actions](https://github.com/hotchpotch/jev-reranker/actions), [PyPI](https://pypi.org/project/jev-reranker/), and the GitHub Release assets. To verify a newly published package, explicitly exempt it from this project's one-week cooldown. Set `release_version` to the version being verified:

```sh
release_version=X.Y.Z
uv run --isolated --no-project --exclude-newer-package jev-reranker=false --with "jev-reranker==$release_version" python -c 'import jev_reranker; from importlib.metadata import version; print(version("jev-reranker"))'
```

CI also scores nonempty sync and async inputs through mock HTTP in an isolated wheel environment without the tokenizer extra. It verifies that tokenizer dependencies are absent and that explicit tokenizer use provides extra-installation guidance.
