# Repository Guidelines

## Project

- `jev-reranker` is a Python library; the import name is `jev_reranker`.
- Use Python 3.11 or newer and uv. Keep source code under `src/jev_reranker/`.
- The initial `0.0.1` release contains package scaffolding only. Do not describe
  reranking functionality as implemented until it exists.
- Keep runtime dependencies minimal and development tools in the `dev` group.
- Preserve the one-week uv cooldown (`exclude-newer = "1 week"`) and commit
  `uv.lock` with dependency changes.
- Do not commit virtual environments, caches, build artifacts, or credentials.

## Validation

Run the same checks used by CI before releasing:

```sh
uv sync --locked --dev
uv run --locked tox
uv build --no-sources
uv run --locked twine check --strict dist/*
```

Use a clean output directory for release builds so previous versions cannot be
uploaded accidentally. CI also installs the wheel in an isolated environment
and verifies that the package imports. Keep `.github/workflows/ci.yml` aligned
with local validation; it runs on main pushes, pull requests, and manual dispatch.

## Release Flow

- `pyproject.toml` is the source of truth for the package version. Update it with
  `uv version X.Y.Z` and include the resulting `uv.lock` change.
- For subsequent releases, prepare a `release/vX.Y.Z` branch from `origin/main`
  and merge a reviewed PR after CI passes. Merging alone does not publish.
- Create an annotated `vX.Y.Z` tag on the exact merged commit whose CI passed;
  pushing it starts `.github/workflows/release.yml`.
- Never move or reuse a published release tag or PyPI version.
- Release builds reuse the CI workflow, check tag/version agreement, run tox,
  build a wheel and sdist, validate metadata, and test wheel installation.
- Publish through PyPI Trusted Publishing using the `pypi` GitHub Environment.
  Do not add long-lived PyPI tokens to repository secrets.
- After PyPI succeeds, create a GitHub Release and attach the same wheel and
  sdist that were published to PyPI. Keep publishing permissions job-scoped.
- When pinning Actions, use the underlying commit SHA, not an annotated tag
  object's SHA. Verify the actual release workflow as well as normal CI.
- If publishing fails, inspect the logs and PyPI before retrying. If PyPI already
  succeeded, rerun only the failed GitHub Release job, not the upload job.
- Confirm both PyPI installation and GitHub Release assets after publication.
- Keep `docs/release.md` synchronized with the actual workflow.
