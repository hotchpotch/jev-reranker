# Repository Guidelines

## Project

- `jev-reranker` is a Python library; the import name is `jev_reranker`.
- Use Python 3.11 or newer and uv. Keep source code under `src/jev_reranker/`.
- The initial `0.0.1` release contains package scaffolding only. Distinguish that historical
  release from the implemented functionality in the development branch.
- Keep runtime dependencies minimal and development tools in the `dev` group.
- Preserve the one-week uv cooldown (`exclude-newer = "1 week"`) and commit
  `uv.lock` with dependency changes.
- Do not commit virtual environments, caches, build artifacts, or credentials.

## Public Documentation

- Write all documentation committed to Git in English, including `AGENTS.md`,
  README files, specifications, guides, release notes, and explanatory comments
  and docstrings in examples.
- Preserve intentional multilingual test data and input fixtures.
- Write for readers outside the development environment. Do not include personal
  absolute paths, private hostnames, Tailscale URLs, or links to unpublished logs.
  Repository-relative paths documenting actual inputs and outputs are appropriate.
- Lead README with evidence selection for RAG, followed by runnable usage examples.
  Keep detailed API reference material inside collapsed `<details>` blocks.
- Use the canonical `rerank()` / `relevance_rerank()` names and their `a_` async
  equivalents. Examples consume the `results` response envelope and rely on
  automatic per-call HTTP cleanup.
- Describe the current API in usage guides. Keep historical release limitations
  in versioned release notes, not in the getting-started path.
- Keep evaluation documentation focused on running the script and interpreting
  metrics. Publish measured benchmark results separately, not in README or guides.
- Use supported behavior and reproducible commands; do not turn model-specific
  observations into general quality, retention, latency, or cost guarantees.

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

## Changelog and Release Notes

- Record user-visible changes in `docs/releases/HEAD.md` during development.
  Describe actual behavior and limitations, not implementation work logs.
- When preparing a release, move the draft into `docs/releases/vX.Y.Z.md`, add
  its release date, and reset `HEAD.md` to only `# HEAD`. Update `CHANGELOG.md`
  with the new version first. Commit these together with the version bump.
- Keep historical release logs accurate; do not include later changes in an
  already released version's notes.
- `python scripts/release-notes.py vX.Y.Z` generates the GitHub Release body.
  Use a nonempty `HEAD.md` first,
  then the versioned file, then `Release vX.Y.Z` as a fallback. Preserve Markdown
  paragraph spacing. Keep `tests/test_release_notes.py` aligned with this rule.
- Before tagging, ensure `HEAD.md` is reset and the versioned release log is
  complete. Preview the generated text to avoid publishing next-release drafts.
- GitHub Releases must contain the generated notes and the same wheel and sdist
  uploaded to PyPI. GitHub Packages has no Python package registry; do not add a
  container publication solely to populate the repository's Packages section.
