# Releasing

Release Please prepares versions and release notes. The **Release** workflow
builds the package, publishes it to [PyPI](https://pypi.org/project/outis/)
through Trusted Publishing, checks that PyPI serves the files it built, and only
then publishes the Git tag and GitHub release. Do not upload releases from a
workstation.

## Preparing a release

Release Please reads Conventional Commit messages on `main`. Pull request
titles become squash commit messages, so title them:

- `fix: ...` proposes a patch release.
- `feat: ...` proposes a minor release.
- `feat!: ...`, `fix!: ...` or a `BREAKING CHANGE:` footer proposes a breaking
  release. Before 1.0, breaking releases increment the minor version.

Other types (`docs:`, `chore:`, `test:`) and untyped messages do not create a
release by themselves.

After a releasable commit reaches `main`, the workflow creates or updates one
Release Please pull request. It updates `pyproject.toml`, `uv.lock`,
`CHANGELOG.md` and `.release-please-manifest.json`. Pull requests opened with
the workflow token do not start other workflows, so the Release workflow
dispatches CI on that branch. Review the proposed version and notes like any
other pull request. Do not change these version files by hand.

## Publishing a release

1. Merge the Release Please pull request after its checks pass.
2. Release Please creates a draft GitHub release, and the workflow builds the
   merge commit.
3. Approve the `pypi` environment deployment if it requires a reviewer.
4. The workflow uploads the wheel and sdist to PyPI and waits until PyPI serves
   files with the same SHA-256 digests.
5. It then attaches the distributions and `SHA256SUMS` to the GitHub release
   and publishes it, which creates the tag.

## First release and recovery

A release pull request is only proposed once the current version is tagged.
The first release, 0.1.0, is therefore published by hand: run the **Release**
workflow from `main` with version `0.1.0`. Use the same input to recover a
release whose publish or finalize job failed. The workflow builds the head of
`main`, which must carry that version; if PyPI already holds the version, it
must hold exactly the rebuilt files, otherwise the run fails.

## One-time setup

1. On PyPI, add a pending Trusted Publisher (Account → Publishing): project
   `outis`, owner `ewhauser`, repository `outis`, workflow `release.yml`,
   environment `pypi`. The first upload creates the project.
2. In the repository, create the `pypi` environment (Settings → Environments),
   optionally with required reviewers, and limit it to the `main` branch.
3. Allow GitHub Actions to create pull requests (Settings → Actions → General →
   Workflow permissions).
