# Releasing Dion

Dion publishes to [PyPI](https://pypi.org/p/dion) from
`.github/workflows/release.yml` when a `v*` tag is pushed. Publishing uses PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API token
is stored in this repository.

## One-time setup

All three steps need a human with the right accounts; none can be done from a PR.

**1. Enable GitHub Actions for workflows in this repository.** As of this
writing no workflow defined in `.github/` has ever run here — the only entries
under *Actions* are the org's managed ones (CodeQL default setup, Copilot
review, Dependabot), and `release.yml` did not trigger on the pull request that
added it. Reading the setting needs admin, so check *Settings → Actions →
General* and confirm that actions are allowed and that `actions/*` and
`pypa/gh-action-pypi-publish` are permitted by the allow-list, if one is in
force. Without this neither half of the workflow runs: no pre-tag validation,
and no publish.

**2. Register the pending publisher on PyPI.** The `dion` name is unclaimed, so
the first upload creates the project. Log in to pypi.org with the account that
should own it, go to *Your projects → Publishing → Add a new pending publisher*,
and enter:

| Field | Value |
| --- | --- |
| PyPI Project Name | `dion` |
| Owner | `microsoft` |
| Repository name | `dion` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

A pending publisher is what lets the workflow create a project that does not
exist yet. Once the first release lands it becomes an ordinary trusted publisher.

**3. Create the `pypi` environment.** In *Settings → Environments*, add an
environment named `pypi`. The name must match the workflow and the pending
publisher exactly. Adding required reviewers here puts a human approval in front
of every upload, which is worth doing on a public package.

## Cutting a release

1. Bump `version` in `setup.py`. Versions below `1.0` are pre-release: breaking
   changes are allowed, but a released version number can never be reused.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading.
3. Open a PR with both changes and merge it. The `Release` workflow builds and
   validates the distributions on that PR, so packaging breakage surfaces before
   the tag exists.
4. Tag the merge commit and push:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

5. The workflow rebuilds, verifies the tag matches the version baked into the
   built distributions, and publishes. If the `pypi` environment requires
   reviewers, approve the run.

## Notes

- **A version is permanent.** PyPI does not allow reuse of a version number or
  of a distribution filename, even after deletion. A bad release is yanked and
  superseded, never replaced. Test on TestPyPI first if a release is unusual:
  configure a second pending publisher at test.pypi.org and run the workflow
  against it, or upload once by hand with
  `twine upload -r testpypi dist/*`.
- **`twine check` is not a metadata check.** It renders the long description and
  little else. The fields PyPI actually validates on upload — `author_email`
  above all — are covered by `tests/test_packaging.py`, which runs in CI.
- **The sdist has to carry `requirements_*.txt`.** `setup.py` reads them at build
  time to populate `install_requires`; if `MANIFEST.in` stops shipping them, a
  build from the sdist still succeeds but declares no dependencies at all.
  `tests/test_packaging.py` asserts against the built artifacts to catch this.
- **`python -m build` is spelled without flags on purpose.** With no arguments it
  builds the sdist from the source tree and then the wheel *from that sdist*,
  which is the round trip that exposes the failure above. `python -m build
  --sdist --wheel` builds both from the source tree and would leave the wheel
  assertions passing over a broken sdist.
- **The workflow pins its actions to commit SHAs.** The publish job holds an
  OIDC identity that PyPI trusts to upload as `dion`, so it must not run code
  fetched from a mutable tag or branch in someone else's repository —
  `pypa/gh-action-pypi-publish@release/v1` is a branch. `.github/dependabot.yml`
  raises PRs to move the pins forward; the trailing `# vX.Y.Z` comment on each
  is what it reads to know the current version.
- **A tag is enough to publish.** Anyone who can push a `v*` tag — or run the
  workflow manually against one — reaches the upload step. Required reviewers on
  the `pypi` environment are what stands between that and PyPI.
