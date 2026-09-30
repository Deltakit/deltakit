# Deltakit Release Procedure

A new version of Deltakit is released on a regular basis. Only stable releases are currently supported.

## Stable Releases

Stable releases are performed whenever new features are implemented or bug fixes are made.
The target audience of stable releases is typical users.
Semantic versioning communicates whether releases include any backward compatible changes so that users can
decide when/whether to upgrade.

A stable release currently consists of:
- a version [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) (human-readable label) associated with a commit,
- assets published to PyPI (which gets added to the "Release history" there), and
- assets published to "Releases" on our GitHub repo.
- documentation, currently published using GitHub pages.

"Assets" currently refers to [wheel](https://peps.python.org/pep-0427/)s (pre-built Python package format)
and sdists ("source distribution") of `deltakit` and all component libraries.

A release is accompanied by release notes. Draft release notes are automatically compiled by [Python
Semantic Release](https://python-semantic-release.readthedocs.io/en/latest/) from the commit messages,
and these are manually reviewed and edited before publication.

To perform a release, the release manager needs to go through the following steps:

1. Create a release branch with naming format `release/v<version number>`. On that branch, bump the version in `pyproject.toml` and open a PR targeting `main`. The PR undertakes the review/approval process. Before merge, any final changes merged to `main` during the review process can be added.

2. Create a new release by clicking the `Draft a new release` button on the GitHub [repository](https://github.com/Deltakit/deltakit/releases). Create a new tag and release notes and finally publish the release. This will trigger building and testing distributions and publishing on [PyPI](https://pypi.org/search/?q=deltakit) and deploying latest docs on the [Deltakit website](https://deltakit.riverlane.com/).

The latest stable version of `deltakit` can be installed with `pip install deltakit`.
