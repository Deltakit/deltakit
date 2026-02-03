# Contributing to Deltakit

```{toctree}
:hidden:

CODE_OF_CONDUCT
RELEASE
SECURITY
```

Thank you for considering a contribution to Deltakit. We welcome contributions of many forms from anyone, most of which don't even require writing code. The sections below describe key definitions for contribution types and recommended processes to help guide you through contributing.

> For introductory information about contributing to open source projects, please see the [Scientific Python Contributor Guide](https://learn.scientific-python.org/contributors/).

Prior to making a contribution, we kindly ask to review our [code of conduct](CODE_OF_CONDUCT.md).

## Reporting an issue

Issues contributions concern reporting a behavioural discrepancy in the code base (bug), an enhancement suggestion or a constructive participation to an existing issue. Please check the [issue tracker](https://github.com/Deltakit/deltakit/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) first to see if a similar report has already been submitted. If so, add a comment with your current observation and details. Otherwise, create a new issue report:

- **Bug reports**: A "bug" is defined as a discrepancy between documented and actual behaviour or an *inaccurate* error message. Bug reports can be created [here](https://github.com/Deltakit/deltakit/issues/new?template=bug.md).
- **Enhancement and feature requests**: Requests for improvements, enhancements, or new features are highly appreciated. Request reports can be created [here](https://github.com/Deltakit/deltakit/issues/new?template=request.md).
- **Issue participation**: It is also possible to constructively participate in current [issues](https://github.com/Deltakit/deltakit/issues/) by reproducing bugs, investigating their causes, or contributing to discussions on best fixes and implementation designs.

## Submitting a Pull Request

Issues can be resolved by submitting a Pull Request (PR). The recommended workflow is to first fork the main branch using the GitHub interface (via the Fork button in the top-right corner). This creates a new repository under your GitHub account, prefixed with your GitHub handle.Resolving an issue is possible by submitting a Pull Request (PR). The recommended process is to fork the `main` branch using GitHub interface button on the top-right corner. This will create a new GitHub repository prefixed with your GitHub handle. 

Clone your fork locally:

```sh
git clone https://github.com/<GITHUB_HANDLE>/deltakit.git
```

Navigate to the cloned repository and add the original Deltakit repository as the `upstream` remote:

```sh
git remote add upstream https://github.com/Deltakit/deltakit.git
```

After completing your development work, push your changes to your fork and open a Pull Request against the upstream repository for review by the code owners. For the best contribution experience, we recommend setting up your local repository in development mode and following the development guidelines outlined below.

## Setup `deltakit` in development mode

We recommend using [`uv`](https://docs.astral.sh/uv/) as the project manager. To synchronise the project dependencies with your environment, simply run:

```sh
uv sync
```

in your local repository. `uv` also allows you to configure and run tasks via [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) and command-line interfaces as illustrated next.

### Executing tests

The complete `deltakit` test suite can be run by following these steps:

1. Sync the environment using a supported Python version and include the `test` dependency group:

```sh
uv sync --all-packages --python 3.13 --resolution lowest-direct --group test
```

The [`resolution`](https://docs.astral.sh/uv/concepts/resolution/) option specifies the [strategy](https://docs.astral.sh/uv/concepts/resolution/#resolution-strategy) used to install the lowest compatible versions of all dependencies in the group.

2. Execute the tests using the [Pytest](https://docs.pytest.org/en/stable/) framework:

```sh
uv run --group test pytest
```

### Building documentation

Similarly, the documentation can be built locally following the steps:

1. First, sync the environment with the `docs` dependency group:

```sh
uv sync --python 3.13 --resolution highest --group docs
```

2. Then build the HTML documentation:

```sh
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

The generated documentation can then be viewed in any web browser.

### Pre-commit

[`pre-commit`](https://pre-commit.com/) is configured to run a set of common checks before each commit. To enable it, run the following command in your local repository:

```sh
uv run pre-commit install
```

## General guidelines for code development

`deltakit` follows the principles of [semantic versioning](https://semver.org/) and uses [semantic release](https://python-semantic-release.readthedocs.io/en/latest/) to automate its release process. Releases are driven by structured issue and PR titles, as well as commit messages, which are parsed according to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Issue and PR titles - commit messages

Please format issue and PR titles, as well as commit messages, in accordance with the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification, with the following exceptions:

- Use `bug` as the prefix for bug report issues; the `fix` prefix is reserved for PRs and commits that resolve a bug.
- Use `request` as the prefix for requests; more specific prefixes such as `feat` or `perf` should be used for the corresponding PRs and commits.

Additional commit types recognised by the project include `release` (for changes related to release tooling or the release process) and `dev` (for development work that does not fit another category). These types are generally intended for use by package maintainers.

### Rebasing and force-Pushing

During PR review, please avoid rebasing or force-pushing to your branch. If changes are needed, feel free to add revert or merge commits and always use regular pushes. This allows reviewers to easily track what has changed since their last review.

Once the PR has been approved but before it is merged, you may perform an interactive rebase (for example, to clean up the commit history), or code owners can merge the PR using a squash merge.

### Inline comment and suggested change resolution

PR contributors are invited to leave inline review comments unresolved so that reviewers can verify their feedback has been addressed. Reviewers are expected to resolve their own comments once you’ve confirmed the changes.

An exception applies when using GitHub’s “Add a suggestion” feature. Contributors are encouraged to use the ["Add suggestion to batch" and "Commit suggestions"](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request) options; comments associated with committed suggestions will be resolved automatically.

### License and use of artificial intelligence

This project is licensed under the [Apache 2.0 license](https://github.com/Deltakit/deltakit/blob/main/LICENSE). Please ensure that all contributions are compatible with this license. If you are unsure, we encourage you to open an issue to ask for clarification before submitting any content that may not be license-compatible.

Please also note that large language models (LLMs) may be trained on, and can sometimes reproduce, material that is incompatible with our license. If an LLM or similar tool influenced your contribution, please describe how it was used so we can verify license compliance.

### Continuous integration (CI) usage

Please use our CI services responsibly. Since CI re-runs on every push to a pull request branch,
please avoid repeated pushes of small commits.

### Minimum supported dependencies

The project follows [SPEC 0](https://scientific-python.org/specs/spec-0000/). Roughly, this
means that we will support Python versions for three years and other core dependencies for
two years.

### Decision making and governance

Decisions are made by consensus of participants in a GitHub issue or PR. In case of disagreement,
[code owners](https://github.com/Deltakit/deltakit/blob/main/CODEOWNERS) have final authority.

### Release

For more information about release processes, see the [Deltakit release procedure](RELEASE.md).

### Security

For more information about security, see the [Deltakit security policy](SECURITY.md).

### Contributor License Agreement

First-time contributors will be asked to agree to a CLA. This is automated using a GitHub
app for open PR.
