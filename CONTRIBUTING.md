# Contributing

```{toctree}
:hidden:

CODE_OF_CONDUCT
release
```

Thank you for considering a contribution to Deltakit!
We accept many types of contributions (most of which don't even require
writing code!) from anyone.

## Types of Contributions
### Issues
#### Bug reports
We define a "bug" as a discrepancy between documented and actual behavior or
an *inaccurate* error message. (If it's not a "bug", see {ref}`contributing-enhancement-requests`.)
First, check to see if the bug has already been reported on the
[issue tracker](https://github.com/Deltakit/deltakit/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug).
If so, leave a comment; if not, create a
[new bug report](https://github.com/Deltakit/deltakit/issues/new?template=bug.md).

(contributing-enhancement-requests)=

#### Enhancement Requests
Other requests (besides bug reports) are also welcome!
For instance, if the documentation needs improvement, if you disagree with
documented behavior, or if you are asking for a new feature, we'd appreciate
your thoughts.
First, check to see if a similar request already has an
[open issue](https://github.com/Deltakit/deltakit/issues?q=is%3Aissue%20state%3Aopen%20label%3Arequest).
If so, leave a comment; if not, create a
[new request](https://github.com/Deltakit/deltakit/issues/new?template=request.md).

#### Issue Participation
Posting an issue is the much-appreciated first step, but there's lots more to do.
Can you try reproducing a bug or finding out more about why it occurs? Can you
help us reach consensus on the appropriate action to take to fix a bug or respond
to a request? We welcome constructive participation in [issues](https://github.com/Deltakit/deltakit/issues/)
that look interesting to you.

### Pull Requests

#### Bug Fixes
Known bugs can be found on the [issue tracker](https://github.com/Deltakit/deltakit/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug).
After reading the bug report carefully and reaching consensus on the appropriate fix,
feel free to open a PR with the agreed-upon fix.

#### Enhancements
Existing enhancement request can be found on the [issue tracker](https://github.com/Deltakit/deltakit/issues?q=is%3Aissue%20state%3Aopen%20label%3Arequest).
After reading the request carefully and reaching consensus on the appropriate action,
feel free to open a PR with a fix.
Note that many enhancements don't require writing any code - documentation improvements
are also appreciated!

#### PR Review
All PRs need review before they can be merged. Please share your expertise in
[PRs](https://github.com/Deltakit/deltakit/pulls)
that are up your alley.

### Other
#### Ask / Answer Questions
Have a question about usage? Knowledgeable about our software and want to share your expertise?
Please ask and answer usage questions on our [Q&A Discussion](https://github.com/Deltakit/deltakit/discussions/categories/q-a).

#### Social Media / Graphic Design / Fundraising
We don't currently have recommendations about these types of contributions, but
if you have ideas, please [contact us](mailto:deltakit@riverlane.com).

## Procedures

### Workflow
For introductory information about contributing to open source (e.g. using GitHub, `git`),
please see the [Scientific Python Contributor Guide](https://learn.scientific-python.org/contributors/).

### Development Environment and Common Tasks
We recommend that contributors use [`pixi`](https://prefix.dev/) to manage their development
environment and run tasks.

After cloning the repository and [installing `pixi`](https://pixi.sh/latest/), navigate to
the root directory of the repository in a terminal and install dependencies with `pixi install`.
You can then activate the Python environment with `pixi shell`.
To deactivate the environment, run `exit`.
To set up the Python interpreter in VS Code, you can set the `python.defaultInterpreterPath`
variable to `"${workspaceFolder}/.pixi/envs/default/bin/python"` in `settings.json`.

```{dropdown} Linux/macOS users...
Depending on system settings, you may experience a `Too many open files (os error 24) at path...`
error. This is [known issue](https://github.com/prefix-dev/pixi/issues/2626) that can easily be
resolved by increasing the maximum number of open file descriptors; e.g., `ulimit -n 512`.
```

`pixi shell` activates a development virtual environment with editable installs of Deltakit
packages so you can make changes and interact with the modified code. This environment
is also available in [several popular IDEs](https://pixi.sh/dev/integration/editor/vscode/).

You can also perform important tasks with `pixi run`. For example:

For instance:

```python3
pixi run tests deltakit-circuit
pixi run lint deltakit-explorer
pixi run docs
```

`pixi` will automatically update or install the dependencies (`pixi install`) and perform the
task as defined in [`pixi.toml`](https://github.com/Deltakit/deltakit/blob/main/pixi.toml). Tasks include:

- `lint <package>`: run `ruff check` on a package
- `mypy <package>`: run `mypy` on a package
- `tests <package>`: run tests on a package
- `testmon`: run only changed tests / tests of changed code (after complete initial run)
- `tests <package> --slow`: run *all* tests on a package, including slow tests
- `docs`: build documentation, serve documentation, and open in browser.
- `build_docs`: build documentation (only)
- `licenses <package>`: run `pip-licenses` to generate a list of PyPI dependency licenses
- `ochrona`/`pip-audit`: check your installed environment for package versions with known security issues
- `bandit`: perform static code analysis for security concerns with `bandit`
- `check_pyproject <package>`: to validate `pyproject.toml`
- `check_workflows`: to validate GHA workflows
- `build <package>`: build wheels and sdist to `dist` directory
- `spellcheck`: to look for common misspellings in Python code
- `vale`: to look for documentation style (including spelling) issues in documentation files

where `<package>` is the optional, hyphenated name of the desired Deltakit
package.

`pre-commit` has been configured to perform several common tests before each commit.
To enable `pre-commit`, run `pre-commit install` within a `pixi shell` (or
`pixi run pre-commit install` otherwise).

### Code of Conduct
When contributing, always follow our [code of conduct](CODE_OF_CONDUCT.md).

### Order of Operations
Contributors without commit privileges are asked to submit or comment on an
[issue](https://github.com/Deltakit/deltakit/issues) before submitting
a pull request.

### Issue / PR Titles / Commit Messages
Our project uses [semantic versioning](https://semver.org/).
In particular, we use [semantic release](https://python-semantic-release.readthedocs.io/en/latest/)
to automate our releases. Consequently, please format your issue/PR titles and commit messages
according to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
specification with the following exceptions:

- Use `bug` as the prefix for a bug-report issue;
 `fix` is reserved for PRs/commits that resolve a bug.
- Use `request` as the prefix for a request;
  more descriptive prefixes (`feat`/`perf`) are used for PRs/commits.

Other commit *types* that we recognize include `release` (for work relating to release tooling
and/or releases) and `dev` (for development work that doesn't fit in another category), but
typically these types will only be used by package maintainers.

### Rebasing / Force-Pushing
During PR review, please refrain from rebasing and force-pushing to your branch.
If needed, feel free to add revert or merge commits; always use regular pushes.
This ensures that reviewers can easily see what has changed since their last review.
Once the PR has been approved but before it is merged, you'll have the opportunity
to do an interactive rebase (e.g. to improve commit history), or we can squash merge.

### Inline Comment / Suggested Change Resolution
PR contributors are asked to leave inline comments unresolved so *reviewers* can confirm
that their comments have been addressed. (Reviewers, kindly resolve your own comments once
you have checked them!) Exception: reviewers are encouraged to make use of the "Add a suggestion"
feature, and contributors are encouraged to make use of the
["Add suggestion to batch" and "Commit suggestions"](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request)
features; these comments will automatically be resolved when the suggestions are committed.

### License / Use of Artificial Intelligence
Our project uses the [Apache 2.0 license](https://github.com/Deltakit/deltakit/blob/main/LICENSE). Please ensure that your contributions
are consistent with that license. If you're unsure, please ask in an issue before posting
content that may not be compatible with our project.
Note that large language models (LLMs) may be trained on and can potentially reproduce
content that is incompatible with our license. If an LLM or similar has influenced your
contribution, please describe how so we can ensure that our project remains free of
license-incompatible content.

### Continuous Integration (CI) Usage
Please use our CI services responsibly. Since CI re-runs on every push to a pull request branch,
please avoid repeated pushes of small commits.

### Minimum Supported Dependencies
The project follows [SPEC 0](https://scientific-python.org/specs/spec-0000/). Roughly, this
means that we will support Python versions for three years and other core dependencies for
two years.

### Decision Making / Governance
Decisions are made by consensus of participants in a GitHub issue or PR. In case of disagreement,
[code owners](https://github.com/Deltakit/deltakit/blob/main/CODEOWNERS) have final authority.

### Code Formatting / Linting / Typing
All packages except `deltakit-decode` use `ruff format`, all packages use `ruff check` to
enforce linting rules, and all packages use `mypy` to enforce typing rules.
See [`pyproject.toml`](https://github.com/Deltakit/deltakit/blob/main/pyproject.toml) for specific rules.

### Release
For more information about release processes, see the [Deltakit release procedure](release.md).

### Contributor License Agreement
First-time contributors will be asked to agree to a CLA. This will be automated using a GitHub
app shortly; in the meantime, please [contact us](mailto:deltakit@riverlane.com).
