# Contributing to Deltakit

```{toctree}
:hidden:

CODE_OF_CONDUCT
RELEASE
SECURITY
```

Thank you for considering a contribution to Deltakit. We welcome contributions of many forms from anyone, most of which don't even require writing code. The sections below describe key definitions for contribution types and recommended processes to help guide you through contributing.

> For introductory information about contributing to open source projects, please see the [Scientific Python Contributor Guide](https://learn.scientific-python.org/contributors/).

Prior to making a contribution, we kindly ask to review our [code of conduct](https://github.com/Deltakit/deltakit/blob/main/docs/CODE_OF_CONDUCT.md).

## Reporting an issue

Issues contributions concern reporting a behavioural discrepancy in the code base (bug), an enhancement suggestion or a constructive participation to an existing issue. Please check the GitHub Issues trackers for the relevant Deltakit package first to see if a similar report has already been submitted. If so, add a comment with your current observation and details. Otherwise, create a new issue report:

- **Bug reports**: A "bug" is defined as a discrepancy between documented and actual behaviour or an *inaccurate* error message. Bug reports can be created [here](https://github.com/Deltakit/deltakit/issues/new?template=bug.yml).
- **Enhancement and feature requests**: Requests for improvements, enhancements, or new features are highly appreciated. Request reports can be created [here](https://github.com/Deltakit/deltakit/issues/new?template=request.yml).
- **Issue participation**: It is also possible to constructively participate in current [issues](https://github.com/Deltakit/deltakit/issues/) by reproducing bugs, investigating their causes, or contributing to discussions on best fixes and implementation designs.

> [!NOTE]
> To submit bug reports or feature requests, please open a GitHub Issue in the relevant Deltakit package repository. For example, if your query is related to `deltakit-compile`, please open an Issue in the [`deltakit-compile` repository](https://github.com/Deltakit/deltakit-compile/issues), and similarly for issues related to [`deltakit-visualise`](https://github.com/Deltakit/deltakit-visualise/issues) and other Deltakit packages. If the issue does not fall under a specific Deltakit package, you can report it in the [issue tracker](https://github.com/Deltakit/deltakit/issues) of this repository. This is the appropriate place to also report issues about documentation related to *any* Deltakit package.

## Picking up work

Most contributions to Deltakit are voluntary. To avoid duplicated contributions, please follow the guidelines below.

1. Issues labelled `good first issue` or `help wanted` are ready for external contributors and a good place to start contributing to Deltakit. For other issues, please ask before starting: some are already planned or in progress within the Deltakit core team ([`CODEOWNERS`](https://github.com/Deltakit/deltakit/blob/main/.github/CODEOWNERS) and maintainers).

2. If you’d like to contribute to an issue already assigned to someone else, express your interest in a comment and discuss splitting the work with the assignee and/or the core Deltakit core team before committing your time.

3. You’re welcome to express interest in unassigned issues by explaining your approach in a comment. Please wait for the Deltakit core team to agree to assign you before starting substantial work. In case of multiple expressions of interest, priority is decided based on the basis of:

    - Quality of the approach
    - First-come-first-serve

4. If you can no longer work on an issue, please let us know in a comment.

Deltakit core team members follow the same rules: a Deltakit issue being worked on internally is assigned to the person doing the work.

If an assigned contributor fails to respond to an updates request within a week's time, they may be unassigned from the issue.

### Duplicate contributions

In cases where multiple Pull Requests (PRs) overlap in scope, the Deltakit core team reserves the right to prioritise one PR based on the above rules and close the other PRs, which are labelled as `duplicate`.

Duplicate issues are closed with the `duplicate` label and a link to the original. Please add any new comments to the original issue. The authors of the duplicate PR are welcome to review the PR that goes ahead.

If the issue has an active assignee, their PR is reviewed first. PRs from others on an assigned issue may not be reviewed while the assignment is active.  If the issue is unassigned, maintainers may first review the PR that is ready for review first: not a draft, linked to the issue, passing CI, and with the CLA signed. Being opened first does not by itself give priority. 

Where multiple PRs target the same feature and are complementary, the Deltakit core team may coordinate the separation of scope and advise on the order in which to review and merge the PRs. All such contributors are recognised as authors.

### Contributions from the Deltakit Community Fund

Issues labelled as `community-fund-proposal` document either:

- A proposed Deltakit Community Fund project if the issue is *unassigned*. If you’re interested in applying for a grant to work on it, you're welcome to check the [application schedule](https://github.com/Deltakit/deltakit/issues/343) and [submit an application](https://unitary.foundation/community/deltakit_community_fund/).

- An ongoing project if the issue is *assigned*.

PRs and other contributions labelled as `community-fund-proposal` are prioritised when duplicate contributions with overlapping scope are submitted. The scope of active (i.e. assigned) `community-fund-proposal` contributions is reserved for the duration of the Deltakit Community Fund project.

Once the project ends, that scope, issue, or proposed contribution becomes available to be picked up. This can be a pathway to build on and extend the output of Deltakit Community Fund projects.

## Submitting a Pull Request

Resolving an issue is possible by submitting a PR. The recommended process is to fork the `main` branch using GitHub interface button on the top-right corner. This will create a new GitHub repository prefixed with your GitHub handle.

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

Deltakit provides a [`justfile`](../justfile) with optional shortcuts for common development tasks. Each `just` recipe is shorthand for an equivalent `uv` command. Learn more about [`just`](https://github.com/casey/just).

After running `uv sync`, run the test suite with either command:

```sh
just test
```

```sh
uv run --group test pytest
```

Tests are executed using the [Pytest](https://docs.pytest.org/en/stable/) framework.

To use the exact dependency versions used in CI (Python 3.13 with `lowest-direct` resolution), sync and run:

```sh
uv sync --python 3.13 --resolution lowest-direct --group test
uv run --no-sync pytest
```

### Building documentation

Build the HTML documentation with either command:

```sh
just docs
```

```sh
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

The generated documentation can then be viewed in any web browser.

### Other common tasks

Run any of these shortcuts or their corresponding `uv` commands directly:

```sh
just lint
just security
just check
just build
```

```sh
uv run --group lint typos
uv run --group lint ruff check
uv run --group lint ruff format --check
uv run --group lint pydoclint .
uv run --group lint mypy
uv run --group lint deptry src
uv run --group security bandit .
uv run --group security pip-audit --ignore-vuln CVE-2025-53000 --ignore-vuln PYSEC-2026-2132
uv build --no-sources
```

`just check` runs the lint, security, and test recipes. Run `just --list` to see all available recipes.

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

For more information about release processes, see the [Deltakit release procedure](https://github.com/Deltakit/deltakit/blob/main/docs/RELEASE.md).

### Security

For more information about security, see the [Deltakit security policy](https://github.com/Deltakit/deltakit/blob/main/docs/SECURITY.md).

### Contributor License Agreement

First-time contributors will be asked to agree to a CLA. This is automated using a GitHub
app for open PR.

## Use of AI tools

You are welcome to use artificial intelligence (AI) tools, such as large language models (LLMs), chat assistants, code completion and coding agents, when contributing to Deltakit, provided that this use complies with our policy outlined in this section.

This policy applies to every repository in the [Deltakit GitHub organisation](https://github.com/Deltakit) and to every contributor, including code owners and maintainers.

### Why we have this policy

The cost of producing a *plausible* PR reduces drastically with Ai use, but the cost of reviewing remains largely the same (or even increases for large or verbose contributions). A PR that took a minute to generate can take hours to review. When unreviewed tool output is sent to maintainers, the work of checking it is moved onto them. Following the [LLVM project](https://llvm.org/docs/AIToolPolicy.html), we call these *extractive* contributions.

A popular golden rule is that **a contribution should be worth more to the project than the time it takes to review it.**

Quantum error correction experiments may fail quietly. People publish results computed with Deltakit, so correctness matters more to us than speed of development. AI tools are good at producing code that looks right, which is exactly the failure mode we most need to guard against.

### Responsibilities of the contributor

#### The author is accountable

You as the contributor are ultimately accountable for the changes, even when AI tools are used. We provide the following advice and guiding principles on using AI tools responsibly when contributing to Deltakit.

- Read and understand every line before you ask for a review. You are expected to be able to explain what a change does without the AI tool's help. Otherwise, it isn't ready for review and you may mark the contribution, e.g. a PR, as *Draft*.
- After the contribution is reviewed, answer review questions yourself.
- Keep changes focused and concise. AI output tends to be verbose: trim unrelated edits, redundant comments, speculative abstractions and docstrings longer than necessary before submitting.
- The advice for *new contributors* is to start small, with changes you fully understand. See more advice on [picking up work](#picking-up-work).
- If an AI tool makes mistakes, these become your mistakes once you sign off the contributions.

**Don't credit tools as authors.** Only humans can agree to the [Contributor License Agreement](#contributor-license-agreement) (CLA) and take responsibility for a contribution.

#### Disclose substantial use

Disclosure helps reviewers decide how much scrutiny to apply. It is not a penalty, and disclosed AI use is not a reason on its own to reject a contribution.

You don't need to disclose purely grammatical or translation help, although mentioning it is welcome.

> [!NOTE]
> **How to disclose:**
>
> - In the PR description, name the tool and model in the *Additional Information* section of the template, explain what parts of the code the tool changed, and how you verified the output. *Example:* "Claude Code, Opus 5.5: drafted the tests in `tests/test_decoder.py`, which I reviewed and extended with edge cases". "Used LLMs" is not sufficient.
>
> - In individual commits, you may also add a footnote description like:
>
>  ```text
>  feat(decode): add weighted matching for leakage-aware graphs
>
>  Assisted-by Claude Code (Claude Opus 5.5)
>  ```

If we find substantial *undisclosed* use of AI tools, we treat it as a policy breach.

#### Obligations to the License and the CLA

Using AI tools doesn't change the obligations described under [License](#license) and [Contributor License Agreement](#contributor-license-agreement). As a reminder:

- You are responsible for making sure your contribution doesn't include material you have no right to contribute. AI tools can reproduce code from their training data, including code under licenses that are incompatible with Deltakit's.
- If generated code looks like it came from an existing project, you are responsible for checking that the contribution complies with the original project's License and Deltakit's Licence. If in doubt, please rewrite the artefact in your own way.

### Project tooling and agent configuration

- Don't commit personal agent configuration, memory or session files (for example `CLAUDE.md`, `.claude/`, `.cursor/`, `.aider*` or chat transcripts).
- Automation that code owners have approved for the project, such as Dependabot, release tooling and the CLA bot, is exempt from the human-in-the-loop rule. New AI-driven automation (for example automated review or triage bots) must be proposed in an issue, and approved by code owners.

### Further reading

You can find more about AI policies in open-source projects at the links below. These resources don't form part of Deltakit's policy. Links are as of 20 Oct 2026.

- [LLVM AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html)
- [NumPy AI policy](https://numpy.org/devdocs/dev/ai_policy.html)
- [SciPy AI policy](https://github.com/scipy/scipy/blob/main/doc/source/dev/conduct/ai_policy.rst)
- [Qiskit: Use of generative AI](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md#use-of-generative-ai)
- [PennyLane AI Tool Use Policy](https://github.com/PennyLaneAI/pennylane/blob/master/AI_Policy.md)
- [Scientific Python: Community considerations around AI](https://blog.scientific-python.org/scientific-python/community-considerations-around-ai/)