"""Check whether the required approvals have been met from REQUIRED_REVIEWERS to allow the PR to
merge. Returns exit code 0 if approvals are met, 1 on error, and 2 if approvals are not met."""

import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import requests


class ReqReviewerRule(NamedTuple):
    """
    A single rule from the ``REQUIRED_REVIEWERS`` file.

    Attributes:
        pattern: Glob pattern used to match file paths.
        owners: Usernames of the reviewers required for matching files.
    """

    pattern: str
    owners: list[str]


def get_changed_files(repo: str, pr_number: int, token: str) -> list[str]:
    """
    Get the filenames of the files changed in a pull request.

    Args:
        repo: Repository in ``owner/repository`` format.
        pr_number: Pull request number.
        token: GitHub API token.

    Returns:
        The list of changed file paths.
    """
    files = requests.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files",
        headers={"Authorization": f"token {token}"},
    ).json()

    return [f["filename"] for f in files]


def get_approved_reviewers(repo: str, pr_number: int, token: str) -> set[str]:
    """
    Get the usernames of reviewers who approved the pull request.

    Args:
        repo: Repository in ``owner/repository`` format.
        pr_number: Pull request number.
        token: GitHub API token.

    Returns:
        The set of usernames of approving reviewers.
    """
    reviews = requests.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews",
        headers={"Authorization": f"token {token}"},
    ).json()

    return {r["user"]["login"] for r in reviews if r["state"] == "APPROVED"}


def parse_required_reviewers_file(path: Path) -> list[ReqReviewerRule]:
    """Parse a ``REQUIRED_REVIEWERS`` file.

    Args:
        path: Path to the ``REQUIRED_REVIEWERS`` file.

    Returns:
        The parsed reviewer rules.
    """
    rules = []
    with path.open("r") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue

            parts = stripped_line.split()
            pattern = parts[0]
            owners = [o.lstrip("@") for o in parts[1:]]
            rules.append(ReqReviewerRule(pattern=pattern, owners=owners))

    return rules


def determine_required_reviewers(
    rules: list[ReqReviewerRule], filenames: list[str], pr_author: str
) -> set[str]:
    """Determine the reviewers required for a pull request.

    The last matching rule for each file overrides previous matches. The
    pull request author is excluded from the required reviewers.

    Args:
        rules: Parsed reviewer rules.
        filenames: Files changed in the pull request.
        pr_author: Username of the pull request author.

    Returns:
        The set of required reviewers.
    """
    required = set()

    for file in filenames:
        last_match_owners = set()

        for rule in rules:
            if fnmatch.fnmatch(file, rule.pattern):
                last_match_owners = rule.owners

        if last_match_owners:
            required.update(last_match_owners)

    required.discard(pr_author)
    return required


def build_reviewer_comment(required: set[str], approved: set[str], is_draft: bool) -> str:
    """
    Build the pull request reviewer status comment.

    Args:
        required: Required reviewers.
        approved: Reviewers who have approved the pull request.
        is_draft: Whether the pull request is a draft.

    Returns:
        The comment body to post on the pull request.
    """
    required_list = ", ".join(sorted(required))
    approved_list = ", ".join(sorted(approved)) or "_None_"

    if is_draft:
        return (
            "📝 **Draft PR - suggested reviewers**\n\n"
            "At least one of the following must approve this PR once it leaves draft:\n\n"
            f"{required_list}"
        )

    if bool(approved & required):
        return """✅ **Reviewer requirement satisfied**"""

    return (
        "❌ **Missing required reviewer approval**\n\n"
        "At least one of the following must approve this PR:\n\n"
        f"{required_list}\n\nCurrently approved by:\n\n{approved_list}"
    )


def post_or_update_comment(
    repo: str, pr_number: str, token: str, comment_marker: str, comment_body: str
) -> None:
    """Post a pull request comment or update an existing one.

    A comment is considered to already exist if it contains the supplied
    marker.

    Args:
        repo: Repository in ``owner/repository`` format.
        pr_number: Pull request number.
        token: GitHub API token.
        comment_marker: Marker used to identify the managed comment.
        comment_body: Comment body to post.
    """
    headers = {"Authorization": f"token {token}"}

    # Find existing bot comment
    comments = requests.get(
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments", headers=headers
    ).json()

    existing = None
    for c in comments:
        if c["user"]["type"] == "Bot" and comment_marker in c["body"]:
            existing = c
            break

    # Create or update comment
    body_with_marker = comment_marker + "\n" + comment_body
    if existing:
        requests.patch(existing["url"], headers=headers, json={"body": body_with_marker})
    else:
        requests.post(
            f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
            headers=headers,
            json={"body": body_with_marker},
        )


if __name__ == "__main__":
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    is_draft = os.environ["PR_IS_DRAFT"] == "true"

    with Path(os.environ["GITHUB_EVENT_PATH"]).open("r") as f:
        event = json.load(f)
    pr_number = event["pull_request"]["number"]
    pr_author = event["pull_request"]["user"]["login"]

    changed_files = get_changed_files(repo, pr_number, token)
    approved_reviewers = get_approved_reviewers(repo, pr_number, token)

    rules = parse_required_reviewers_file(Path(".github/REQUIRED_REVIEWERS"))
    required_reviewers = determine_required_reviewers(rules, changed_files, pr_author)

    marker = "<!-- required-reviewers-check -->"
    comment_body = build_reviewer_comment(required_reviewers, approved_reviewers, is_draft)
    post_or_update_comment(repo, pr_number, token, marker, comment_body)

    if bool(approved_reviewers & required_reviewers):
        print("Valid approval found")
        sys.exit(0)
    else:
        print("Missing required approval")
        sys.exit(2)
