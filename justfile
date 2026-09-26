# Show available recipes when `just` is run with no arguments.
default:
    @just --list

# Run the test suite.
test:
    uv run --group test pytest

# Run static-analysis checks.
lint:
    uv run --group lint typos
    uv run --group lint ruff check
    uv run --group lint ruff format --check
    uv run --group lint pydoclint .
    uv run --group lint mypy
    uv run --group lint deptry src

# Run security checks.
security:
    uv run --group security bandit .
    uv run --group security pip-audit --ignore-vuln CVE-2025-53000 --ignore-vuln PYSEC-2026-2132

# Build the HTML documentation.
docs:
    uv run --group docs sphinx-build -W -b html docs docs/_build/html

# Build distributions.
build:
    uv build --no-sources --all-packages

# Run the main local validation suite.
check: lint security test
