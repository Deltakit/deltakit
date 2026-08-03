# (c) Copyright Riverlane 2020-2025.
"""Helpers for GF(2) linear algebra assertions in tests."""

import galois
import numpy as np
import numpy.typing as npt


def rank_gf2(matrix: npt.NDArray[np.integer]) -> int:
    """Return the rank of ``matrix`` over GF(2).

    Args:
        matrix: Binary matrix to rank.

    Returns:
        Rank of ``matrix`` over GF(2).
    """
    return int(np.linalg.matrix_rank(galois.GF2(np.asarray(matrix, dtype=np.int_))))


def quotient_dimension_gf2(
    basis: npt.NDArray[np.integer], quotient_basis: npt.NDArray[np.integer]
) -> int:
    """Return ``dim(span(basis) / span(quotient_basis))`` over GF(2).

    Args:
        basis: Rows spanning the larger vector space.
        quotient_basis: Rows spanning the subspace to quotient out.

    Returns:
        Dimension of ``span(basis) / span(quotient_basis)`` over GF(2).
    """
    return rank_gf2(np.vstack([quotient_basis, basis])) - rank_gf2(quotient_basis)


def assert_valid_css_logical_basis(
    logicals: npt.NDArray[np.integer],
    null_space_defining_checks: npt.NDArray[np.integer],
    quotient_space_defining_checks: npt.NDArray[np.integer],
    null_space_dimension: int,
    quotient_space_dimension: int,
) -> None:
    """Assert CSS logical basis properties without fixing the returned basis.

    Args:
        logicals: Candidate logical rows.
        null_space_defining_checks: Checks whose kernel must contain
            ``logicals``.
        quotient_space_defining_checks: Checks whose row space defines
            stabiliser-equivalent rows.
        null_space_dimension: Dimension of the kernel defined by
            ``null_space_defining_checks``.
        quotient_space_dimension: Dimension of the stabiliser-equivalent row
            space defined by ``quotient_space_defining_checks``.
    """
    logicals = np.asarray(logicals, dtype=np.int_)
    expected_num_logicals = null_space_dimension - quotient_space_dimension

    assert logicals.ndim == 2
    assert logicals.shape == (
        expected_num_logicals,
        null_space_defining_checks.shape[1],
    )
    assert logicals.shape[1] == quotient_space_defining_checks.shape[1]
    assert np.all(np.isin(logicals, [0, 1]))
    assert np.all((null_space_defining_checks @ logicals.T) % 2 == 0)
    assert (
        quotient_dimension_gf2(logicals, quotient_space_defining_checks)
        == expected_num_logicals
    )
