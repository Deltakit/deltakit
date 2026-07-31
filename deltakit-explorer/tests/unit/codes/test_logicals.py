from collections.abc import Callable
from itertools import product

import deltakit_circuit as circuit
import deltakit_stim as stim
import numpy as np
import numpy.typing as npt
import pytest

from deltakit_explorer.codes import (
    RotatedPlanarCode,
    UnrotatedPlanarCode,
)
from deltakit_explorer.codes._logicals import (
    css_code_compute_logicals,
    paulistring_to_operator,
)
from tests.helpers._codes import (
    bivariate_bicycle_parity_check_matrices,
    planar_code_parity_check_matrices,
)
from tests.helpers._gf2 import assert_valid_css_logical_basis, rank_gf2


@pytest.mark.parametrize("string", ["+XY", "+YX", "-ZXY"])
def test_paulistring_to_operator(string):
    paulistring = stim.PauliString(string)
    index_to_qubit = {i: circuit.Qubit(i) for i in range(len(string[1:]))}
    operator = paulistring_to_operator(paulistring, index_to_qubit)
    for i, (op, char) in enumerate(zip(operator, string[1:], strict=True)):
        gate = getattr(circuit, f"Pauli{char}")
        assert op == gate(circuit.Qubit(i))


# Uncomment when properties have been verified
# # Test cases from:
# # https://quantumcomputing.stackexchange.com/questions/37812/how-to-find-a-set-of-independent-logical-operators-for-a-stabilizer-code-with-st
# @pytest.mark.parametrize(
#     "stabilisers, operators",
#     [
#         (["XXXX", "ZZZZ"], [["+_X_X", "+Z__Z"], ["+_XX_", "+Z_Z_"]]),
#         (["XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"], [["-Z_XX_", "-_ZXZ_"]]),
#     ],
# )
# def test_get_str_logical_operators_from_tableau(stabilisers, operators):
#     stabilisers = [stim.PauliString(i) for i in stabilisers]
#     operators_ref = [tuple(stim.PauliString(i) for i in j) for j in operators]
#     operators_res = get_str_logical_operators_from_tableau(stabilisers)
#     operators_res == operators_ref


def _binary_matrix(rows: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Convert binary rows into a dense ``uint8`` array.

    Args:
        rows: Row vectors to place in the matrix.

    Returns:
        A dense ``uint8`` array built from ``rows``.
    """
    return np.asarray(rows, dtype=np.uint8)


def _empty_binary_matrix(n_cols: int) -> npt.NDArray[np.uint8]:
    """Construct an empty dense ``uint8`` array with a fixed column count.

    Args:
        n_cols: Number of columns in the returned matrix.

    Returns:
        A dense ``uint8`` array of shape ``(0, n_cols)``.
    """
    return np.zeros((0, n_cols), dtype=np.uint8)


def _binary_row_space(matrix: npt.NDArray[np.uint8]) -> set[tuple[int, ...]]:
    """Enumerate the row space of a binary array using GF(2) arithmetic.

    Args:
        matrix: A dense binary ``uint8`` array.

    Returns:
        The set of all binary vectors obtainable as mod-2 sums of rows of
        ``matrix``, represented as tuples.

    Notes:
        This helper uses a brute-force enumeration over all binary choices of row
        coefficients, so it is only suitable for the small matrices used in these
        tests.
    """
    num_rows, num_cols = matrix.shape

    if num_rows == 0:
        return {tuple([0] * num_cols)}

    vectors: set[tuple[int, ...]] = set()
    for coefficients in product((0, 1), repeat=num_rows):
        vector = np.zeros(num_cols, dtype=np.uint8)
        for coefficient, row in zip(coefficients, matrix, strict=True):
            if coefficient:
                vector ^= row
        vectors.add(tuple(int(value) for value in vector))

    return vectors


def _binary_null_space(matrix: npt.NDArray[np.uint8]) -> set[tuple[int, ...]]:
    """Enumerate the null space of a binary array using GF(2) arithmetic.

    Args:
        matrix: A dense binary ``uint8`` array.

    Returns:
        The set of all binary vectors ``v`` such that ``matrix @ v == 0`` over
        GF(2), represented as tuples.

    Notes:
        This helper checks every binary vector of the appropriate length, so it is
        a brute-force construction intended only for the small test cases in this
        module.
    """
    _, num_cols = matrix.shape

    vectors: set[tuple[int, ...]] = set()
    for entries in product((0, 1), repeat=num_cols):
        vector = np.asarray(entries, dtype=np.uint8)
        if np.all((matrix @ vector) % 2 == 0):
            vectors.add(entries)

    return vectors


def _dimension(space: set[tuple[int, ...]]) -> int:
    return int(np.log2(len(space)))


@pytest.mark.parametrize(
    ("hx", "hz"),
    [
        (_empty_binary_matrix(3), _empty_binary_matrix(3)),
        (_binary_matrix([[1, 1, 0], [0, 1, 1]]), _empty_binary_matrix(3)),
        (_binary_matrix(np.eye(3, dtype=np.uint8).tolist()), _empty_binary_matrix(3)),
        (
            _binary_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
            _empty_binary_matrix(3),
        ),
        (
            _binary_matrix([[1, 1, 1, 1]]),
            _binary_matrix([[1, 1, 0, 0], [0, 0, 1, 1]]),
        ),
    ],
)
def test_css_code_compute_logicals_returns_valid_css_logical_bases(
    hx: npt.NDArray[np.uint8], hz: npt.NDArray[np.uint8]
):
    assert np.all((hx @ hz.T) % 2 == 0)

    lx, lz = css_code_compute_logicals(hx.astype(float), hz.astype(float))

    assert_valid_css_logical_basis(
        lx,
        hz,
        hx,
        null_space_dimension=_dimension(_binary_null_space(hz)),
        quotient_space_dimension=_dimension(_binary_row_space(hx)),
    )
    assert_valid_css_logical_basis(
        lz,
        hx,
        hz,
        null_space_dimension=_dimension(_binary_null_space(hx)),
        quotient_space_dimension=_dimension(_binary_row_space(hz)),
    )


def test_css_code_compute_logicals_is_invariant_to_redundant_rows():
    hx = _binary_matrix([[1, 1, 0], [0, 1, 1]])
    redundant_hx = _binary_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    hz = _empty_binary_matrix(3)

    lx, lz = css_code_compute_logicals(hx.astype(float), hz.astype(float))
    redundant_lx, redundant_lz = css_code_compute_logicals(
        redundant_hx.astype(float), hz.astype(float)
    )

    assert _binary_row_space(np.asarray(lx, dtype=np.uint8)) == _binary_row_space(
        np.asarray(redundant_lx, dtype=np.uint8)
    )
    assert _binary_row_space(np.asarray(lz, dtype=np.uint8)) == _binary_row_space(
        np.asarray(redundant_lz, dtype=np.uint8)
    )


def test_css_code_compute_logicals_preserves_logical_dimension_for_known_rank_checks():
    # Adapted from the explicit GF(2) row-reduction fixture in galois/tests/fields/test_linalg.py.
    hx = _binary_matrix(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    hz = _empty_binary_matrix(8)

    rank_hx = _dimension(_binary_row_space(hx))
    assert rank_hx == 4

    lx, lz = css_code_compute_logicals(hx.astype(float), hz.astype(float))

    assert lx.shape == (8 - rank_hx, 8)
    assert lz.shape == (8 - rank_hx, 8)
    assert_valid_css_logical_basis(
        lx,
        hz,
        hx,
        null_space_dimension=_dimension(_binary_null_space(hz)),
        quotient_space_dimension=_dimension(_binary_row_space(hx)),
    )
    assert_valid_css_logical_basis(
        lz,
        hx,
        hz,
        null_space_dimension=_dimension(_binary_null_space(hx)),
        quotient_space_dimension=_dimension(_binary_row_space(hz)),
    )


@pytest.mark.parametrize(
    ("parity_check_matrix_factory", "expected_n", "expected_num_logicals"),
    [
        pytest.param(
            lambda: bivariate_bicycle_parity_check_matrices(3, 5, [1, 1, 4], [0, 1, 2]),
            30,
            8,
            id="bivariate-bicycle-3x5",
        ),
        pytest.param(
            lambda: planar_code_parity_check_matrices(lambda: RotatedPlanarCode(3, 3)),
            9,
            1,
            id="rotated-planar-3x3",
        ),
        pytest.param(
            lambda: planar_code_parity_check_matrices(
                lambda: UnrotatedPlanarCode(3, 3)
            ),
            13,
            1,
            id="unrotated-planar-3x3",
        ),
    ],
)
def test_css_code_compute_logicals_returns_valid_bases_for_code_fixtures(
    parity_check_matrix_factory: Callable[
        [], tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]
    ],
    expected_n: int,
    expected_num_logicals: int,
):
    hx, hz = parity_check_matrix_factory()

    assert hx.shape[1] == hz.shape[1] == expected_n
    assert np.all((hx @ hz.T) % 2 == 0)
    assert expected_num_logicals == expected_n - rank_gf2(hx) - rank_gf2(hz)

    lx, lz = css_code_compute_logicals(hx.astype(float), hz.astype(float))

    assert_valid_css_logical_basis(
        lx,
        hz,
        hx,
        null_space_dimension=expected_n - rank_gf2(hz),
        quotient_space_dimension=rank_gf2(hx),
    )
    assert_valid_css_logical_basis(
        lz,
        hx,
        hz,
        null_space_dimension=expected_n - rank_gf2(hx),
        quotient_space_dimension=rank_gf2(hz),
    )
    anticommutation_matrix = (
        np.asarray(lx, dtype=np.int_) @ np.asarray(lz, dtype=np.int_).T
    ) % 2
    assert rank_gf2(anticommutation_matrix) == expected_num_logicals
