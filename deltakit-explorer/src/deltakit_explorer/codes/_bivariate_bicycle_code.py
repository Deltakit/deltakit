# (c) Copyright Riverlane 2020-2025.
"""
This module contains an implementation of IBM's Bivariate Bicycle
qLDPC codes as specified in https://arxiv.org/pdf/2308.07915.pdf.

The code is flexible and should allow exploration of codes with parameters
other than those listed in the paper.
"""
from __future__ import annotations

from functools import reduce
from itertools import product

import galois
import numpy as np
import numpy.typing as npt
from deltakit_circuit import PauliX, PauliZ, Qubit
from deltakit_circuit._basic_types import Coord2D

from deltakit_explorer.codes._css._css_code import CSSCode
from deltakit_explorer.codes._logicals import css_code_compute_logicals
from deltakit_explorer.codes._stabiliser import Stabiliser


def _find_anticommuting_pairs(
    x_logs_as_vecs: list[list[int]], z_logs_as_vecs: list[list[int]], k: int, n: int
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Given a set of logical X and Z operators, find k anti-commuting pairs of
    logical operators that are also independent.

    Given a set of x and z logical operators,
    (0) for each x (z) find all z (x) that anticommute
    (1) pick an x and find a z that anticommutes with it
    (2) for each other z that anticommutes with the chosen x, multiply it by the chosen z so that it commutes with the chosen x
    (3) for each other x that anticommutes with the chosen z, multiply it by the chosen x so that it commutes with the chosen z
    Repeat until a set of desired size is found.

    Logical operators are converted into integers for optimsation purposes. This is because
    integer operations can be completed in one CPU clock cycle whereas numpy arrays can take longer.
    The integers are computed from the logical operator's binary vector representation, by just
    interpreting the vector as an integer.

    Params
    ------
    x_logs_as_vecs : List[List[int]]
        List of all logical X operators, as binary vectors, e.g, Lists with entry 0 or 1.
    z_logs_as_vecs : List[List[int]]
        List of all logical Z operators, as binary vectors, e.g, Lists with entry 0 or 1.
    k : int
        Number of logical qubits in the code.
    n : int
        Number of physical qubits in the code.

    Returns
    -------
    Tuple[List[List[int]], List[List[int]]]
        Tuple of lists of k X and Z logical operators respectively, as binary vectors,
        that is, Lists with entry 0 or 1.
    """
    # convert to ints for optimsation
    x_logs = [int("".join("01"[i] for i in vec), 2) for vec in x_logs_as_vecs]
    z_logs = [int("".join("01"[i] for i in vec), 2) for vec in z_logs_as_vecs]

    chosen_x_logs: list[int] = []
    chosen_z_logs: list[int] = []
    while len(chosen_x_logs) < k:
        if len(x_logs) == 0 or len(z_logs) == 0:
            msg = f"Unable to construct {k} logical operators, could only find {len(chosen_x_logs)}"
            raise ValueError(msg)
        # get which x anti-commutes with which z and vice-versa
        # use dictionaries for constant-time lookup
        x_ac_with_z: dict[int, dict] = {x: {} for x in x_logs}
        z_ac_with_x: dict[int, dict] = {z: {} for z in z_logs}
        for x_log, z_log in product(x_logs, z_logs):
            if (x_log & z_log).bit_count() % 2 == 1:
                x_ac_with_z[x_log][z_log] = {}
                z_ac_with_x[z_log][x_log] = {}

        # pick the first available x and first available z that
        # anti-commutes with that x
        chosen_x = x_logs[0]
        try:
            chosen_z = next(iter(x_ac_with_z[chosen_x].keys()))
        except IndexError:
            # this logical X doesnt have any anticommuting Zs, so
            # try a different X
            del x_logs[0]
            continue
        chosen_x_logs.append(chosen_x)
        chosen_z_logs.append(chosen_z)

        # update the remaining logicals by multiplying in chosen z to the other zs
        # that anti-commutes with the chosen x, and vice versa
        for z_log_i, z_log in enumerate(z_logs):
            if z_log in x_ac_with_z[chosen_x]:
                z_logs[z_log_i] = z_log ^ chosen_z
        for x_log_i, x_log in enumerate(x_logs[1:]):
            if x_logs[x_log_i + 1] in z_ac_with_x[chosen_z]:
                x_logs[x_log_i + 1] = x_log ^ chosen_x

        # remove these logicals from the running
        x_logs.remove(chosen_x)
        z_logs.remove(
            0
        )  # chosen z will be zero since it will have been xor'd with itself

    # convert from int back to binary array
    chosen_x_logs_as_vec = [
        [int(bit) for bit in f"{op_as_int:b}".zfill(n)] for op_as_int in chosen_x_logs
    ]
    chosen_z_logs_as_vec = [
        [int(bit) for bit in f"{op_as_int:b}".zfill(n)] for op_as_int in chosen_z_logs
    ]
    return chosen_x_logs_as_vec, chosen_z_logs_as_vec


class Monomial:
    """
    Class used to encapsulate the "Monomial" concept used in the
    IBM paper.

    Monomials are expressions of the form (x^a)*(y^b) where the
    powers a and b are modulo the code parameters `l` and `m` respectively.

    These monomials describe individual data qubits, and a collection
    of Monomials (Polynomials) can be used to describe a collection of data qubits.

    The inverse of a Monomial is defined to be another Monomial with powers a', b'
    such that a+a'0, b+b'=0. That is, a Monomial that one can multiply by to get back
    to the identity element x^0 y^0.

    Params
    ------
    x_pow : int
        The power of the x-term in the Monomial.
    y_pow : int
        The power of the y-term in the Monomial.
    l : int
        The l parameter of the code the Monomial is
        a part of.
    m : int
        The m parameter of the code the Monomnial is
        a part of.
    """

    def __init__(self, x_pow: int, y_pow: int, l: int, m: int):  # noqa: E741
        if l > 0 and m > 0:
            self.l = l
            self.m = m
        else:
            msg = "l and m must be >= 0"
            raise ValueError(msg)
        self.x_pow = x_pow % l
        self.y_pow = y_pow % m

    def __mul__(self, other: object) -> Monomial:
        if not isinstance(other, Monomial):
            msg = f"Can only multiply Monomials by other Monomials, not {type(other)}"
            raise NotImplementedError(msg)
        if self.l != other.l or self.m != other.m:
            msg = (
                "Cannot multiply monomials of differing max degree."
                f" lhs: l={self.l}, m={self.m} | rhs: l={other.l}, m={other.m}"
            )
            raise ValueError(msg)
        return Monomial(
            (other.x_pow + self.x_pow) % self.l,
            (other.y_pow + self.y_pow) % self.m,
            self.l,
            self.m,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Monomial):
            return False
        return (
            self.x_pow == other.x_pow
            and self.y_pow == other.y_pow
            and self.l == other.l
            and self.m == other.m
        )

    def __repr__(self) -> str:
        return f"Monomial({self.x_pow}, {self.y_pow}, {self.l}, {self.m})"

    def __str__(self) -> str:
        return f"x^{self.x_pow} y^{self.y_pow}"

    def inverse(self) -> Monomial:
        """
        The inverse of a monomial is a monomial with powers
        such that when multiplied, the powers add to 0.
        """
        return Monomial(
            (self.l - self.x_pow) % self.l,
            (self.m - self.y_pow) % self.m,
            self.l,
            self.m,
        )

    def __rmul__(self, lhs: Monomial):
        return self.__mul__(lhs)


class Polynomial:
    """
    Class to encapsulate the "Polynomial" concept used in
    the IBM paper.
    """

    def __init__(self, monomials: list[Monomial]):
        self.monomials = monomials

    @staticmethod
    def from_vec(vec: list[int], l, m) -> Polynomial:  # noqa: E741
        """
        Create a Polynomial from a binary vector.

        Args
        ----
        vec : List[int]
            Binary vector, of the form of a List of integer entries 0 or 1.
            The vector is interpreted as the indices of the qubits that are
            represented by the Polynomial. E.g, index 0 in the vector is
            qubit 0, index 1 is qubit 1, and a 0 or 1 in these indices
            indicates whether that qubit is a part of the polynomial or not.
        l : int
            Max dimension of x component.
        m : int
            Max dimension of y component.

        Returns
        -------
        Polynomial
            Polynomial from the vector with monomial elements corresponding
            to the position of each 1 in the vector.
        """
        vec_of_monomials = []
        for i, x in enumerate(vec):
            if x != 0:
                a_i = i // m
                b_i = i % m
                vec_of_monomials.append(Monomial(a_i, b_i, l, m))
        return Polynomial(vec_of_monomials)

    def to_vec(self) -> list[int]:
        """
        Convert a Polynomial back to a binary vector.

        Returns
        -------
        List[int]
            Int (bool) vector representation of the Polynomial.
        """
        if len(self.monomials) == 0:
            return []
        vec = [0] * (self.monomials[0].l * self.monomials[0].m)
        for mon in self.monomials:
            vec[(mon.x_pow * mon.m) + mon.y_pow] = 1
        return vec

    def reverse(self) -> Polynomial:
        """
        Return reverse of a polynomial by computing the inverse
        of its constituent Monomials.
        """
        if len(self.monomials) == 0:
            return Polynomial([])
        return Polynomial([mon.inverse() for mon in self.monomials])

    def mult_by_monomial(self, monomial: Monomial) -> Polynomial:
        """
        Multiply a monomial into a polynomial. Will element-wise multiply
        the monomial into each monomial in the polynomial.

        Args
        ----
        monomial : Monomial
            Monomial to multiply into polynomial

        Returns
        -------
        Polynomial
            Result of multiplication.
        """
        if len(self.monomials) == 0:
            return Polynomial([])
        return Polynomial([mon * monomial for mon in self.monomials])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return False
        return all(m1 == m2 for m1, m2 in zip(self.monomials, other.monomials))

    def __repr__(self) -> str:
        return f"{[str(mon) for mon in self.monomials]}"

    def __str__(self) -> str:
        return self.__repr__()


class BivariateBicycleCode(CSSCode):
    r"""
    Class to represent the IBM Bivariate Bicycle qLDPC codes, as specified
    in arXiv:2308.07915.
    If `validate=True`, will check a series of conditions as the code is
    constructed to validate that the produced code is a valid BB code.

    Parameters
    ----------
    param_l : int
        The parameter `l`, used to construct a code of length
        `n = 2lm`.
    param_m : int
        The parameter `m`, used to construct a code of length
        `n = 2lm`.
    m_A_powers : List[int]
        The powers of the polynomial terms used to construct
        the matrix A. Each polynomial is of the form
        `x^a + y^b + y^c` so the sequence should specify
        `[a, b, c]`.
    m_B_powers : List[int]
        The powers of the polynomial terms used to construct
        the matrix B. Each polynomial is of the form
        `y^a + x^b + x^c` so the sequence should specify
        `[a, b, c]`.
    validate : bool, Optional
        If True, will perform a series of check on the code's
        parameters to ensure the constructed code is a valid BB code.
        By default, True.
    check_logical_operators_are_independent : bool, optional
        Whether to check if logical X-Z operator pairs are independent. If True, then
        the commuting relations [X_k, Z_l] = 0 (k != l) are checked for the logical
        operators, ensuring that the logical operators correspond to separate logical
        qubits. If False, this check is ignored, in which case the logical operators
        can be dependent, e.g. instead of only accepting X_1, X_2 and Z_1, Z_2, the
        input X_1, X_1*X_2 and Z_1, Z_1*Z_2 is also accepted.
        Note this is forced to False if validate is False.
        By default, False.

    Attributes
    ----------
    n : int
        The parameter n as in the [[n,k,d]] specification of an error-correcting code.
        E.g, number of physical qubits.
    k : int
        The parameter n as in the [[n,k,d]] specification of an error-correcting code.
        E.g, number of logical qubits.
    m_x : npt.NDArray[np.int\_]
        A binary matrix representing the matrix `x` in the paper.
        Used to construct matrices A and B for constructing the
        parity check matrices.
    m_y : npt.NDArray[np.int\_]
        A binary matrix representing the matrix `y` in the paper.
        Used to construct matrices A and B for constructing the
        parity check matrices.
    m_A_submatrices : Tuple[npt.NDArray[np.int\_], npt.NDArray[np.int\_], npt.NDArray[np.int\_]]
        A length-3 tuple containing the submatrices A1, A2 and A3 of A.
        Used to calculate stabilisers.
    m_B_submatrices : Tuple[npt.NDArray[np.int\_], npt.NDArray[np.int\_], npt.NDArray[np.int\_]]
        A length-3 tuple containing the submatrices B1, B2 and B3 of B.
        Used to calculate stabilisers.
    m_Hx : npt.NDArray[np.int\_]
        Parity check matrix for X checks.
    m_Hz : npt.NDArray[np.int\_]
        Parity check matrix for Z checks.
    """

    def __init__(
        self,
        param_l: int,
        param_m: int,
        m_A_powers: list[int],
        m_B_powers: list[int],
        validate: bool = True,
        check_logical_operators_are_independent: bool = False,
    ):
        if validate:
            self._validate_input_parameters(param_l, param_m, m_A_powers, m_B_powers)
        self.param_l = param_l
        self.param_m = param_m
        self.m_A_powers = m_A_powers
        self.m_B_powers = m_B_powers

        # work out code parameters
        self.n = 2 * param_l * param_m

        # create S and I for l and m
        m_S_l = np.roll(np.eye(param_l, dtype=np.int_), shift=1, axis=1)
        m_S_m = np.roll(np.eye(param_m, dtype=np.int_), shift=1, axis=1)
        m_I_l = np.eye(param_l, dtype=np.int_)
        m_I_m = np.eye(param_m, dtype=np.int_)

        # create x and y. Note that x and y commute
        m_x = np.kron(m_S_l, m_I_m)
        m_y = np.kron(m_I_l, m_S_m)

        if validate:
            # assert properties of x,y to catch errors
            self._assert_x_y_properties(self.param_l, self.param_m, m_x, m_y)
        self.m_x = m_x
        self.m_y = m_y

        # create A and B
        # A = x^a + y^b + y^c, B = y^d + x^e + x^f
        # add powers of x and y to create A and B
        m_A1 = np.linalg.matrix_power(m_x, m_A_powers[0])
        m_A2 = np.linalg.matrix_power(m_y, m_A_powers[1])
        m_A3 = np.linalg.matrix_power(m_y, m_A_powers[2])
        m_A = reduce(np.add, [m_A1, m_A2, m_A3]) % 2

        m_B1 = np.linalg.matrix_power(m_y, m_B_powers[0])
        m_B2 = np.linalg.matrix_power(m_x, m_B_powers[1])
        m_B3 = np.linalg.matrix_power(m_x, m_B_powers[2])
        m_B = reduce(np.add, [m_B1, m_B2, m_B3]) % 2

        if validate:
            # assert properties of A and B:
            self._assert_matrix_ma_mb_properties(m_A, m_B)
        self.m_A_submatrices = (m_A1, m_A2, m_A3)
        self.m_B_submatrices = (m_B1, m_B2, m_B3)

        # create Hx and Hz check matrices. Note that since A and B
        # commute, these are valid check matrices
        self.m_Hx = np.hstack((m_A, m_B))
        self.m_Hz = np.hstack((m_B.T, m_A.T))

        # place qubits on a square grid
        (
            self._mat_col_to_l_data_map,
            self._mat_col_to_r_data_map,
            self._mat_col_to_x_anc_map,
            self._mat_col_to_z_anc_map,
        ) = self._get_qubit_coords()

        self._gf2 = galois.GF2
        self.k = (
            self.n
            - np.linalg.matrix_rank(self._gf2(self.m_Hx))
            - np.linalg.matrix_rank(self._gf2(self.m_Hz))
        )

        # work out stabilisers
        stabilisers = self._get_stabilisers()

        # work out logicals
        x_logicals, z_logicals = self._get_logicals()

        # construct CSSCode
        super().__init__(
            stabilisers=[stabilisers],
            x_logical_operators=x_logicals,
            z_logical_operators=z_logicals,
            use_ancilla_qubits=True,
            check_logical_operators_are_independent=check_logical_operators_are_independent,
        )

    @staticmethod
    def _validate_input_parameters(
        ell: int,
        m: int,
        m_A_powers: list[int],
        m_B_powers: list[int],
    ):
        """
        Validate that the input parameters are valid as specified by
        the IBM paper.

        Params
        ------
        ell : int
            Parameter `l` as in the paper.
        m : int
            Parameter `m` as in the paper.
        m_A_powers : List[int]
            Powers for the individual A_i matrices to be raised to.
        m_B_powers : List[int]
            Powers for the individual B_i matrices to be raised to.
        """
        if ell < 1:
            msg = "param_l should be greater than or equal to 1."
            raise ValueError(msg)
        if m < 1:
            msg = "param_m should be greater than or equal to 1."
            raise ValueError(msg)
        if len(m_A_powers) != 3:
            msg = "m_A_powers should contain 3 integers."
            raise ValueError(msg)
        if len(m_B_powers) != 3:
            msg = "m_B_powers should contain 3 integers."
            raise ValueError(msg)
        if [n == 0 for n in m_A_powers + m_B_powers].count(True) > 1:
            msg = "Should only be at most one power of 0 in A and B powers"
            raise ValueError(msg)
        for n in [m_A_powers[0], *m_B_powers[1:]]:
            if not 0 <= n < ell:
                msg = "Powers of x must be in between 0 and l-1 inclusive"
                raise ValueError(msg)
            if (m_B_powers[1] - m_B_powers[2]) % ell == 0:
                msg = "Powers of x in the same polynomial must be distinct"
                raise ValueError(msg)
        for n in [m_B_powers[0], *m_A_powers[1:]]:
            if not 0 <= n < m:
                msg = "Powers of y must be in between 0 and m-1 inclusive"
                raise ValueError(msg)
            if (m_A_powers[1] - m_A_powers[2]) % m == 0:
                msg = "Powers of y in the same polynomial must be distinct"
                raise ValueError(msg)

    @staticmethod
    def _assert_x_y_properties(
        param_l: int, param_m: int, m_x: npt.NDArray, m_y: npt.NDArray
    ):
        # (1) xy = yx
        if not np.allclose(m_x @ m_y, m_y @ m_x):
            msg = "y*x should equal x*y"
            raise ValueError(msg)

        # (2) x^l = y^m = I_lm
        if not np.allclose(
            np.linalg.matrix_power(m_x, param_l),
            np.identity(param_l * param_m, dtype=np.int_),
        ):
            msg = "x^l should equal the identity matrix of shape (l*m, l*m)"
            raise ValueError(msg)
        if not np.allclose(
            np.linalg.matrix_power(m_y, param_m),
            np.identity(param_l * param_m, dtype=np.int_),
        ):
            msg = "y^m should equal the identity matrix of shape (l*m, l*m)"
            raise ValueError(msg)

    @staticmethod
    def _assert_matrix_ma_mb_properties(m_A: npt.NDArray, m_B: npt.NDArray):
        # (1) exactly 3 non-zero entries in any row or column
        if not (np.count_nonzero(m_A, axis=0) == 3).all():
            msg = "Matrix A should have exactly 3 non-zero entries in each column"
            raise ValueError(msg)
        if not (np.count_nonzero(m_A, axis=1) == 3).all():
            msg = "Matrix A should have exactly 3 non-zero entries in each row"
            raise ValueError(msg)
        if not (np.count_nonzero(m_B, axis=0) == 3).all():
            msg = "Matrix B should have exactly 3 non-zero entries in each column"
            raise ValueError(msg)
        if not (np.count_nonzero(m_B, axis=1) == 3).all():
            msg = "Matrix B should have exactly 3 non-zero entries in each row"
            raise ValueError(msg)

        # (2) AB = BA
        if not np.allclose(m_A @ m_B, m_B @ m_A):
            msg = "A*B should equal B*A"
            raise ValueError(msg)

    def _get_qubit_coords(
        self,
    ) -> tuple[dict[int, Qubit], dict[int, Qubit], dict[int, Qubit], dict[int, Qubit]]:
        """
        Attempt to lay out the qubits on a square-ish grid, the size of which is dictated
        by the periodic boundary conditions of the torus the code may or may not
        naturally lie on. Not all BB codes permit a toric layout; see Lemma 4 of the
        paper for more details. Hence, if the code is one of codes listed in Table 3 of
        the paper, this code will work, but is not guaranteed to work (and likely won't)
        for codes outside of these.

        We arbitrarily pick the first row of the Hx matrix to be the bottom-left-most
        X ancilla qubit. We may then follow the scheduling of the syndrome measurement to
        determine the placement of the data qubits and rest of the ancillas.

        The method used is to start in the bottom left of the matrix, calling this
        Qubit(0,0) arbitrarily, then move along the rows assigning Qubits(1,0),
        ubits(2,0) etc as per the construction described in the paper, detailed above.
        There are rules there for how the submatrices connect to each other which dictate
        what the local connections are and what the long range connections are.

        NB: The [[90, 8, 10]] code is a special case and uses different parameters, but
        should still arrive at a similar grid layout.

        Returns
        -------
        Tuple[Dict[int, Qubit], Dict[int, Qubit], Dict[int, Qubit], Dict[int, Qubit]]
            A tuple of dictionaries mapping the index of rows/columns of the parity
            check matrices to qubits, in order:
                left data qubits, right data qubits, x ancillas, z ancillas.
            For instance, column 0 of the Hx (WLOG Hz) matrix is the first data qubit in
            the "L" (R) group of data qubits, and would be described in dictionary 0 (1)
            with entry {0: Qubit(...)}. The first row of Hx (Hz) is the first X (Z)
            ancilla and is described by dictionary 2 (3) with entry {0: Qubit(...)}.
        """
        # assign the first row of the Hx matrix to be the bottom-left-most qubit
        x_coord, y_coord, matrix_index_of_next_qubit = 0, 0, 0
        x_lookup: dict[int, Qubit] = {}
        z_lookup: dict[int, Qubit] = {}
        dl_lookup: dict[int, Qubit] = {}
        dr_lookup: dict[int, Qubit] = {}

        # when scanning over qubit connections, the bottom row alternates Z-anc to L-data
        # and alternate rows between R-data and X-anc, so use the qubit coord mod 2 to
        # access the correct dictionary - y % 2 is which type of row we are, x % 2 is whether
        # we are assigning a data or an ancilla
        dict_lookup = ((z_lookup, dl_lookup), (dr_lookup, x_lookup))

        # when scanning over rows, the submatrix we must check to find the adjacent qubit
        # changes based on the type of qubit we are currently looking at.
        # keep the relevant dictionaries in tuples so that they may be accessed using the
        # current coordinate mod 2. for a deeper explanation of the choice of matrices here,
        # refer to Fig 4 of the paper, along with Lemma 3 and 4.
        submatrix_lookup_y = (
            # z_anc -> R_d, L_d -> x_anc
            (self.m_A_submatrices[1].T, self.m_A_submatrices[1].T),
            # R_d -> z_anc, x_anc -> l_d
            (self.m_A_submatrices[2], self.m_A_submatrices[2]),
        )
        if (
            self.param_l == 15
            and self.param_m == 3
            and self.m_A_powers == [9, 1, 2]
            and self.m_B_powers == [0, 2, 7]
        ):
            submatrix_lookup_x = (
                # z_anc -> L_d, L_d -> z_anc
                (self.m_B_submatrices[2].T, self.m_B_submatrices[0]),
                # R_d -> x_anc, x_anc -> R_d
                (self.m_B_submatrices[2].T, self.m_B_submatrices[0]),
            )
        else:
            submatrix_lookup_x = (
                (self.m_B_submatrices[2].T, self.m_B_submatrices[1]),
                (self.m_B_submatrices[2].T, self.m_B_submatrices[1]),
            )
        max_iters = 6 * self.n  # should only need 2*n
        current_iter_index = 0
        while current_iter_index < max_iters:
            dict_lookup[y_coord % 2][x_coord % 2][matrix_index_of_next_qubit] = Qubit(
                Coord2D(x_coord, y_coord)
            )
            matrix_index_of_next_qubit = np.where(
                submatrix_lookup_x[y_coord % 2][x_coord % 2][matrix_index_of_next_qubit]
                == 1
            )[0][0]
            x_coord += 1
            if matrix_index_of_next_qubit in dict_lookup[y_coord % 2][x_coord % 2]:
                # reached the torus boundary, so move up a row via the relevant connection
                matrix_index_of_next_qubit = np.where(
                    submatrix_lookup_y[y_coord % 2][x_coord % 2][
                        matrix_index_of_next_qubit
                    ]
                    == 1
                )[0][0]
                x_coord = 0
                y_coord += 1
                if matrix_index_of_next_qubit in dict_lookup[y_coord % 2][x_coord % 2]:
                    # reached torus y boundary, so we're done
                    break
            current_iter_index += 1
        else:
            msg = (
                "Max iteration limit exceeded for coord placement; consider"
                " changing code parameters"
            )
            raise ValueError(msg)

        return dl_lookup, dr_lookup, x_lookup, z_lookup

    def _get_stabilisers(self) -> list[Stabiliser]:
        """
        Given the appropriate matrices describing the code, construct
        the stabilisers with valid scheduling.

        Returns
        -------
        List[Stabiliser]
            List of alternating X and Z stabilisers. Starting with X.
        """
        # use the order in equation (9), Section 5
        # this order was found empirically and has no easy intuitive explanation.
        # essentially, we just pick one check at a time from each A or B sub matrix
        # and order them such that they are hopefully distance preserving.
        zip_x_stab_qubit_coords = [
            # 2 indicates left qubits, which for X type is A
            # 3 is then right which is B
            (self._mat_col_to_l_data_map, self.m_A_submatrices[1]),  # A2
            (self._mat_col_to_r_data_map, self.m_B_submatrices[1]),  # B2
            (self._mat_col_to_r_data_map, self.m_B_submatrices[0]),  # B1
            (self._mat_col_to_r_data_map, self.m_B_submatrices[2]),  # B3
            (self._mat_col_to_l_data_map, self.m_A_submatrices[0]),  # A1
            (self._mat_col_to_l_data_map, self.m_A_submatrices[2]),  # A3
        ]
        zip_z_stab_qubit_coords = [
            # 2 indicates left qubits, which for Z type is B
            # 3 is then right which is A
            (self._mat_col_to_r_data_map, self.m_A_submatrices[0]),  # A1
            (self._mat_col_to_r_data_map, self.m_A_submatrices[2]),  # A3
            (self._mat_col_to_l_data_map, self.m_B_submatrices[0]),  # B1
            (self._mat_col_to_l_data_map, self.m_B_submatrices[1]),  # B2
            (self._mat_col_to_l_data_map, self.m_B_submatrices[2]),  # B3
            (self._mat_col_to_r_data_map, self.m_A_submatrices[1]),  # A2
        ]
        x_stabilisers = []
        z_stabilisers = []
        for check_index in range(self.param_l * self.param_m):
            # X stabilisers
            x_stabilisers.append(
                Stabiliser(
                    paulis=[
                        None,
                        *(
                            PauliX(
                                col_ind_to_qubit_dict[
                                    np.nonzero(matrix[check_index, :])[0][0]
                                ]
                            )
                            for col_ind_to_qubit_dict, matrix in zip_x_stab_qubit_coords
                        ),
                    ],
                    ancilla_qubit=self._mat_col_to_x_anc_map[check_index],
                )
            )

            # Z stabilisers
            z_stabilisers.append(
                Stabiliser(
                    paulis=[
                        *(
                            PauliZ(
                                col_ind_to_qubit_dict[
                                    np.nonzero(matrix[:, check_index])[0][0]
                                ]
                                # indices of non-zero elements in row of matrix
                            )
                            for col_ind_to_qubit_dict, matrix in zip_z_stab_qubit_coords
                        ),
                        None,
                    ],
                    ancilla_qubit=self._mat_col_to_z_anc_map[check_index],
                )
            )

        return x_stabilisers + z_stabilisers

    def _get_logicals(
        self,
    ) -> tuple[list[list[PauliX[Coord2D]]], list[list[PauliZ[Coord2D]]]]:
        """
        Will return a tuple of lists of the X and Z logicals, respectively.
        They are paired up in order, that is, they anticommute when the
        indices of the two lists are aligned and commute otherwise.

        Returns
        -------
        Tuple[List[List[PauliX[Coord2D]]], List[List[PauliZ[Coord2D]]]]
            Tuple of X and Z logical operators, in order such that
            x_logicals[i],z_logicals[j] anti-commute exactly when i=j
            and commute otherwise.
        """
        # get a subset of logicals from css_code, and compute the rest
        x_logs, _ = css_code_compute_logicals(self.m_Hx, self.m_Hz)

        # get f and g,h from X logicals
        # X logicals are of the form X(f,0) and X(g,h), eq. (16)
        f_vecs, g_vecs, h_vecs = [], [], []
        for x_log in x_logs:
            x_log_list = x_log.reshape(-1).tolist()
            half = len(x_log_list) // 2
            if all(x == 0 for x in x_log_list[half:]):
                f_vecs.append(x_log_list[:half])
            else:
                g_vecs.append(x_log_list[:half])
                h_vecs.append(x_log_list[half:])

        # logical operators are cyclic (as are stabilisers) so we
        # may permute them by monomials.
        # we have a subset of all possible f,g,h, so find the rest
        # by multiplying in each monomial to get a set of |M|=lm
        all_monomials_of_lm = [
            Monomial(x, y, self.param_l, self.param_m)
            for x, y in product(range(self.param_l), range(self.param_m))
        ]
        all_f_vecs = f_vecs.copy()
        all_gT_vecs = [
            Polynomial.from_vec(vec, self.param_l, self.param_m).reverse().to_vec()
            for vec in g_vecs
        ]
        all_hT_vecs = [
            Polynomial.from_vec(vec, self.param_l, self.param_m).reverse().to_vec()
            for vec in h_vecs
        ]
        for list_to_extend, list_to_iterate, is_g_or_h in [
            (all_f_vecs, f_vecs, False),
            (all_gT_vecs, g_vecs, True),
            (all_hT_vecs, h_vecs, True),
        ]:
            for vec in list_to_iterate:
                vec_as_poly = Polynomial.from_vec(vec, self.param_l, self.param_m)
                for mon in all_monomials_of_lm:
                    if is_g_or_h:
                        shifted_vec = (
                            vec_as_poly.reverse().mult_by_monomial(mon).to_vec()
                        )
                    else:
                        shifted_vec = vec_as_poly.mult_by_monomial(mon).to_vec()
                    if not np.any(np.equal(list_to_extend, shifted_vec).all(1)):
                        list_to_extend.append(shifted_vec)

        # now combine f,g,h to make logical operators
        # create the unprimed block, X(f, 0) and Z(h.T, g.T)
        all_f_logs = [list(f) + [0] * (self.n // 2) for f in all_f_vecs]
        all_gh_logs = [list(hT) + list(gT) for gT, hT in zip(all_gT_vecs, all_hT_vecs)]

        # find pairs with correct commutation relations
        # find anticommuting pairs of logical operators
        unprime_x_logs, unprime_z_logs = _find_anticommuting_pairs(
            all_f_logs, all_gh_logs, self.k // 2, self.n
        )

        # get primed block from unprimed
        # X(g, h) and Z(0, f.T)
        prime_x_logs, prime_z_logs = [], []
        for up_x_log in unprime_x_logs:
            prime_z_logs.append(
                [0] * (len(up_x_log) // 2)
                + Polynomial.from_vec(
                    up_x_log[: len(up_x_log) // 2], self.param_l, self.param_m
                )
                .reverse()
                .to_vec()
            )
        for up_z_log in unprime_z_logs:
            prime_x_logs.append(
                Polynomial.from_vec(
                    up_z_log[(len(up_z_log) // 2) :], self.param_l, self.param_m
                )
                .reverse()
                .to_vec()
                + Polynomial.from_vec(
                    up_z_log[: (len(up_z_log) // 2)], self.param_l, self.param_m
                )
                .reverse()
                .to_vec()
            )

        # convert binary arrays to arrays of Pauli terms.
        # the first coordinate is whether its in A or B submatrix
        # the second coordinate is the relative coordinate within A or B
        dict_lookup = {2: self._mat_col_to_l_data_map, 3: self._mat_col_to_r_data_map}
        x_logicals_as_paulis = [
            [
                PauliX(
                    dict_lookup[2 + j // (self.param_l * self.param_m)][
                        j % (self.param_l * self.param_m)
                    ]
                )
                for j in np.where(logical)[0]
            ]
            for logical in unprime_x_logs + prime_x_logs
        ]
        z_logicals_as_paulis = [
            [
                PauliZ(
                    dict_lookup[2 + j // (self.param_l * self.param_m)][
                        j % (self.param_l * self.param_m)
                    ]
                )
                for j in np.where(logical)[0]
            ]
            for logical in unprime_z_logs + prime_z_logs
        ]

        return x_logicals_as_paulis, z_logicals_as_paulis
