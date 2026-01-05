# (c) Copyright Riverlane 2020-2025.
"""Classes which represent independent Pauli noise for qubits."""

from __future__ import annotations

import math
from typing import ClassVar, get_args

from deltakit_circuit._qubit_identifiers import Qubit, T
from deltakit_circuit.noise_channels._abstract_noise_channels import (
    MultiProbabilityNoiseChannel,
    OneQubitNoiseChannel,
    OneQubitOneProbabilityNoiseChannel,
    TwoQubitNoiseChannel,
)


class PauliXError(OneQubitOneProbabilityNoiseChannel[T]):
    """Applies a Pauli X with a given probability.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to apply bit flip noise to.
    probability: float
        A single float specifying the probability of applying an X operation.

    Notes
    -----
    | Pauli Mixture:
    |   ``1-p: I``
    |   ``p : X``
    """

    stim_string: ClassVar[str] = "X_ERROR"


class PauliYError(OneQubitOneProbabilityNoiseChannel[T]):
    """Applies a Pauli Y with a given probability.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to apply Y flip noise to.
    probability: float
        A single float specifying the probability of applying a Y operation.

    Notes
    -----
    | Pauli Mixture:
    |   ``1-p: I``
    |   ``p : Y``
    """

    stim_string: ClassVar[str] = "Y_ERROR"


class PauliZError(OneQubitOneProbabilityNoiseChannel[T]):
    """Applies a Pauli Z with a given probability.

    Parameters
    ----------
    qubit: Qubit[T] | T
        The qubit to apply phase flip noise to.
    probability: float
        A single float specifying the probability of applying a Z operation.

    Notes
    -----
    | Pauli Mixture:
    |   ``1-p: I``
    |   ``p : Z``
    """

    stim_string: ClassVar[str] = "Z_ERROR"


class PauliChannel1(OneQubitNoiseChannel[T], MultiProbabilityNoiseChannel[T]):
    """A single qubit Pauli error channel with explicitly specified
    probabilities for each case.

    Parameters
    ----------
    qubit : Qubit[T] | T
        The qubit to apply the custom noise channel to.
    p_x : float, optional
        Probability of applying an X operation, by default 0.0
    p_y : float, optional
        Probability of applying a Y operation, by default 0.0
    p_z : float, optional
        Probability of applying a Z operation, by default 0.0

    Notes
    -----
    | Pauli Mixture:
    |   ``1-px-py-pz: I``
    |   ``px: X``
    |   ``py: Y``
    |   ``pz: Z``
    """

    stim_string: ClassVar[str] = "PAULI_CHANNEL_1"

    def __init__(
        self,
        qubit: Qubit[T] | T,
        p_x: float = 0.0,
        p_y: float = 0.0,
        p_z: float = 0.0,
        *,
        tag: str | None = None,
    ):
        super().__init__(qubit=qubit, probabilities=(p_x, p_y, p_z), tag=tag)
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z

    def approx_equals(
        self, other: object, *, rel_tol: float = 1e-9, abs_tol: float = 0
    ) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.qubit == other.qubit
            and len(self.probabilities) == len(other.probabilities)
            and all(
                math.isclose(self_prob, other_prob, rel_tol=rel_tol, abs_tol=abs_tol)
                for self_prob, other_prob in zip(
                    self.probabilities, other.probabilities, strict=True
                )
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.qubit == other.qubit
            and self.probabilities == other.probabilities
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self.qubit, self.probabilities))

    def __repr__(self) -> str:
        tag_repr = f"[{self.tag}]" if self.tag is not None else ""
        return (
            f"{self.stim_string}{tag_repr}({self.qubit}, p_x={self.p_x}, "
            f"p_y={self.p_y}, p_z={self.p_z})"
        )


class PauliChannel2(MultiProbabilityNoiseChannel[T], TwoQubitNoiseChannel[T]):
    """A two-qubit Pauli error channel with explicitly specified probabilities
    for each case.

    Parameters
    ----------
    qubit1: Qubit[T] | T
        The first qubit in the noise channel.
    qubit2: Qubit[T] | T
        The second qubit in the noise channel.
    p_ix : float, optional
        Probability of applying an IX operation, by default 0.0
    p_iy : float, optional
        Probability of applying an IY operation, by default 0.0
    p_iz : float, optional
        Probability of applying an IZ operation, by default 0.0
    p_xi : float, optional
        Probability of applying an XI operation, by default 0.0
    p_xx : float, optional
        Probability of applying an XX operation, by default 0.0
    p_xy : float, optional
        Probability of applying an XY operation, by default 0.0
    p_xz : float, optional
        Probability of applying an XZ operation, by default 0.0
    p_yi : float, optional
        Probability of applying a YI operation, by default 0.0
    p_yx : float, optional
        Probability of applying a YX operation, by default 0.0
    p_yy : float, optional
        Probability of applying a YY operation, by default 0.0
    p_yz : float, optional
        Probability of applying a YZ operation, by default 0.0
    p_zi : float, optional
        Probability of applying a ZI operation, by default 0.0
    p_zx : float, optional
        Probability of applying a ZX operation, by default 0.0
    p_zy : float, optional
        Probability of applying a ZY operation, by default 0.0
    p_zz : float, optional
        Probability of applying a ZZ operation, by default 0.0

    Notes
    -----
    | Pauli Mixture:
    |   ``1-pix-piy-piz-pxi-pxx-pxy-pxz-pyi-pyx-pyy-pyz-pzi-pzx-pzy-pzz: II``
    |   ``p_ix: IX``
    |   ``p_iy: IY``
    |   ``p_iz: IZ``
    |   ``p_xi: XI``
    |   ``p_xx: XX``
    |   ``p_xy: XY``
    |   ``p_xz: XZ``
    |   ``p_yi: YI``
    |   ``p_yx: YX``
    |   ``p_yy: YY``
    |   ``p_yz: YZ``
    |   ``p_zi: ZI``
    |   ``p_zx: ZX``
    |   ``p_zy: ZY``
    |   ``p_zz: ZZ``

    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,
    # pylint: disable=too-many-locals

    stim_string: ClassVar[str] = "PAULI_CHANNEL_2"

    def __init__(  # noqa: PLR0913
        self,
        qubit1: Qubit[T] | T,
        qubit2: Qubit[T] | T,
        p_ix: float = 0.0,
        p_iy: float = 0.0,
        p_iz: float = 0.0,
        p_xi: float = 0.0,
        p_xx: float = 0.0,
        p_xy: float = 0.0,
        p_xz: float = 0.0,
        p_yi: float = 0.0,
        p_yx: float = 0.0,
        p_yy: float = 0.0,
        p_yz: float = 0.0,
        p_zi: float = 0.0,
        p_zx: float = 0.0,
        p_zy: float = 0.0,
        p_zz: float = 0.0,
        *,
        tag: str | None = None,
    ):
        probabilities = (
            p_ix,
            p_iy,
            p_iz,
            p_xi,
            p_xx,
            p_xy,
            p_xz,
            p_yi,
            p_yx,
            p_yy,
            p_yz,
            p_zi,
            p_zx,
            p_zy,
            p_zz,
        )
        super().__init__(
            qubit1=qubit1, qubit2=qubit2, probabilities=probabilities, tag=tag
        )
        self.p_ix = p_ix
        self.p_iy = p_iy
        self.p_iz = p_iz
        self.p_xi = p_xi
        self.p_xx = p_xx
        self.p_xy = p_xy
        self.p_xz = p_xz
        self.p_yi = p_yi
        self.p_yx = p_yx
        self.p_yy = p_yy
        self.p_yz = p_yz
        self.p_zi = p_zi
        self.p_zx = p_zx
        self.p_zy = p_zy
        self.p_zz = p_zz

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.qubits == other.qubits
            and self.probabilities == other.probabilities
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self._qubit1, self._qubit2, self.probabilities))

    def __repr__(self) -> str:
        tag_repr = f"[{self.tag}]" if self.tag is not None else ""
        return (
            f"{self.stim_string}{tag_repr}"
            f"(qubit1={self._qubit1}, qubit2={self._qubit2}, "
            f"p_ix={self.p_ix}, p_iy={self.p_iy}, p_iz={self.p_iz}, "
            f"p_xi={self.p_xi}, p_xx={self.p_xx}, p_xy={self.p_xy}, "
            f"p_xz={self.p_xz}, p_yi={self.p_yi}, p_yx={self.p_yx}, "
            f"p_yy={self.p_yy}, p_yz={self.p_yz}, p_zi={self.p_zi}, "
            f"p_zx={self.p_zx}, p_zy={self.p_zy}, p_zz={self.p_zz})"
        )


_PauliNoise = (
    PauliXError[T]
    | PauliYError[T]
    | PauliZError[T]
    | PauliChannel1[T]
    | PauliChannel2[T]
)
ALL_PAULI_NOISE: frozenset[type[_PauliNoise]] = frozenset(get_args(_PauliNoise))
