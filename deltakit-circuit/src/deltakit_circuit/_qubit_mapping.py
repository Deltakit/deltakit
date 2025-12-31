# (c) Copyright Riverlane 2020-2025.
"""Module which defines mappings between qubits that deltakit_circuit uses and indices
which stim uses."""

from collections.abc import Iterable

from deltakit_circuit._qubit_identifiers import Qubit, T


def default_qubit_mapping(qubits: Iterable[Qubit[T]]) -> dict[Qubit[T], int]:
    """Generate a mapping from each qubit to an int by default.
    If the unique identifier for the qubit is an int then the unique
    identifier is the value for that qubit.
    If the unique identifier is any other type the value in the mapping is the
    position in the iteration.

    Examples
    --------
    input: [Qubit(1), Qubit(2), Qubit(3), Qubit(4)]
    output: {Qubit(1): 1, Qubit(2): 2, Qubit(3): 3, Qubit(4): 4}

    input: [Qubit((1, 0)), Qubit((0, 1)), Qubit((0, 0))]
    output: {Qubit((1, 0)): 0, Qubit((0, 1)): 1, Qubit((0, 0)): 2}

    Parameters
    ----------
    qubits : Iterable[Qubit[T]]
        The list of qubits which need mapping.

    Raises
    ------
    ValueError
        If the types of all qubits' unique identifiers are not identical

    Returns
    -------
    Dict[Qubit[T], int]
        The mapping of qubit to integers.
    """
    if len({type(qubit.unique_identifier) for qubit in qubits}) > 1:
        msg = "All Qubit.unique_identifier fields must be of the same type"
        raise TypeError(msg)

    mapping = {}
    for index, qubit in enumerate(qubits):
        qubit_id = qubit.unique_identifier
        mapping[qubit] = qubit_id if isinstance(qubit_id, int) else index
    return mapping
