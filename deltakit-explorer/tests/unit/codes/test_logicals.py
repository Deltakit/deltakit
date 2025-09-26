import pytest
import stim
from deltakit_explorer.codes._logicals import (
    paulistring_to_operator,
)

import deltakit_circuit as circuit


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
