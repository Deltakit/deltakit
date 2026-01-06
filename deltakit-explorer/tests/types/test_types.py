# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from deltakit_circuit.gates import PauliBasis

from deltakit_explorer.enums import DataFormat, QECECodeType, QECExperimentType
from deltakit_explorer.types import (
    BinaryDataType,
    CircuitParameters,
    DataString,
    MatrixSpecifications,
    PhysicalNoiseModel,
    QECExperiment,
    QECExperimentDefinition,
    RAMData,
    Sizes,
    TypedData,
    TypedDataFile,
    TypedDataString,
)
from deltakit_explorer.types._types import JSONable


class TestQECExperimentDefinitionAndNested:

    @pytest.mark.parametrize(
            ("mspec", "mres"),
            [
                (None, None),
                (MatrixSpecifications(0, 1, [23, 45], [67, 89]),
                {"paramL": 0, "paramM": 1, "mAPowers": [23, 45], "mBPowers": [67, 89]}),
            ]
    )
    @pytest.mark.parametrize(
            ("sizes", "sres"),
            [
                (None, None),
                (Sizes(sizes=[1, 2, 3]), {"sizes": [1, 2, 3]})
            ]
    )
    def test_get_experiment_definition(self, mspec, mres, sizes, sres):
        definition = QECExperimentDefinition(
            experiment_type=QECExperimentType.QUANTUM_MEMORY,
            code_type=QECECodeType.BIVARIATE_BICYCLE,
            observable_basis=PauliBasis.X,
            num_rounds=3,
            basis_gates=["CX", "H"],
            parameters=CircuitParameters(mspec, sizes),
        )
        gql = definition.get_parameters_gql_string()
        assert gql == {
            "sizes": sres,
            "matrixSpecifications": mres,
        }

    def test_get_experiment_definition_no_params(self):
        definition = QECExperimentDefinition(
            experiment_type=QECExperimentType.QUANTUM_MEMORY,
            code_type=QECECodeType.BIVARIATE_BICYCLE,
            observable_basis=PauliBasis.X,
            num_rounds=3,
            basis_gates=["CX", "H"],
        )
        gql = definition.get_parameters_gql_string()
        assert gql is None

    def test_get_rotated_planar_z_quantum_memory(self):
        definition = QECExperimentDefinition.get_rotated_planar_z_quantum_memory(
            3, 3, basis_gates=["CX", "H"],
        )
        assert definition.code_type == QECECodeType.ROTATED_PLANAR
        assert definition.observable_basis == PauliBasis.Z
        assert definition.experiment_type == QECExperimentType.QUANTUM_MEMORY
        assert definition.get_parameters_gql_string() == {
            'sizes': {'sizes': [3, 3]}, 'matrixSpecifications': None,
        }

    def test_matrix_specifications(self):
        spec = MatrixSpecifications(0, 1, [23, 45], [67, 89])
        assert spec.to_gql() == {
            "paramL": 0,
            "paramM": 1,
            "mAPowers": [23, 45],
            "mBPowers": [67, 89],
        }

    def test_sizes(self):
        spec = Sizes(sizes=[1, 2, 3])
        assert spec.to_gql() == {
            "sizes": [1, 2, 3]
        }

    @pytest.mark.parametrize(
            ("mspec", "mres"),
            [
                (None, None),
                (MatrixSpecifications(0, 1, [23, 45], [67, 89]),
                {"paramL": 0, "paramM": 1, "mAPowers": [23, 45], "mBPowers": [67, 89]}),
            ]
    )
    @pytest.mark.parametrize(
            ("sizes", "sres"),
            [
                (None, None),
                (Sizes(sizes=[1, 2, 3]), {"sizes": [1, 2, 3]})
            ]
    )
    def test_generate_circuit_parameters(self, mspec, mres, sizes, sres):
        gql = CircuitParameters(mspec, sizes).to_gql()
        assert gql == {
            "sizes": sres,
            "matrixSpecifications": mres,
        }

    def test_from_matrix_specification(self):
        parameters = CircuitParameters.from_matrix_specification(
            6, 5, [3, 1, 2], [3, 1, 0],
        )
        assert parameters.to_gql() == {
            "sizes": None,
            "matrixSpecifications": {
                "paramL": 6,
                "paramM": 5,
                "mAPowers": [3, 1, 2],
                "mBPowers": [3, 1, 0],
            }
        }


class TestInternalClasses:

    @pytest.mark.parametrize(
        "data",
        [np.array([[0, 0, 1, 0, 1]] * 3), [[0, 0, 1, 0, 1]] * 3]
    )
    @pytest.mark.parametrize(
        "data_format", [DataFormat.B8, DataFormat.CSV, DataFormat.F01]
    )
    def test_ram_data_ok(self, data, data_format):
        holder = RAMData(content=data, data_format=data_format)
        assert holder.as_01_string() == "00101\n" * 3
        assert holder.as_b8_bytes() == b"\x14" * 3
        assert np.allclose(
            holder.as_numpy(), [[0, 0, 1, 0, 1]] * 3)

        assert holder.get_width() == 5

    def test_ram_data_width_undetected(self):
        data = RAMData(data_format=DataFormat.B8, content=[])
        assert data.get_width() == -1

    def test_binary_data_as_data_string_raises_no_format(self):
        data = BinaryDataType([[0, 1], [0, 1]])
        with pytest.raises(NotImplementedError):
            data.as_data_string(DataFormat.TEXT)

    def test_binary_data_raises_on_wrong_data(self):
        msg = f"Not supported argument of type {str}"
        with pytest.raises(ValueError, match=msg):
            BinaryDataType("some string")


class TestTypedDataString:

    class NewTypeData(TypedData):

        def as_01_string(self):
            pass

        def as_b8_bytes(self):
            pass

        def as_numpy(self):
            pass


    def test_typed_data_string_as_datastring(self):
        ds = TypedDataString(DataFormat.F01, DataString("0101"), 4)
        assert ds.as_data_string() == "duck://30313031"

    def test_typed_data_string_from_data(self):
        ds = TypedDataString.from_data(
            RAMData(DataFormat.CSV, [[0, 1, 0, 1]], 4)
        )
        assert ds.as_data_string() == "duck://302c312c302c310a"

    def test_typed_data_string_from_file(self):
        TypedDataString.from_data(
            TypedDataFile(
                DataFormat.F01,
                pathlib.Path(__file__).parent / "../resources/rep_code_noisy_stim_dets.01",
            )
        )

    def test_typed_data_string_raises_with_wrong_data_type(self):
        ds = TypedDataString(DataFormat.TEXT, DataString("0101"), 4)
        with pytest.raises(NotImplementedError):
            ds.as_01_string()
        with pytest.raises(NotImplementedError):
            ds.as_numpy()
        with pytest.raises(NotImplementedError):
            ds.as_b8_bytes()

    def test_typed_data_string_from_data_ok(self):
        ds = TypedDataString.from_data(RAMData(DataFormat.F01, [[0, 1]]))
        assert ds.as_data_string() == "duck://30310a"

    def test_typed_data_string_from_data_raises(self):
        msg = "Cannot extract binary data from .*"
        with pytest.raises(ValueError, match=msg):
            TypedDataString.from_data(
                TestTypedDataString.NewTypeData(DataFormat.B8, None)
            )


class TestTypedDataFile:

    @pytest.mark.parametrize(
        ("data", "data_format"),
        [
            (b"00101\n00101", DataFormat.F01),
            (b"0,0,1,0,1\n0,0,1,0,1", DataFormat.CSV),
            (b"\x14\x14", DataFormat.B8),
        ]
    )
    def test_typed_data_files(self, data, data_format, tmp_path):
        p = tmp_path / "data"
        with pathlib.Path.open(p, "wb") as f:
            f.write(data)
        holder = TypedDataFile(data_format=data_format, content=p, data_width=5)
        assert holder.as_01_string().strip() == "00101\n00101"
        assert holder.as_b8_bytes() == b"\x14\x14"
        assert np.allclose(holder.as_numpy(), [[0, 0, 1, 0, 1]] * 2)

    def test_typed_data_file_raises_with_wrong_data_type(self):
        ds = TypedDataFile(DataFormat.TEXT, DataString("0101"), 4)
        with pytest.raises(NotImplementedError):
            ds.as_01_string()
        with pytest.raises(NotImplementedError):
            ds.as_numpy()
        with pytest.raises(NotImplementedError):
            ds.as_b8_bytes()

    def test_ramdata_as_numpy(self):
        data = RAMData(data_format=DataFormat.B8, content=[[1,0],[0,1]])
        arr = data.as_numpy()
        assert arr.shape == (2, 2)

    def test_typeddatafile_not_implemented(self):
        with TemporaryDirectory() as direct:
            file = pathlib.Path(direct) / "file.txt"
            file.write_text("data")
            tdf = TypedDataFile(data_format="INVALID", content=file)
            with pytest.raises(NotImplementedError):
                tdf.as_01_string()

    @pytest.mark.parametrize("num_parts", [1, 2, 3])
    def test_binarydatatype_split_and_combine(self, num_parts):
        data = BinaryDataType([[1,0],[0,1], [1,0],[0,1]], data_format=DataFormat.B8)
        parts = data.split(num_parts)
        assert len(parts) == num_parts
        combined = BinaryDataType.combine(parts)
        assert np.allclose(combined.as_numpy(), data.as_numpy())


class TestQECExperiment:

    def test_qec_experiment_from_circuit_and_measurements_no_sweeps(self):
        exp = QECExperiment.from_circuit_and_measurements(
            pathlib.Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim",
            pathlib.Path(__file__).parent / "../resources/rep_code_noisy_stim_dets.01",
            DataFormat.F01,
        )
        assert exp.measurements.as_numpy().shape == (5000, 8)

    def test_qec_experiment_from_circuit_and_measurements_with_sweeps(self):
        exp = QECExperiment.from_circuit_and_measurements(
            pathlib.Path(__file__).parent / "../resources/rep_code_mutated_default_noise_data.stim",
            pathlib.Path(__file__).parent / "../resources/rep_code_noisy_stim_dets.01",
            DataFormat.F01,
            pathlib.Path(__file__).parent / "../resources/rep_code_noisy_stim_dets.01",
            DataFormat.F01,
        )
        assert exp.measurements.as_numpy().shape == (5000, 8)
        assert exp.sweep_bits.as_numpy().shape == (5000, 8)

    def test_qec_experiment_from_circuit_and_measurements_with_failed_sweeps_raises(self):
        msg = "sweep_path and sweep_format should be both provided, or should both be None."
        with pytest.raises(ValueError, match=msg):
            QECExperiment.from_circuit_and_measurements(
                pathlib.Path("tests/resources/rep_code_mutated_default_noise_data.stim"),
                pathlib.Path("tests/resources/rep_code_noisy_stim_dets.01"),
                DataFormat.F01,
                pathlib.Path("tests/resources/rep_code_noisy_stim_dets.01"),
            )


class TestNoiseModel:

    def test_get_ion_trap_noise(self):
        model = PhysicalNoiseModel.get_ion_trap_noise()
        assert pytest.approx(model.time_reset) == 1e-5

    def test_get_superconducting_noise(self):
        model = PhysicalNoiseModel.get_superconducting_noise()
        assert pytest.approx(model.time_reset) == 160e-9

    def test_get_floor_superconducting_noise(self):
        model = PhysicalNoiseModel.get_floor_superconducting_noise()
        assert pytest.approx(model.time_reset) == 160e-9
        assert pytest.approx(model.p_reset_error) == 1.99e-3


class TestJSONable:

    @dataclass
    class DummyJSONable(JSONable):
        data: bytes

    def test_jsonable_fails(self):
        obj = TestJSONable.DummyJSONable(b"1243")
        msg = f"Object of type {bytes} is not a dataclass or serialisable object."
        with pytest.raises(ValueError, match=msg):
            obj.to_json()
