from dataclasses import dataclass
from pathlib import Path

import numpy
import numpy.typing as npt
import pandas as pd
import pytest

from deltakit_explorer.analysis.budget._post_processing import (
    _filter_non_close_noise_parameters,
)


@dataclass
class ErrorBudgetingResults:
    data: pd.DataFrame
    noise_parameters: npt.NDArray[numpy.floating]
    parameter_names: list[str]

    def to_tuple(self) -> tuple[pd.DataFrame, npt.NDArray[numpy.floating], list[str]]:
        return self.data, self.noise_parameters, self.parameter_names


@pytest.fixture
def error_budgeting_data() -> ErrorBudgetingResults:
    _resources_folder = Path(__file__).parent.parent.parent / "resources"
    _data_file = _resources_folder / "error_budget_data.csv"
    data = pd.read_csv(_data_file)
    return ErrorBudgetingResults(
        data,
        numpy.array([5e-4, 4e-3, 5e-3, 5e-3, 5e-3]),
        ["1q error", "2q error", "Reset error", "Measurement error", "Readout flip"],
    )


def test_data_matches_csv(error_budgeting_data: ErrorBudgetingResults) -> None:
    """Test that the hard-coded parts in the data fixture agree with the CSV file."""
    header_names: frozenset[str] = frozenset(error_budgeting_data.data.columns.values)
    for name in error_budgeting_data.parameter_names:
        assert f"noise_{name}" in header_names

    for noise_name, noise_value in zip(
        error_budgeting_data.parameter_names,
        error_budgeting_data.noise_parameters,
        strict=True,
    ):
        values = error_budgeting_data.data[f"noise_{noise_name}"]
        assert numpy.any(numpy.isclose(values, noise_value))


def test_filter_non_close_noise_parameters(
    error_budgeting_data: ErrorBudgetingResults,
) -> None:
    data, parameters, names = error_budgeting_data.to_tuple()
    all_noises_index = [f"noise_{name}" for name in names]
    data_frame = _filter_non_close_noise_parameters(data, parameters, names)
    filtered_columns_data_frame = data_frame[all_noises_index]
    for row in filtered_columns_data_frame.to_numpy():
        numpy.testing.assert_allclose(row, parameters)


def test_filter_non_close_noise_parameters_random(
    error_budgeting_data: ErrorBudgetingResults,
    random_generator: numpy.random.Generator,
) -> None:
    data, _, names = error_budgeting_data.to_tuple()
    all_noises_index = [f"noise_{name}" for name in names]
    random_row_index = random_generator.integers(data.shape[0])
    random_parameters = data[all_noises_index].to_numpy()[random_row_index, :]
    data_frame = _filter_non_close_noise_parameters(data, random_parameters, names)
    filtered_columns_data_frame = data_frame[all_noises_index]
    for row in filtered_columns_data_frame.to_numpy():
        numpy.testing.assert_allclose(row, random_parameters)
