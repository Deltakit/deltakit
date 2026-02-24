import pytest
import numpy as np
import matplotlib.pyplot as plt
"""Tests for detection probability patch visualization."""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from deltakit_explorer.plotting import DetectionProbabilityPatchResult, DetectorDistribution, AggregationMethod, QecCodeType, plot_detection_probability_patch, aggregate_detection_probability, create_inset_axes

class TestDetectionProbabilityPatchResult:
    """Test the DetectionProbabilityPatchResult dataclass."""

    def test_create_basic_result(self):
        """Test creating a basic result object with 2D data."""
        result = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), aggregation='average')
        assert result.grid_shape == (5, 5)
        assert result.distribution.detection_prob.shape == (5, 5)
        assert result.aggregation == 'average'

    def test_create_3d_result(self):
        """Test creating a result object with 3D data for per-round analysis."""
        result = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(10, 5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), rounds=np.arange(10), aggregation='per_round')
        assert result.grid_shape == (5, 5)
        assert result.distribution.detection_prob.shape == (10, 5, 5)
        assert result.aggregation == 'per_round'
        assert result.rounds is not None
        assert len(result.rounds) == 10

    def test_invalid_grid_shape(self):
        """Test that invalid grid_shape raises ValueError."""
        with pytest.raises(ValueError, match='grid_shape must be a tuple'):
            DetectionProbabilityPatchResult(grid_shape=(5,), distribution=DetectorDistribution(detection_prob=np.random.rand(5), detector_coords=np.array([[0, 0]])))

    def test_invalid_prob_dimensions(self):
        with pytest.raises(ValueError, match="detection_prob must be 2D or 3D"):
            DetectorDistribution(detection_prob=np.random.rand(5), detector_coords=np.array([[0, 0]]))

    def _skip_test_invalid_prob_dimensions(self):
        pass
        """Test that invalid detection_prob dimensions raises ValueError."""
        with pytest.raises(ValueError, match='detection_prob must be 2D or 3D'):
            DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5), detector_coords=np.array([[0, 0]])))

    def test_invalid_detector_coords(self):
        with pytest.raises(ValueError, match="detector_coords must be 2D array"):
            DetectorDistribution(detection_prob=np.random.rand(5, 5), detector_coords=np.array([0, 1, 2, 3]))

    def _skip_test_invalid_detector_coords(self):
        pass
        """Test that invalid detector_coords raises ValueError."""
        with pytest.raises(ValueError, match='detector_coords must be 2D array'):
            DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5), detector_coords=np.array([0, 1, 2, 3])))

    def test_invalid_aggregation(self):
        """Test that invalid aggregation mode raises ValueError."""
        with pytest.raises(ValueError, match='aggregation must be one of'):
            DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5), detector_coords=np.array([[0, 0]])), aggregation='invalid_mode')

    def test_empty_detector_coords_valid(self):
        """Test that empty detector_coords array (0, 2) is accepted."""
        result = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.empty((0, 2))))
        assert result.distribution.detector_coords.shape == (0, 2)

    def test_physics_realistic_values(self):
        """Test with physics-realistic detection probabilities."""
        rng = np.random.default_rng(seed=42)
        detection_prob = rng.uniform(low=0.001, high=0.015, size=(5, 5))
        result = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=detection_prob, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), aggregation='average')
        assert np.all(result.distribution.detection_prob >= 0)
        assert np.all(result.distribution.detection_prob <= 1)
        assert result.aggregation == 'average'

class TestAggregateDetectionProbability:
    """Test the aggregate_detection_probability function."""

    def test_average_aggregation(self):
        """Test average aggregation over rounds."""
        data_3d = np.random.rand(10, 5, 5)
        result = aggregate_detection_probability(data_3d, 'average')
        assert result.shape == (5, 5)
        expected = np.mean(data_3d, axis=0)
        assert np.allclose(result, expected)

    def test_median_aggregation(self):
        """Test median aggregation over rounds."""
        data_3d = np.random.rand(10, 5, 5)
        result = aggregate_detection_probability(data_3d, 'median')
        assert result.shape == (5, 5)
        expected = np.median(data_3d, axis=0)
        assert np.allclose(result, expected)

    def test_variance_aggregation(self):
        """Test variance aggregation over rounds."""
        data_3d = np.random.rand(10, 5, 5)
        result = aggregate_detection_probability(data_3d, 'variance')
        assert result.shape == (5, 5)
        expected = np.var(data_3d, axis=0)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize('method,expected_fn', [('average', lambda x: np.mean(x, axis=0)), ('median', lambda x: np.median(x, axis=0)), ('variance', lambda x: np.var(x, axis=0))])
    def test_all_methods_correct(self, method, expected_fn):
        """Parametrized test: all aggregation methods return correct values."""
        rng = np.random.default_rng(seed=0)
        data_3d = rng.random((10, 5, 5))
        result = aggregate_detection_probability(data_3d, method)
        assert result.shape == (5, 5)
        assert np.allclose(result, expected_fn(data_3d))

    def test_invalid_input_dimensions(self):
        """Test that 2D input raises ValueError."""
        data_2d = np.random.rand(5, 5)
        with pytest.raises(ValueError, match='must be 3D array'):
            aggregate_detection_probability(data_2d, 'average')

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        data_3d = np.random.rand(10, 5, 5)
        with pytest.raises(ValueError, match='Unknown aggregation method'):
            aggregate_detection_probability(data_3d, 'invalid')

class TestCreateInsetAxes:
    """Test the create_inset_axes function."""

    def test_create_inset_default_bounds(self):
        """Test creating inset with default bounds."""
        fig, main_ax = plt.subplots()
        inset_ax = create_inset_axes(fig, main_ax)
        assert inset_ax is not None
        assert inset_ax.figure is fig
        plt.close(fig)

    def test_create_inset_custom_bounds(self):
        """Test creating inset with custom bounds."""
        fig, main_ax = plt.subplots()
        custom_bounds = (0.1, 0.1, 0.4, 0.4)
        inset_ax = create_inset_axes(fig, main_ax, bounds=custom_bounds)
        assert inset_ax is not None
        plt.close(fig)

class TestPlotDetectionProbabilityPatch:
    """Test the plot_detection_probability_patch function."""

    def test_basic_plot_creation(self):
        """Test basic plot creation with 2D data."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), aggregation='average')
        fig, ax = plot_detection_probability_patch(data)
        assert fig is not None
        assert ax is not None
        assert len(ax.images) > 0
        plt.close(fig)

    def test_with_existing_axes(self):
        """Test plot with provided fig and ax."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        fig, ax = plt.subplots()
        fig_result, ax_result = plot_detection_probability_patch(data, fig=fig, ax=ax)
        assert fig_result is fig
        assert ax_result is ax
        plt.close(fig)

    def test_inconsistent_fig_ax_raises(self):
        """Test that providing only fig or only ax raises ValueError."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match='both None or both set'):
            plot_detection_probability_patch(data, fig=fig)
        with pytest.raises(ValueError, match='both None or both set'):
            plot_detection_probability_patch(data, ax=ax)
        plt.close(fig)

    def test_no_detectors_overlay(self):
        """Test plot without detector overlay."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([]).reshape(0, 2)))
        fig, ax = plot_detection_probability_patch(data, show_detectors=False)
        assert fig is not None
        plt.close(fig)

    def test_no_colorbar(self):
        """Test plot without colorbar."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        fig, ax = plot_detection_probability_patch(data, colorbar=False)
        assert fig is not None
        plt.close(fig)

    def test_no_grid(self):
        """Test plot without grid lines."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        fig, ax = plot_detection_probability_patch(data, show_grid=False)
        assert fig is not None
        plt.close(fig)

    def test_custom_colormap(self):
        """Test plot with custom colormap."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        fig, ax = plot_detection_probability_patch(data, cmap='plasma')
        assert fig is not None
        plt.close(fig)

    def test_per_round_plotting(self):
        """Test per-round plotting with 3D data."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(10, 5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), rounds=np.arange(10), aggregation='per_round')
        fig, ax = plot_detection_probability_patch(data, round_index=5)
        assert fig is not None
        plt.close(fig)

    def test_per_round_invalid_index(self):
        """Test that invalid round_index raises ValueError."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(10, 5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), rounds=np.arange(10), aggregation='per_round')
        with pytest.raises(ValueError, match='round_index.*out of range'):
            plot_detection_probability_patch(data, round_index=15)
        with pytest.raises(ValueError, match='round_index.*out of range'):
            plot_detection_probability_patch(data, round_index=-1)

    def test_per_round_with_2d_data_raises(self):
        """Test that round_index with 2D data raises ValueError."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), aggregation='average')
        with pytest.raises(ValueError, match='detection_prob is 2D'):
            plot_detection_probability_patch(data, round_index=0)

    def test_inset_plot(self):
        """Test creating an inset plot."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        fig, ax = plot_detection_probability_patch(data, inset=True)
        assert fig is not None
        plt.close(fig)

    def test_inset_with_custom_bounds(self):
        """Test inset plot with custom bounds."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        custom_bounds = (0.1, 0.1, 0.3, 0.3)
        fig, ax = plot_detection_probability_patch(data, inset=True, inset_bounds=custom_bounds)
        assert fig is not None
        plt.close(fig)

    def test_different_code_types(self):
        """Test plotting with different code types."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        for code_type in ['rotated_surface', 'surface', 'color']:
            fig, ax = plot_detection_probability_patch(data, code_type=code_type)
            assert fig is not None
            plt.close(fig)

    def test_different_aggregation_modes(self):
        """Test plotting with different aggregation modes."""
        for agg_mode in ['average', 'median', 'variance']:
            data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])), aggregation=agg_mode)
            fig, ax = plot_detection_probability_patch(data)
            assert fig is not None
            plt.close(fig)

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch between prob and grid raises ValueError."""
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(7, 7) * 0.1, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        with pytest.raises(ValueError, match="doesn't match grid_shape"):
            plot_detection_probability_patch(data)

class TestIntegration:
    """Integration tests simulating real-world usage."""

    def test_full_workflow_2d(self):
        """Test complete workflow with 2D aggregated data."""
        n_rounds = 10
        grid_size = 7
        detection_prob_3d = np.random.rand(n_rounds, grid_size, grid_size) * 0.15
        detection_prob_avg = aggregate_detection_probability(detection_prob_3d, 'average')
        detector_coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
        data = DetectionProbabilityPatchResult(grid_shape=(grid_size, grid_size), distribution=DetectorDistribution(detection_prob=detection_prob_avg, detector_coords=detector_coords), aggregation='average')
        fig, ax = plot_detection_probability_patch(data)
        assert fig is not None
        assert ax is not None
        assert len(ax.images) > 0
        plt.close(fig)

    def test_full_workflow_per_round(self):
        """Test complete workflow with per-round analysis."""
        n_rounds = 10
        grid_size = 7
        detection_prob_3d = np.random.rand(n_rounds, grid_size, grid_size) * 0.15
        detector_coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
        data = DetectionProbabilityPatchResult(grid_shape=(grid_size, grid_size), distribution=DetectorDistribution(detection_prob=detection_prob_3d, detector_coords=detector_coords), rounds=np.arange(n_rounds), aggregation='per_round')
        fig, ax = plot_detection_probability_patch(data, round_index=5)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_inset_in_larger_figure(self):
        """Test embedding detection probability plot in larger figure.

        When inset=True with an existing fig/ax, the function should:
        1. Create a NEW inset Axes added to fig (not the same as main_ax).
        2. Return (fig, inset_ax) where inset_ax is the new inset.
        3. main_ax remains unchanged with its original content.
        """
        fig, main_ax = plt.subplots(figsize=(12, 8))
        main_ax.plot([1, 2, 3], [1, 4, 9])
        main_ax.set_title('Main Analysis')
        initial_axes_count = len(fig.axes)
        data = DetectionProbabilityPatchResult(grid_shape=(5, 5), distribution=DetectorDistribution(detection_prob=np.random.rand(5, 5) * 0.01, detector_coords=np.array([[i, j] for i in range(5) for j in range(5)])))
        fig, inset_ax = plot_detection_probability_patch(data, fig=fig, ax=main_ax, inset=True, inset_bounds=(0.65, 0.65, 0.3, 0.3))
        assert inset_ax is not main_ax, 'plot_detection_probability_patch with inset=True must return a new inset Axes, not the original main_ax.'
        assert len(fig.axes) == initial_axes_count + 1
        assert inset_ax.figure is fig
        assert main_ax is not None
        assert main_ax.get_title() == 'Main Analysis'
        plt.close(fig)
if __name__ == '__main__':
    pytest.main([__file__, '-v'])