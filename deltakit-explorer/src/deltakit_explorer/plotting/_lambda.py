import matplotlib.pyplot as plt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import interpolate_lambda


def plot_lambda(
    lambda_data: LambdaData,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Interpolate and plot Λ-fitted data.

    This function interpolates and plots both the logical error-probability per round that has been used
    to compute Λ, the associated error-rates if provided, and the resulting fit, showing
    how close the fit is from actual data.

    Args:
        lambda_data: Result of a fit containing Λ, Λ₀, their standard deviations, and the original data.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.
        fig: A matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: A matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:
        from deltakit_explorer.analysis import calculate_lambda_and_lambda_std

        lambda_data = calculate_lambda_and_lambda_std(
            distances=[5, 7, 9],
            leppr=[0.15, 0.1, 0.05],
            leppr_stddev=[0.01, 0.008, 0.005],
        )
        fig, ax = plot_lambda(
            lambda_data=lambda_data,
        )
        ax.set_yscale("log")
        plt.show()
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # These should be already checked by the above code, but type checkers are not able
    # to infer that information, so including the asserts explicitly for type checkers
    # to understand.
    assert ax is not None
    assert fig is not None

    # Plot the logical error probabilities per round
    ax.errorbar(
        lambda_data.distances,
        lambda_data.leppr,
        yerr=lambda_data.leppr_std * num_sigmas,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[1],
        label=f"Logical error probabilities per round (±{num_sigmas}σ)",  # noqa: RUF001
    )

    lambda_result = interpolate_lambda(
        lambda_data, num_sigmas=num_sigmas, num_points=num_points
    )

    plot(lambda_result, fig=fig, ax=ax)
    return fig, ax
