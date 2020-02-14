import pytest
from pathlib import Path
import time

import numpy as np
import matplotlib

from gpflux2.utils.live_plotter import live_plot
from gpflux2.exceptions import InvalidPlotFunctionError

# No GUI for testing
matplotlib.use("agg")


@live_plot
def default_plotter(x_data, y_data, fig=None, axes=None):
    axes[0].scatter(x_data, y_data)


@live_plot(fig_kwargs={"figsize": (10, 5)}, subplots_kwargs={"nrows": 1, "ncols": 2})
def custom_plotter(x_data, y_data, fig=None, axes=None):
    axes[0].scatter(x_data, y_data)
    axes[1].scatter(y_data, x_data)


@live_plot(do_animation=True, animation_dir="animation_testing")
def animation_live_plotter(x_data, y_data, fig=None, axes=None):
    axes[0].scatter(x_data, y_data)


@pytest.mark.parametrize("plotter_func", [default_plotter, custom_plotter])
def test_live_plotter(plotter_func):
    for _ in range(5):
        x_data = np.random.randn(50)
        y_data = np.random.randn(50)

        plotter_func(x_data, y_data)
        time.sleep(0.3)


def test_live_plotter_user_controlled_fig():
    @live_plot
    def plot_two(x_data, y_data, title, fig=None, axes=None):
        axes[0].scatter(x_data, y_data)
        axes[1].scatter(y_data, x_data)
        fig.suptitle(title, fontsize=16)

    fig = matplotlib.pyplot.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    axes = [ax1, ax2]
    for _ in range(5):
        x_data = np.random.randn(50)
        y_data = np.random.randn(50)

        plot_two(x_data, y_data, "A nice plot!", fig=fig, axes=axes)
        time.sleep(0.3)


def test_animation():
    # 1.. Gen video
    for _ in range(5):
        x_data = np.random.randn(50)
        y_data = np.random.randn(50)

        animation_live_plotter(x_data, y_data)
        time.sleep(0.3)

    animation_live_plotter.save()

    # 2. Test files created
    animation_path = Path("animation_testing")
    tmp_dir = animation_path / "tmp"
    assert animation_path.exists() and animation_path.is_dir()
    assert tmp_dir.exists() and tmp_dir.is_dir()

    files = list(animation_path.glob("*.mp4"))
    assert len(files) == 1

    # 3. Clean Up
    files[0].unlink()
    tmp_dir.rmdir()
    animation_path.rmdir()


def test_bad_wrapped_function():
    with pytest.raises(InvalidPlotFunctionError):

        @live_plot
        def invalid_plotting_function(*args):
            pass

    with pytest.raises(InvalidPlotFunctionError):

        @live_plot
        def invalid_plotting_function(*args, fig=None):
            pass

    with pytest.raises(InvalidPlotFunctionError):

        @live_plot
        def invalid_plotting_function(*args, axes=None):
            pass


if __name__ == "__main__":
    matplotlib.use("tkagg")  # GUI back on

    test_live_plotter(default_plotter)
    test_live_plotter(custom_plotter)
    test_animation()
    test_live_plotter_user_controlled_fig()
    test_bad_wrapped_function()
