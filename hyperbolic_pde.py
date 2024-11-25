from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import marimo as mo
import numpy as np
from matplotlib.pyplot import subplots
from numpy import newaxis
from pandas import DataFrame
from seaborn import lineplot

from parabolic_pde import _Solvable

if TYPE_CHECKING:
    from typing import Collection, Final

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def setup_conditions(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """根据初始条件、边界条件准备预备解

    Params:
        t[#t]
        x[#x]
    Returns:
        u[#x, #t]
    """
    assert t.ndim == 1
    assert x.ndim == 1

    u = np.zeros((x.size, t.size))
    u[:, 0] = (-1 < x) * (x < 0) * 2 - (0 < x) * (x < 1) * 2

    return u


@mo.cache
def ref(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """真解

    Params:
        t[#t]
        x[#x]
    Returns:
        u[#x, #t]
    """
    assert t.ndim == 1
    assert x.ndim == 1
    x_initial = x[:, newaxis] - 5 * t[newaxis, :]
    return (-1 < x_initial) * (x_initial < 0) * 2 - (0 < x_initial) * (
        x_initial < 1
    ) * 2


class _PerformanceMixin(_Solvable):
    def error(self) -> np.ndarray:
        return self.u - ref(self.t, self.x)

    def max_error(self) -> float:
        return np.abs(self.error()).max()


class Solver(_PerformanceMixin, ABC):
    """PDE solver

    init (and post_init) → solve
    """

    dx: Final[float]
    dt: Final[float]

    # x[#x]
    x: Final[np.ndarray]

    # t[#t]
    t: Final[np.ndarray]

    # u[#x, #t]
    u: np.ndarray

    def __init__(self, *, t: np.ndarray, x: np.ndarray) -> None:
        assert x.ndim == 1
        assert t.ndim == 1

        self.dt = np.diff(t).mean()
        self.dx = np.diff(x).mean()
        self.t = t
        self.x = x

        self.u = setup_conditions(t, x)

        self.post_init()

    def post_init(self) -> None:
        """Prepare after `__init__`"""
        pass

    @abstractmethod
    def step(self, t: int) -> None:
        """Update u[1:-1, t]"""
        pass

    def solve(self) -> None:
        """Solve u"""
        for t in range(self.t.size):
            if t == 0:
                continue
            self.step(t)

    def validate(self, t: int) -> None:
        """Validate the PDE at `t`

        An optional abstract method.

        Raise an `AssertionError` if invalid.
        """
        pass


def benchmark(
    solver_cls: type[Solver],
    *,
    t_min: float,
    t_max: float,
    x_min: float,
    x_max: float,
    dt_dx_list: Collection[tuple[float, float]],
) -> DataFrame:
    """Benchmark

    Returns:
        列为dx、最大误差、时长
    """
    assert issubclass(solver_cls, Solver)

    # (dt, dx, max_error)[]
    stat: deque[tuple[float, float, float]] = deque()

    for dt, dx in mo.status.progress_bar(dt_dx_list):  # type: ignore
        dt: float
        dx: float
        x = np.arange(x_min, x_max + dx, dx)
        t = np.arange(t_min, t_max + dt, dt)

        solver = solver_cls(x=x, t=t)
        solver.solve()
        stat.append((dt, dx, solver.max_error()))

    return DataFrame(
        [[dt, dx, error] for (dt, dx, error) in stat],
        columns=["dt", "dx", "最大误差"],
    )


def plot_benchmark(data: DataFrame) -> tuple[Figure, Axes]:
    """Plot the benchmark result

    Params:
        `df`: Output of `benchmark()`
    """
    fig, axs = subplots(nrows=2, layout="constrained")

    lineplot(ax=axs[0], data=data, x="dx", y="最大误差", markers=True)
    axs[0].set_xlabel(r"$\mathrm{d}x$")
    lineplot(ax=axs[1], data=data, x="dt", y="最大误差", markers=True)
    axs[1].set_xlabel(r"$\mathrm{d}t$")

    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
    return fig, axs
