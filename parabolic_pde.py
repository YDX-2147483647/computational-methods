from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

import marimo as mo
import numpy as np
from pandas import DataFrame

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Final


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
    return np.exp(-x[:, np.newaxis] + t[np.newaxis, :])


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
    u[:, 0] = np.exp(-x)
    u[0, :] = np.exp(t)
    u[-1, :] = np.exp(t - 1)

    return u


class Solver(ABC):
    """PDE solver

    init (and post_init) → solve
    """

    dx: float
    dt: float

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

    def error(self) -> np.ndarray:
        return self.u - ref(self.t, self.x)

    def max_error(self) -> float:
        return np.abs(self.error()).max()

    def timing(self, *, number: int = 10, repeat: int = 7) -> deque[float]:
        """Measure the time of `solve`

        Params:
            `number`: number of executions
            `repeat`: repeat count

        Returns:
            average duration of an execution of each repetition, in seconds.
        """

        durations: deque[float] = deque()

        for _ in range(repeat):
            start = perf_counter()
            for _ in range(number):
                self.solve()
            durations.append((perf_counter() - start) / number)

        return durations


def benchmark(
    solver_cls: type[Solver],
    *,
    t_min: float,
    t_max: float,
    x_min: float,
    x_max: float,
    dx_list: Collection[float],
    dt: float = 0.01,
) -> tuple[DataFrame, DataFrame]:
    """Benchmark

    Returns:
        `timing`: 列为dx、时长
        `error`: 列为dx、最大误差
    """
    # (dx, durations, max error)[]
    stat: deque[tuple[float, deque[float], float]] = deque()

    for dx in mo.status.progress_bar(dx_list):  # type: ignore
        dx: float
        x = np.arange(x_min, x_max + dx, dx)
        t = np.arange(t_min, t_max + dt, dt)

        solver = solver_cls(x=x, t=t)
        timing = solver.timing()
        stat.append((dx, timing, solver.max_error()))

    return (
        DataFrame(
            [[dx, duration] for (dx, timing, _) in stat for duration in timing],
            columns=["dx", "时长"],
        ),
        DataFrame(
            [[dx, error] for (dx, _, error) in stat],
            columns=["dx", "最大误差"],
        ),
    )
