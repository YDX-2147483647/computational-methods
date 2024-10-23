from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import marimo as mo
import numpy as np

if TYPE_CHECKING:
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

    init → post_init → solve
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
