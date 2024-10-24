from __future__ import annotations

from subprocess import CalledProcessError, run
from sys import stderr
from typing import TYPE_CHECKING

import marimo as mo
import numpy as np
from matplotlib import cm
from matplotlib.pyplot import subplots

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Literal

    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D


@mo.cache
def _typst_compile(
    typ: str,
    *,
    prelude="#set page(width: auto, height: auto, margin: 10pt)\n",
    format="svg",
) -> bytes:
    """Compile a Typst document

    https://github.com/marimo-team/marimo/discussions/2441
    """
    try:
        return run(
            ["typst", "compile", "-", "-", "--format", format],
            input=(prelude + typ).encode(),
            check=True,
            capture_output=True,
        ).stdout
    except CalledProcessError as err:
        stderr.write(err.stderr.decode())
        raise err


def typst(typ: str) -> mo.Html:
    """Write typst in marimo notebooks"""
    return mo.Html(_typst_compile(typ).decode())


def multi_diag(coefficients: Collection[int], /, size: int) -> np.ndarray | Literal[0]:
    """三/多对角阵"""
    assert len(coefficients) % 2 == 1
    half_diag = (len(coefficients) - 1) // 2
    assert half_diag < size

    return sum(
        np.diagflat(c * np.ones(size - abs(k)), k=k)
        for k, c in enumerate(coefficients, start=-half_diag)
    )


def plot_surface(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    title: str | None = None,
    invert_t_axis=True,
    **kwargs,
) -> tuple[Figure, Axes3D]:
    """个人习惯版`Axes3D.plot_surface`

    Params:
        t[#t]
        x[#x]
        u[#x, #t]
        title
        invert_t_axis: 是否反转t轴。有时反转可避免遮挡，更清晰
    """
    assert t.ndim == 1
    assert x.ndim == 1
    assert u.shape == (x.size, t.size)

    ax: Axes3D
    fig, ax = subplots(layout="constrained", subplot_kw={"projection": "3d"})  # type: ignore
    ax.plot_surface(t[np.newaxis, :], x[:, np.newaxis], u, cmap=cm.coolwarm, **kwargs)
    ax.xaxis.set_inverted(invert_t_axis)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_zlabel("$u$")

    if title is not None:
        ax.set_title(title)

    return fig, ax


if __name__ == "__main__":
    assert np.all(
        multi_diag([-1, 2, 3], size=5)
        == np.array(
            [
                [2, 3, 0, 0, 0],
                [-1, 2, 3, 0, 0],
                [0, -1, 2, 3, 0],
                [0, 0, -1, 2, 3],
                [0, 0, 0, -1, 2],
            ]
        )
    )
