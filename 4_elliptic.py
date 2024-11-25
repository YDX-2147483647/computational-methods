import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# §7 椭圆方程的差分格式""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 准备工作""")
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import numpy as np
    from numpy import linalg, newaxis, pi, sin

    np.set_printoptions(precision=3, suppress=True)
    return linalg, newaxis, np, pi, sin


@app.cell
def __():
    from matplotlib import pyplot as plt
    return (plt,)


@app.cell
def __():
    from util import typst
    return (typst,)


@app.cell
def __(Axes3D, Figure, np, plt):
    from matplotlib import cm


    def plot_surface(
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        title: str | None = None,
        invert_x_axis=True,
        **kwargs,
    ) -> tuple[Figure, Axes3D]:
        """个人习惯版`Axes3D.plot_surface`

        Params:
            x[#x]
            y[#y]
            u[#x, #y]
            title
            invert_x_axis: 是否反转x轴。有时反转可避免遮挡，更清晰
        """
        assert x.ndim == 1
        assert y.ndim == 1
        assert u.shape == (x.size, y.size)

        ax: Axes3D
        fig, ax = plt.subplots(
            layout="constrained", subplot_kw={"projection": "3d"}
        )  # type: ignore
        ax.plot_surface(
            x[:, np.newaxis], y[np.newaxis, :], u, cmap=cm.coolwarm, **kwargs
        )
        ax.xaxis.set_inverted(invert_x_axis)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$u$")

        if title is not None:
            ax.set_title(title)

        return fig, ax
    return cm, plot_surface


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 问题""")
    return


@app.cell(hide_code=True)
def __(typst):
    typst(r"""
    #import "@preview/physica:0.9.3": laplacian, eval

    $
    - laplacian u &= (pi^2 - 1) e^x sin(pi y),
      quad (x,y) in [1,2] times [0,1]. \
    eval(u)_(x=1) &= e sin(pi y). \
    eval(u)_(x=2) &= e^2 sin(pi y). \
    eval(u)_(y=0) &= eval(u)_(y=1) = 0.
    $
    """)
    return


@app.cell
def __():
    dx = 0.1
    dy = 0.1
    return dx, dy


@app.cell
def __(dx, dy, np):
    x = np.arange(1, 2 + dx, dx)
    y = np.arange(0, 1 + dy, dy)
    return x, y


@app.cell
def __(mo, np, pi, sin):
    @mo.cache
    def ref(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """真解

        Params:
            x[#x]
            y[#y]
        Returns:
            u[#x, #y]
        """
        assert x.ndim == 1
        assert y.ndim == 1
        return np.exp(x[:, np.newaxis]) * sin(pi * y[np.newaxis, :])
    return (ref,)


@app.cell
def __(plot_surface, ref, x, y):
    plot_surface(x, y, ref(x, y), title="真解")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 0 基础""")
    return


@app.cell
def __(mo, np, pi, ref):
    @mo.cache
    def ref_rhs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """右端项

        Params:
            x[#x]
            y[#y]
        Returns:
            f[#x, #y]
        """
        return -(pi**2 - 1) * ref(x, y)
    return (ref_rhs,)


@app.cell
def __(mo, np, pi, sin):
    @mo.cache
    def setup_boundary(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """根据边界条件准备预备解

        Params:
            x[#x]
            y[#y]
        Returns:
            u[#x, #y]
        """
        assert x.ndim == 1
        assert y.ndim == 1

        u = np.zeros((x.size, y.size))
        u[0, :] = np.exp(1) * sin(pi * y)
        u[-1, :] = np.exp(2) * sin(pi * y)

        return u
    return (setup_boundary,)


@app.cell
def __(np, ref_rhs, setup_conditions):
    class Solver:
        u: np.ndarray
        x: np.ndarray
        y: np.ndarray

        rhs: np.ndarray

        dx: float
        dy: float

        def __init__(self, *, x: np.ndarray, y: np.ndarray) -> None:
            assert x.ndim == 1
            assert y.ndim == 1

            self.dx = np.diff(x).mean()
            self.dy = np.diff(y).mean()
            self.x = x
            self.y = y

            self.u = setup_conditions(x, y)
            self.rhs = ref_rhs(x, y)

        def solve(self) -> None: ...

        def validate(self) -> None:
            dv_x_2 = (self.u[2:, :] + self.u[:-2, :] - 2 * self.u[1:-1, :])[
                :, 1:-1
            ] / self.dx**2
            dv_y_2 = (self.u[:, 2:] + self.u[:, :-2] - 2 * self.u[:, 1:-1])[
                1:-1, :
            ] / self.dy**2
            assert np.allclose(dv_x_2 + dv_y_2, self.rhs[1:-1, 1:-1], rtol=1e-2)
    return (Solver,)


@app.cell
def __(Solver, ref, x, y):
    solver = Solver(x=x, y=y)
    solver.u = ref(solver.x, solver.y)
    solver.validate()
    return (solver,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 1 五点法

        > [Sparse arrays currently must be two-dimensional.](https://docs.scipy.org/doc/scipy/reference/sparse.html)
        """
    )
    return


if __name__ == "__main__":
    app.run()
