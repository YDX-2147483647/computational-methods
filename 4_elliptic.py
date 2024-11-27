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
    dy = 0.05
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
def __(np, ref, ref_rhs, setup_boundary):
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

            self.u = setup_boundary(x, y)
            self.rhs = ref_rhs(x, y)

            self.post_init()

        def post_init(self) -> None:
            """Prepare after `__init__`"""
            pass

        def solve(self) -> None: ...

        def validate(self) -> None:
            dv_x_2 = (self.u[2:, :] + self.u[:-2, :] - 2 * self.u[1:-1, :])[
                :, 1:-1
            ] / self.dx**2
            dv_y_2 = (self.u[:, 2:] + self.u[:, :-2] - 2 * self.u[:, 1:-1])[
                1:-1, :
            ] / self.dy**2
            assert np.allclose(
                dv_x_2 + dv_y_2,
                self.rhs[1:-1, 1:-1],
                # 这里定得很松，这样其它方法的解也能通过验证
                rtol=1e-2,
            )

        def error(self) -> np.ndarray:
            return self.u - ref(self.x, self.y)

        def max_error(self) -> float:
            return abs(self.error()).max()
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


@app.cell
def __(Solver, linalg, np):
    class Solver_5(Solver):
        def post_init(self) -> None:
            # x,y without boundaries ← x,y with boundaries
            t = np.zeros((self.x.size, self.y.size, self.x.size, self.y.size))
            for i_x in range(1, self.x.size - 1):
                all_y = np.arange(1, self.y.size - 1)
                t[i_x, all_y, i_x - 1, all_y] = 1 / self.dx**2
                t[i_x, all_y, i_x, all_y] = -2 / self.dx**2
                t[i_x, all_y, i_x + 1, all_y] = 1 / self.dx**2
            for i_y in range(1, self.y.size - 1):
                all_x = np.arange(1, self.x.size - 1)
                t[all_x, i_y, all_x, i_y - 1] += 1 / self.dy**2
                t[all_x, i_y, all_x, i_y] += -2 / self.dy**2
                t[all_x, i_y, all_x, i_y + 1] += 1 / self.dy**2

            # x,y without boundaries ← x,y with boundaries
            self.laplacian = t[1:-1, 1:-1, ...]

            # boundary terms affecting x,y without boundaries
            self.boundary = np.einsum(
                "xyuv,uv->xy",
                # Select x_min and x_max, including the y boundaries
                self.laplacian[..., :: self.x.size - 1, :],
                self.u[:: self.x.size - 1, :],
            ) + np.einsum(
                "xyuv,uv->xy",
                # Select y_min and y_max, excluding the x boundaries
                self.laplacian[..., 1:-1, :: self.y.size - 1],
                self.u[1:-1, :: self.y.size - 1],
            )

        def solve(self) -> None:
            # All to RHS + flatten
            self.u[1:-1, 1:-1].flat = linalg.solve(
                self.laplacian[..., 1:-1, 1:-1].reshape(
                    (-1, (self.x.size - 2) * (self.y.size - 2))
                ),
                (self.rhs[1:-1, 1:-1] - self.boundary).flat,
            )

        def validate(self) -> None:
            # Original
            assert np.allclose(
                np.einsum("xyuv,uv->xy", self.laplacian, self.u),
                self.rhs[1:-1, 1:-1],
            )

            # Flatten
            assert np.allclose(
                self.laplacian.reshape((-1, self.x.size * self.y.size))
                @ self.u.flat,
                self.rhs[1:-1, 1:-1].flat,
            )

            # Apart
            assert np.allclose(
                np.einsum(
                    "xyuv,uv->xy",
                    self.laplacian[..., 1:-1, 1:-1],
                    self.u[1:-1, 1:-1],
                )
                + self.boundary,
                self.rhs[1:-1, 1:-1],
            )

            # All to RHS
            assert np.allclose(
                np.einsum(
                    "xyuv,uv->xy",
                    self.laplacian[..., 1:-1, 1:-1],
                    self.u[1:-1, 1:-1],
                ),
                self.rhs[1:-1, 1:-1] - self.boundary,
            )

            # All to RHS + flatten
            assert np.allclose(
                self.laplacian[..., 1:-1, 1:-1].reshape(
                    (-1, (self.x.size - 2) * (self.y.size - 2))
                )
                @ self.u[1:-1, 1:-1].flat,
                (self.rhs[1:-1, 1:-1] - self.boundary).flat,
            )
    return (Solver_5,)


@app.cell
def __(Solver_5, x, y):
    solver_5 = Solver_5(x=x, y=y)
    solver_5.solve()
    solver_5.validate()
    return (solver_5,)


@app.cell
def __(plot_surface, solver_5, x, y):
    plot_surface(x, y, solver_5.u, title="近似解")
    return


@app.cell
def __(plot_surface, solver_5, x, y):
    plot_surface(x, y, solver_5.error(), title="误差")
    return


@app.cell
def __(solver_5):
    solver_5.max_error()
    return


if __name__ == "__main__":
    app.run()
