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
def __(Axes, Collection, Figure, Solver, mo, np, plt):
    from collections import deque

    from pandas import DataFrame
    from seaborn import lineplot


    def benchmark(
        solver_cls: type[Solver],
        *,
        dx_dy_list: Collection[tuple[float, float]],
    ) -> DataFrame:
        """Benchmark

        Returns:
            列为dx、dy、最大误差
        """
        assert issubclass(solver_cls, Solver)

        # (dx, dy, max_error)[]
        stat: deque[tuple[float, float, float]] = deque()

        for dx, dy in mo.status.progress_bar(dx_dy_list):  # type: ignore
            dx: float
            dy: float
            x = np.arange(1, 2 + dx, dx)
            y = np.arange(0, 1 + dy, dy)

            solver = solver_cls(x=x, y=y)
            solver.solve()
            stat.append((dx, dy, solver.max_error()))

        return DataFrame(
            list(stat),
            columns=["dx", "dy", "最大误差"],
        )


    def plot_benchmark(data: DataFrame) -> tuple[Figure, Axes]:
        """Plot the benchmark result

        Params:
            `df`: Output of `benchmark()`
        """
        fig, axs = plt.subplots(nrows=2, layout="constrained")

        lineplot(ax=axs[0], data=data, x="dx", y="最大误差", markers=True)
        axs[0].set_xlabel(r"$\mathrm{d}x$")
        lineplot(ax=axs[1], data=data, x="dy", y="最大误差", markers=True)
        axs[1].set_xlabel(r"$\mathrm{d}y$")

        for ax in axs:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True)
        return fig, axs
    return DataFrame, benchmark, deque, lineplot, plot_benchmark


@app.cell
def __(np):
    benchmark_kwargs = dict(
        dx_dy_list=[(_dx, _dx) for _dx in 2.0 ** np.arange(-4, -1, 1)],
    )
    return (benchmark_kwargs,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 1 五点""")
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
def __(Solver_5, np):
    _x = np.arange(7)
    _from = Solver_5(x=_x, y=_x).laplacian[2, 2, :]
    print(_from)

    assert np.count_nonzero(_from) == 5
    assert np.isclose(_from.sum(), 0)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 单次""")
    return


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


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 统计""")
    return


@app.cell
def __(Solver_5, benchmark, benchmark_kwargs, plot_benchmark):
    _b = benchmark(Solver_5, **benchmark_kwargs)
    plot_benchmark(_b)[0]
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2 九点

        这回用稀疏矩阵试试。

        > [Sparse arrays currently must be two-dimensional.](https://docs.scipy.org/doc/scipy/reference/sparse.html)
        """
    )
    return


@app.cell
def __():
    from scipy.sparse import diags_array
    from scipy.sparse.linalg import spsolve
    return diags_array, spsolve


@app.cell(hide_code=True)
def __(typst):
    typst(r"""
    #import "@preview/physica:0.9.3": pdv, laplacian

    首先要处理一下 $h_x != h_y$ 的问题。

    #let xx = $cal(X)$
    #let yy = $cal(Y)$

    设 $xx := h_x pdv(,x), thick yy := h_y pdv(,y)$，要凑出
    $ laplacian = (xx / h_x)^2 + (yy / h_y)^2. $

    注意
    $
      e^xx - e^(-xx) &= 2 cosh xx &~ 2 (1 + xx^2/2). \
      e^yy - e^(-yy) &= 2 cosh yy &~ 2 (1 + yy^2/2). \

      e^(xx+yy) + e^(xx-yy) + e^(-xx+yy) + e^(-xx-yy)
      &= (e^xx + e^(-xx)) (e^yy + e^(-yy))
      = 4 cosh xx cosh yy
      &~ 4 (1 + xx^2 / 2 + yy^2 /2). \
    $

    等一下，真能凑出来吗？
    """)
    return


@app.cell
def __(Solver, diags_array, np, pi, spsolve):
    class Solver_9(Solver):
        def post_init(self) -> None:
            assert np.isclose(self.dx, self.dy)

            # without boundaries
            shape = (self.x.size - 2, self.y.size - 2)

            def merge_xy(x: int, y: int) -> int:
                # Convert (Δx, Δy) to Δxy,
                # ≈ ravel_multi_index of NumPy,
                # but support negative indices
                return x * shape[1] + y

            center = -10 / 3 / self.dx**2
            edges = 2 / 3 / self.dx**2
            corners = 1 / 6 / self.dx**2

            # xy without boundaries ← xy without boundaries
            self.laplacian = diags_array(
                [center] + [edges] * 4 + [corners] * 4,
                offsets=[
                    # Center
                    merge_xy(0, 0),
                    # Edges
                    merge_xy(1, 0),
                    merge_xy(-1, 0),
                    merge_xy(0, 1),
                    merge_xy(0, -1),
                    # Corners
                    merge_xy(1, 1),
                    merge_xy(1, -1),
                    merge_xy(-1, -1),
                    merge_xy(-1, 1),
                ],
                shape=2 * (np.prod(shape),),
                # To perform inversion, first convert to either CSC or CSR format.
                format="csc",
            )

            # boundary terms affecting x,y without boundaries
            self.boundary = np.zeros(shape)
            # Select x_min and x_max
            self.boundary[:: shape[0] - 1, :] += (
                edges * self.rhs[:: self.x.size - 1, 1:-1]
                + corners * self.rhs[:: self.x.size - 1, :-2]
                + corners * self.rhs[:: self.x.size - 1, 2:]
            )
            # Select y_min and y_max
            self.boundary[:, :: shape[1] - 1] += (
                edges * self.u[1:-1, :: self.y.size - 1]
                + corners * self.u[:-2, :: self.y.size - 1]
                + corners * self.u[2:, :: self.y.size - 1]
            )

            # RHS 要改成 f + 1/12 (h ∇)² f = (1 + 1/12 (h (π²-1))²) f
            self.rhs *= 1 + (self.dx * (pi**2 - 1)) ** 2 / 12

        def solve(self) -> None:
            self.u[1:-1, 1:-1].flat = spsolve(
                self.laplacian, (self.rhs[1:-1, 1:-1] - self.boundary).flat
            )

        def validate(self) -> None:
            # Apart
            assert np.allclose(
                np.einsum(
                    "xyuv,uv->xy",
                    self.laplacian.toarray().reshape(
                        self.x.size - 2,
                        self.y.size - 2,
                        self.x.size - 2,
                        self.y.size - 2,
                    ),
                    self.u[1:-1, 1:-1],
                )
                + self.boundary,
                self.rhs[1:-1, 1:-1],
            )
    return (Solver_9,)


@app.cell
def __(Solver_9, np):
    _x = np.arange(7)
    _from = (
        Solver_9(x=_x, y=_x)
        .laplacian.toarray()
        .reshape((_x.size - 2,) * 4)[1, 2, :]
    )
    print(_from)

    assert np.count_nonzero(_from) == 9
    assert np.isclose(_from.sum(), 0)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 单次""")
    return


@app.cell
def __(Solver_9, x, y):
    solver_9 = Solver_9(x=y - y[0] + x[0], y=y)
    solver_9.solve()
    solver_9.validate()
    return (solver_9,)


@app.cell
def __(plot_surface, solver_9):
    plot_surface(solver_9.x, solver_9.y, solver_9.u, title="近似解")
    return


@app.cell
def __(plot_surface, solver_9):
    plot_surface(solver_9.x, solver_9.y, solver_9.error(), title="误差")
    return


@app.cell
def __(solver_9):
    solver_9.max_error()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 统计""")
    return


@app.cell
def __(Solver_9, benchmark, benchmark_kwargs, plot_benchmark):
    _b = benchmark(Solver_9, **benchmark_kwargs)
    plot_benchmark(_b)[0]
    return


if __name__ == "__main__":
    app.run()
