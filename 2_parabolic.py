import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# §4 抛物方程的差分格式<br>95页练习题8 隐格式如何选步长并消除振荡""")
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import numpy as np
    from numpy import linalg

    np.set_printoptions(precision=3, suppress=True)
    return linalg, np


@app.cell
def __():
    from matplotlib import pyplot as plt, cm
    import seaborn as sns
    return cm, plt, sns


@app.cell
def __():
    from collections import deque
    return (deque,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 工具函数""")
    return


@app.cell
def __():
    from util import multi_diag, typst, plot_surface
    return multi_diag, plot_surface, typst


@app.cell
def __(multi_diag):
    multi_diag([-1, 2, 3], size=5)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 问题""")
    return


@app.cell(hide_code=True)
def __(typst):
    typst(r"""
    #import "@preview/physica:0.9.3": pdv, eval

    $
    pdv(u,t) = pdv(u,x,2), quad x in (0,1), t in RR^+.
    $

    $
    eval(u)_(t=0) &= e^(-x), & x &in (0,1). \
    eval(u)_(x=0) &= e^t, & t &in RR^+. \
    eval(u)_(x=1) &= e^(t-1), & t &in RR^+. \
    $
    """)
    return


@app.cell(hide_code=True)
def __(typst):
    typst(r"""
    网格 $h = 2^(-n), 2 <= n <= 10$。
    """)
    return


@app.cell
def __():
    x_min = 0
    x_max = 1
    t_min = 0
    t_max = 1.602176634
    return t_max, t_min, x_max, x_min


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""先用一组参数测试。（分开设置，这样更新时可只重算一部分。）""")
    return


@app.cell(hide_code=True)
def __(mo):
    lb_dx = mo.ui.slider(
        -10,
        -2,
        1,
        value=-2,
        label=r"$\log_2 \mathrm{d}x$",
        show_value=True,
        debounce=True,
    )
    lb_dx
    return (lb_dx,)


@app.cell(hide_code=True)
def __(lb_dx, mo):
    dx = 2**lb_dx.value
    mo.md(rf"$\mathrm{{d}}x = {dx}$.")
    return (dx,)


@app.cell(hide_code=True)
def __(mo):
    r = mo.ui.slider(0.1, 0.6, 0.1, label=r"$r$", show_value=True, debounce=True)
    r
    return (r,)


@app.cell(hide_code=True)
def __(dx, mo, r):
    dt = r.value * dx**2
    mo.md(rf"$\mathrm{{d}}t = {dt}$.")
    return (dt,)


@app.cell
def __(dx, np, x_max, x_min):
    x = np.arange(x_min, x_max + dx, dx)
    x[:5], x[-5:]
    return (x,)


@app.cell
def __(dt, np, t_max, t_min):
    t = np.arange(t_min, t_max + dt, dt)
    t[:5], t[-5:]
    return (t,)


@app.cell(hide_code=True)
def __(plot_surface, ref, t, x):
    plot_surface(t, x, ref(t, x), title="真解")
    return


@app.cell
def __():
    from parabolic_pde import ref, setup_conditions, Solver, benchmark
    return Solver, benchmark, ref, setup_conditions


@app.cell
def __(np, t_max, t_min, x_max, x_min):
    benchmark_kwargs = dict(
        x_max=x_max,
        x_min=x_min,
        t_max=t_max,
        t_min=t_min,
        dx_list=2.0 ** np.arange(-10, -2, 1),
        dt=0.01,
    )
    return (benchmark_kwargs,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 0 最简显格式（`ex`）

        本想实现最简隐格式，不小心做成了显格式……
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""其实都是一次求一行（一个 $t$）。""")
    return


@app.cell
def __(Solver, multi_diag, np):
    class SolverExplicit(Solver):
        def post_init(self) -> None:
            # to_next_u[#next_x, #current_x]
            self.to_next_u = (
                np.eye(self.x.size)
                + self.dt * multi_diag([1, -2, 1], size=self.x.size) / self.dx**2
            )[1:-1, :]

        def step(self, t) -> None:
            self.u[1:-1, t] = self.to_next_u @ self.u[:, t - 1]
    return (SolverExplicit,)


@app.cell
def __(SolverExplicit, t, x):
    solver_ex = SolverExplicit(t=t, x=x)
    solver_ex.solve()
    solver_ex.to_next_u
    return (solver_ex,)


@app.cell
def __(plot_surface, solver_ex, t, x):
    plot_surface(t, x, solver_ex.u, title="近似解")
    return


@app.cell
def __(plot_surface, solver_ex, t, x):
    plot_surface(t, x, solver_ex.error(), title="误差")
    return


@app.cell
def __(solver_ex):
    solver_ex.max_error()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 1 最简隐格式（`im`）""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 单次""")
    return


@app.cell
def __(Solver, linalg, multi_diag, np):
    class SolverImplicit(Solver):
        def post_init(self) -> None:
            # to_previous_u[#previous_x, #current_x] (without boundary)
            self.to_previous_u = (
                np.eye(self.x.size - 2)
                - self.dt
                * multi_diag([1, -2, 1], size=self.x.size - 2)
                / self.dx**2
            )
            self.to_previous_u_inv = linalg.inv(self.to_previous_u)

            self.rhs = np.empty(self.x.size - 2)

        def step(self, t) -> None:
            # RHS is derived from the equation in “validate”
            self.rhs[:] = self.u[1:-1, t - 1]
            self.rhs[0] += self.dt * self.u[0, t] / self.dx**2
            self.rhs[-1] += self.dt * self.u[-1, t] / self.dx**2

            self.u[1:-1, t] = self.to_previous_u_inv @ self.rhs

        def validate(self, t: int) -> None:
            # to_previous_u[#previous_x, #current_x] (only with current boundary)
            to_previous_u = (
                np.eye(self.x.size)
                - self.dt * multi_diag([1, -2, 1], size=self.x.size) / self.dx**2
            )[1:-1, :]
            # to_previous_u (only with current boundary) @ current_x (with boundary) = previous_x (without boundary)
            assert (
                np.abs(to_previous_u @ self.u[:, t] - self.u[1:-1, t - 1]).max()
                < 1e-6
            )
    return (SolverImplicit,)


@app.cell
def __(SolverImplicit, t, x):
    solver_im = SolverImplicit(t=t, x=x)
    solver_im.solve()

    # Validate the last `t`
    solver_im.validate(solver_im.t.size - 1)

    solver_im.to_previous_u
    return (solver_im,)


@app.cell
def __(plot_surface, solver_im, t, x):
    plot_surface(t, x, solver_im.u, title="近似解")
    return


@app.cell
def __(plot_surface, solver_im, t, x):
    plot_surface(t, x, solver_im.error(), title="误差")
    return


@app.cell
def __(solver_im):
    solver_im.max_error()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 统计性能""")
    return


@app.cell
def __(SolverImplicit, benchmark, benchmark_kwargs, mo):
    with mo.persistent_cache("stat_im"):
        stat_im = benchmark(SolverImplicit, **benchmark_kwargs)
    stat_im
    return (stat_im,)


@app.cell(hide_code=True)
def __(plt, sns, stat_im):
    _fig, _axs = plt.subplots(nrows=3, layout="constrained")

    sns.lineplot(ax=_axs[0], data=stat_im, x="dx", y="最大误差", markers=True)
    _axs[0].set_xlabel(r"$\mathrm{d}x$")
    sns.lineplot(ax=_axs[1], data=stat_im, x="dx", y="时长", markers=True)
    _axs[1].set_xlabel(r"$\mathrm{d}x$")
    _axs[2].set_ylabel("时长 / s")
    # 时长需要计算误差线，必须放到纵轴
    sns.lineplot(ax=_axs[2], data=stat_im, y="时长", x="最大误差", markers=True)
    _axs[2].set_ylabel("时长 / s")

    for _ax in _axs:
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.grid(True)
    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 2 CN格式（`cn`）""")
    return


@app.cell
def __(Solver, linalg, multi_diag, np):
    class SolverCrankNicolson(Solver):
        def post_init(self) -> None:
            # (∂²/∂x²)[[#x_current_without_boundary, #x_previous_with_boundary]
            self.dv_x_2_previous = (
                multi_diag([1, -2, 1], size=self.x.size)[1:-1, :] / self.dx**2
            )

            # a_current[#previous_x, #current_x] (without boundary)
            a_current = (
                -np.eye(self.x.size - 2) / self.dt
                + multi_diag([1, -2, 1], size=self.x.size - 2) / self.dx**2 / 2
            )
            self.a_current_inv = linalg.inv(a_current)

            self.rhs = np.empty(self.x.size - 2)

        def step(self, t) -> None:
            # A @ u_current + boundary terms + A' @ u_previous = 0

            self.rhs[:] = (
                self.u[1:-1, t - 1] / self.dt
                + self.dv_x_2_previous @ self.u[:, t - 1] / 2
            )

            self.rhs[0] += self.u[0, t] / self.dx**2 / 2
            self.rhs[-1] += self.u[-1, t] / self.dx**2 / 2

            self.u[1:-1, t] = self.a_current_inv @ -self.rhs

        def validate(self, t: int) -> None:
            # (∂²/∂x²)[#x_without_boundary, #x_with_boundary]
            dv_x_2 = multi_diag([1, -2, 1], size=self.x.size)[1:-1, :] / self.dx**2
            # (approximate ∂²u/∂x²)[#x_without_boundary]
            approx_dv_x_2 = dv_x_2 @ (self.u[:, t] + self.u[:, t - 1]) / 2
            # (approximate ∂u/∂t)[[#x_without_boundary]
            approx_dv_t = (self.u[1:-1, t] - self.u[1:-1, t - 1]) / self.dt
            assert np.abs(approx_dv_t - approx_dv_x_2).max() < 1e-6
    return (SolverCrankNicolson,)


@app.cell
def __(SolverCrankNicolson, t, x):
    solver_cn = SolverCrankNicolson(t=t, x=x)
    solver_cn.solve()

    # Validate the last `t`
    solver_cn.validate(solver_cn.t.size - 1)

    solver_cn.dv_x_2_previous
    return (solver_cn,)


@app.cell
def __(plot_surface, solver_cn, t, x):
    plot_surface(t, x, solver_cn.u, title="近似解")
    return


@app.cell
def __(plot_surface, solver_cn, t, x):
    plot_surface(t, x, solver_cn.error(), title="误差")
    return


@app.cell
def __(solver_cn):
    solver_cn.max_error()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 3 振荡现象""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4 消除振荡

        ### 法一：自适应时间步长
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 法二：加权平均""")
    return


if __name__ == "__main__":
    app.run()
