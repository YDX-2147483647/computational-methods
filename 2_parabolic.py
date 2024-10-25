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
    from matplotlib import pyplot as plt
    return (plt,)


@app.cell
def __():
    from typing import override
    return (override,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 工具函数""")
    return


@app.cell
def __():
    from util import multi_diag, plot_surface, typst
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
    r = mo.ui.slider(0.1, 10.0, 0.2, label=r"$r$", show_value=True, debounce=True)
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
    from parabolic_pde import (
        AdaptiveSolver,
        Solver,
        benchmark,
        plot_benchmark,
        ref,
        setup_conditions,
    )
    return (
        AdaptiveSolver,
        Solver,
        benchmark,
        plot_benchmark,
        ref,
        setup_conditions,
    )


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
def __(Solver, multi_diag, np, override):
    class SolverExplicit(Solver):
        @override
        def post_init(self) -> None:
            # to_next_u[#next_x, #current_x]
            self.to_next_u = (
                np.eye(self.x.size)
                + self.dt * multi_diag([1, -2, 1], size=self.x.size) / self.dx**2
            )[1:-1, :]

        @override
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
def __(Solver, linalg, multi_diag, np, override):
    class SolverImplicit(Solver):
        @override
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

        @override
        def step(self, t) -> None:
            # RHS is derived from the equation in “validate”
            self.rhs[:] = self.u[1:-1, t - 1]
            self.rhs[0] += self.dt * self.u[0, t] / self.dx**2
            self.rhs[-1] += self.dt * self.u[-1, t] / self.dx**2

            self.u[1:-1, t] = self.to_previous_u_inv @ self.rhs

        @override
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
def __(plot_benchmark, stat_im):
    _fig, _axs = plot_benchmark(stat_im)
    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 2 CN格式（`cn`）""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 单次""")
    return


@app.cell
def __(Solver, linalg, multi_diag, np, override):
    class SolverCrankNicolson(Solver):
        @override
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

        @override
        def step(self, t) -> None:
            # A @ u_current + boundary terms + A' @ u_previous = 0

            self.rhs[:] = (
                self.u[1:-1, t - 1] / self.dt
                + self.dv_x_2_previous @ self.u[:, t - 1] / 2
            )

            self.rhs[0] += self.u[0, t] / self.dx**2 / 2
            self.rhs[-1] += self.u[-1, t] / self.dx**2 / 2

            self.u[1:-1, t] = self.a_current_inv @ -self.rhs

        @override
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
    mo.md(r"""### 统计性能""")
    return


@app.cell
def __(SolverCrankNicolson, benchmark, benchmark_kwargs, mo):
    with mo.persistent_cache("stat_cn"):
        stat_cn = benchmark(SolverCrankNicolson, **benchmark_kwargs)
    stat_cn
    return (stat_cn,)


@app.cell(hide_code=True)
def __(plot_benchmark, stat_cn):
    _fig, _axs = plot_benchmark(stat_cn)
    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 3 振荡现象""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 原问题""")
    return


@app.cell(hide_code=True)
def __(dx, lb_dx, mo):
    mo.md(rf"""
    {lb_dx}
    $\mathrm{{d}}x = {dx}$.
    """)
    return


@app.cell(hide_code=True)
def __(dt, mo, r):
    mo.md(rf"""
    {r}
    $\mathrm{{d}}t = {dt}$.
    """)
    return


@app.cell
def __(plot_surface, solver_cn, t, x):
    plot_surface(t, x, solver_cn.u, title="近似解")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""调到最粗 $\mathrm{d}x = 0.25, \mathrm{d}t = 0.625$ 也没振荡呢……""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 更改条件（`gate`）""")
    return


@app.cell(hide_code=True)
def __(typst):
    typst(r"""
    #import "@preview/physica:0.9.3": eval

    $
    eval(u)_(t=0) &= I_((1/3, 2/3)). \
    eval(u)_(x=0) &= 0, & t &in RR^+. \
    eval(u)_(x=1) &= 0, & t &in RR^+. \
    $
    """)
    return


@app.cell
def __(SolverCrankNicolson, dt_gate, np, t_max, t_min, x_max, x_min):
    solver_gate = SolverCrankNicolson(
        t=np.arange(t_min, t_max, dt_gate.value),
        x=np.linspace(x_min, x_max, 20),
    )

    # 另设初始条件
    solver_gate.u[:, 0] = (1 / 3 < solver_gate.x) & (solver_gate.x < 2 / 3)
    solver_gate.u[0, :] = 0
    solver_gate.u[-1, :] = 0

    solver_gate.solve()

    # Validate the last `t`
    solver_gate.validate(solver_gate.t.size - 1)
    return (solver_gate,)


@app.cell(hide_code=True)
def __(mo):
    dt_gate = mo.ui.dropdown(
        {str(v): v for v in [0.001, 0.002, 0.008, 0.016, 0.032]},
        "0.032",
        label=r"$\mathrm{d}t =$",
        allow_select_none=False,
    )
    dt_gate
    return (dt_gate,)


@app.cell(hide_code=True)
def __(plot_surface, solver_gate):
    plot_surface(
        solver_gate.t,
        solver_gate.x,
        solver_gate.u,
        title="近似解",
        invert_t_axis=False,
    )
    return


@app.cell(hide_code=True)
def __(plot_surface, solver_gate):
    _concerned = (0.1 < solver_gate.t) & (solver_gate.t < 0.3)
    plot_surface(
        solver_gate.t[_concerned],
        solver_gate.x,
        solver_gate.u[:, _concerned],
        title="近似解（局部放大）",
        invert_t_axis=False,
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        边界条件为零，$u$ 逐渐衰减。为避免初始条件遮挡，这里反转了 $t$ 轴。

        ——总之出现振荡了。
        """
    )
    return


@app.cell(hide_code=True)
def __(plt, solver_gate):
    _x_range = slice(4, 9)
    _t_range = solver_gate.t < 0.3

    _fig, _ax = plt.subplots()
    _ax.plot(
        solver_gate.t[_t_range],
        solver_gate.u[_x_range, _t_range].T,
        label=[f"$x = {_x:.2f}$" for _x in solver_gate.x[_x_range]],
    )
    _ax.set_xlabel("$t$")
    _ax.set_ylabel("$u$")
    _ax.grid(True)
    _ax.legend()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4 消除振荡

        ### 法一：自适应时间步长

        首先要检测振荡。振荡在时间、空间上都有，主要是检测时间上的。
        """
    )
    return


@app.cell(hide_code=True)
def __(dt_gate):
    dt_gate
    return


@app.cell(hide_code=True)
def __(np, plt, solver_gate):
    _t_range = solver_gate.t < 0.3

    _fig, _axs = plt.subplots(nrows=3, sharex=True, layout="constrained")

    _axs[0].pcolorfast(
        solver_gate.t[_t_range],
        solver_gate.x,
        np.diff(solver_gate.u[:, _t_range], axis=1),
    )
    _axs[0].set_title(r"$\Delta_t u$（增量）")

    _axs[1].pcolorfast(
        solver_gate.t[_t_range],
        solver_gate.x,
        np.sign(np.diff(solver_gate.u[:, _t_range], axis=1)).astype(np.int8),
    )
    _axs[1].set_title(r"$\operatorname{sgn} \Delta_t u$（单调性）")

    _axs[2].pcolorfast(
        solver_gate.t[_t_range],
        solver_gate.x,
        np.diff(
            np.sign(np.diff(solver_gate.u[:, _t_range], axis=1)).astype(np.int8),
            axis=1,
        )
        != 0,
    )
    _axs[2].set_title(
        r"$\Delta_t \operatorname{sgn} \Delta_t u \neq 0$（单调性在变化）"
    )

    for _ax in _axs:
        _ax.set_ylabel("$x$")
    _axs[-1].set_xlabel("$t$")

    _fig
    return


@app.cell
def __(is_trembling, solver_gate):
    is_trembling(solver_gate.u[:, :5].T)
    return


@app.cell
def __(Collection, Final, np):
    def is_trembling(u: Final[Collection[np.ndarray]]) -> bool:
        """检查u是否在振荡

        Params:
            u[#t][#x]

        若超过10%的x有超过50%的t发生单调性变化（Δₜ sgn Δₜ u ≠ 0），则判为振荡；否则判为不振荡
        """
        # changed[#t, #x]
        changed = np.diff(np.sign(np.diff(u, axis=0)).astype(np.int8), axis=0) != 0
        return (changed.mean(axis=0) > 0.5).mean() > 0.1
    return (is_trembling,)


@app.cell
def __(is_trembling, np):
    _t = np.arange(6)
    _x = np.arange(5)

    assert not is_trembling(np.ones((_t.size, _x.size)))
    assert is_trembling(np.einsum("t,x->tx", _t, _x) % 2)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""然后再根据是否存在振荡自适应调整 $\mathrm{d}t$。""")
    return


@app.cell
def __(AdaptiveSolver, is_trembling, linalg, multi_diag, np, override):
    class AdaptiveSolverCrankNicolson(AdaptiveSolver):
        @override
        def post_init(self) -> None:
            # (∂²/∂x²)[[#x_current_without_boundary, #x_previous_with_boundary]
            self.dv_x_2_previous = (
                multi_diag([1, -2, 1], size=self.x.size)[1:-1, :] / self.dx**2
            )

            self.refresh_for_dt_changes()

            self.rhs = np.empty(self.x.size - 2)

        def refresh_for_dt_changes(self) -> None:
            # a_current[#previous_x, #current_x] (without boundary)
            a_current = (
                -np.eye(self.x.size - 2) / self.dt
                + multi_diag([1, -2, 1], size=self.x.size - 2) / self.dx**2 / 2
            )
            self.a_current_inv = linalg.inv(a_current)

        def try_step(self) -> tuple[float, np.ndarray]:
            """Calculate the next t,u without checking trembling"""
            # A @ u_current + boundary terms + A' @ u_previous = 0

            # Next t
            t = self.t[-1] + self.dt

            # Prepare next u
            u = np.zeros(self.x.size)
            u[0] = self.u_boundary[0](t)
            u[-1] = self.u_boundary[1](t)

            self.rhs[:] = (
                self.u[-1][1:-1] / self.dt + self.dv_x_2_previous @ self.u[-1] / 2
            )

            self.rhs[0] += u[0] / self.dx**2 / 2
            self.rhs[-1] += u[-1] / self.dx**2 / 2

            u[1:-1] = self.a_current_inv @ -self.rhs

            return t, u

        @override
        def step(self) -> None:
            t, u = self.try_step()

            # 定期检查是否有振荡
            # 最初 len(self.t) == 1，故进入if则必已有足够数据
            if len(self.t) % 5 == 0:
                trembling = is_trembling([self.u[n] for n in range(-4, 0)] + [u])

                if not trembling:
                    # 若无，考虑以后增大dt
                    if len(self.t) % 10 == 0:
                        self.dt *= 2
                        self.refresh_for_dt_changes()
                else:
                    # 若有，回退，缩小dt，重算
                    while trembling:
                        # 回退
                        for _ in range(-4, 0):
                            self.t.pop()
                            self.u.pop()

                        self.dt /= 2
                        self.refresh_for_dt_changes()

                        # 重算
                        for _ in range(-4, 0):
                            t, u = self.try_step()
                            self.t.append(t)
                            self.u.append(u)
                        t, u = self.try_step()

                        # 再次检查
                        trembling = is_trembling(
                            [self.u[n] for n in range(-4, 0)] + [u]
                        )

            self.t.append(t)
            self.u.append(u)

        @override
        def validate(self, t: int) -> None:
            # (∂²/∂x²)[#x_without_boundary, #x_with_boundary]
            dv_x_2 = multi_diag([1, -2, 1], size=self.x.size)[1:-1, :] / self.dx**2
            # (approximate ∂²u/∂x²)[#x_without_boundary]
            approx_dv_x_2 = dv_x_2 @ (self.u[t] + self.u[t - 1]) / 2
            # (approximate ∂u/∂t)[[#x_without_boundary]
            approx_dv_t = (self.u[t][1:-1] - self.u[t - 1][1:-1]) / self.dt
            assert np.abs(approx_dv_t - approx_dv_x_2).max() < 1e-6
    return (AdaptiveSolverCrankNicolson,)


@app.cell
def __(
    AdaptiveSolverCrankNicolson,
    dt_gate,
    np,
    t_max,
    t_min,
    x_max,
    x_min,
):
    _x = np.linspace(x_min, x_max, 20)
    solver_gate_adaptive = AdaptiveSolverCrankNicolson(
        t=(t_min, t_max, dt_gate.value),
        x=_x,
        u_initial=(1 / 3 < _x) & (_x < 2 / 3),
        u_boundary=(lambda t: 0, lambda t: 0),
    )

    solver_gate_adaptive.solve()

    # Validate the last `t`
    solver_gate_adaptive.validate(-1)
    return (solver_gate_adaptive,)


@app.cell(hide_code=True)
def __(mo, solver_gate_adaptive):
    mo.md(rf"""
    - Initial $\mathrm{{d}}t = {solver_gate_adaptive.t[1] - solver_gate_adaptive.t[0]}$.
    - Final $\mathrm{{d}}t = {solver_gate_adaptive.dt}$.
    - Final $t = {solver_gate_adaptive.t[-1]}$.
    """)
    return


@app.cell(hide_code=True)
def __(np, plt, solver_gate_adaptive):
    _fig, _axs = plt.subplots(nrows=2)

    _axs[0].plot(solver_gate_adaptive.t)
    _axs[0].set_ylabel("$t$")

    _axs[1].semilogy(np.diff(solver_gate_adaptive.t))
    _axs[1].set_ylabel(r"$\mathrm{d}t$")

    for _ax in _axs:
        _ax.set_xlabel("#step")
        _ax.grid(True)

    _fig
    return


@app.cell(hide_code=True)
def __(dt_gate):
    dt_gate
    return


@app.cell(hide_code=True)
def __(plot_surface, solver_gate_adaptive):
    plot_surface(
        solver_gate_adaptive.t_array(),
        solver_gate_adaptive.x,
        solver_gate_adaptive.u_array(),
        title="近似解",
        invert_t_axis=False,
    )
    return


@app.cell(hide_code=True)
def __(plot_surface, solver_gate_adaptive):
    _t = solver_gate_adaptive.t_array()
    _concerned = (0.1 < _t) & (_t < 0.3)
    plot_surface(
        _t[_concerned],
        solver_gate_adaptive.x,
        solver_gate_adaptive.u_array()[:, _concerned],
        title="近似解（局部放大）",
        invert_t_axis=False,
    )
    return


@app.cell(hide_code=True)
def __(plt, solver_gate_adaptive):
    _t = solver_gate_adaptive.t_array()

    _x_range = slice(4, 9)
    _t_range = _t < 0.3

    _fig, _ax = plt.subplots()
    _ax.plot(
        _t[_t_range],
        solver_gate_adaptive.u_array()[_x_range, _t_range].T,
        label=[f"$x = {_x:.2f}$" for _x in solver_gate_adaptive.x[_x_range]],
    )
    _ax.set_xlabel("$t$")
    _ax.set_ylabel("$u$")
    _ax.grid(True)
    _ax.legend()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""不过不能看误差，因为真解不太好写出来。""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 法二：加权平均""")
    return


if __name__ == "__main__":
    app.run()
