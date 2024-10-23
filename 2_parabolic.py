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
    return cm, plt


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
    from parabolic_pde import ref, setup_conditions, Solver
    return Solver, ref, setup_conditions


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
    solver_ex.to_next_u
    return (solver_ex,)


@app.cell
def __(solver_ex):
    solver_ex.solve()
    return


@app.cell
def __(plot_surface, solver_ex, t, x):
    plot_surface(t, x, solver_ex.u, title="近似解")
    return


@app.cell
def __(plot_surface, ref, solver_ex, t, x):
    plot_surface(t, x, solver_ex.u - ref(t, x), title="误差")
    return


@app.cell
def __(np, ref, solver_ex, t, x):
    np.abs(solver_ex.u - ref(t, x)).max()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 1 最简隐格式（`im`）""")
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

            self.rhs = np.zeros(self.x.size - 2)

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
    solver_im.to_previous_u
    return (solver_im,)


@app.cell
def __(solver_im):
    solver_im.solve()

    # Validate the last `t`
    solver_im.validate(solver_im.t.size - 1)
    return


@app.cell
def __(plot_surface, solver_im, t, x):
    plot_surface(t, x, solver_im.u, title="近似解")
    return


@app.cell
def __(plot_surface, ref, solver_im, t, x):
    plot_surface(t, x, solver_im.u - ref(t, x), title="误差")
    return


@app.cell
def __(np, ref, solver_im, t, x):
    np.abs(solver_im.u - ref(t, x)).max()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 2 CN格式""")
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
