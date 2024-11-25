import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# §5 双曲方程的差分方法""")
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
    from numpy import linalg, newaxis

    np.set_printoptions(precision=3, suppress=True)
    return linalg, newaxis, np


@app.cell
def __():
    from matplotlib import pyplot as plt
    return (plt,)


@app.cell
def __():
    from typing import override
    return (override,)


@app.cell
def __():
    from hyperbolic_pde import Solver, benchmark, ref
    return Solver, benchmark, ref


@app.cell
def __():
    from util import multi_diag, plot_surface, typst
    return multi_diag, plot_surface, typst


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 问题""")
    return


@app.cell
def __():
    a = 5
    return (a,)


@app.cell(hide_code=True)
def __(a, typst):
    typst(rf"""
    #import "@preview/physica:0.9.3": pdv, eval

    $
    pdv(u,t) + {a} pdv(u,x) = 0, x in RR. \
    eval(u)_(t=0) = cases(
      2 &"if" x in (-1,0),
      1 &"if" x = 0,
      -2 &"if" x in (0,1),
      0 &"otherwise"
    ).
    $
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    dx = mo.ui.slider(
        0.06, 0.5, 0.04, label=r"$\mathrm{d} x$", show_value=True, debounce=True
    )
    dx
    return (dx,)


@app.cell
def __():
    r = 1 / 6
    return (r,)


@app.cell(hide_code=True)
def __(dx, mo, r):
    dt = r * dx.value
    mo.md(rf"$\mathrm{{d}}t = {dt}$.")
    return (dt,)


@app.cell
def __(dx, np):
    x = np.arange(-5, 5 + dx.value, dx.value)
    return (x,)


@app.cell
def __(dt, np):
    t = np.arange(0, 1.0, dt)
    return (t,)


@app.cell
def __(plot_surface, ref, t, x):
    plot_surface(t, x, ref(t, x), title="真解")
    return


@app.cell
def __(np, t, x):
    benchmark_kwargs = dict(
        x_max=x[-1],
        x_min=x[0],
        t_max=x[-1],
        t_min=t[0],
        dt_dx_list=[(_dx / 6, _dx) for _dx in 2.0 ** np.arange(-6, -2, 1)],
    )
    return (benchmark_kwargs,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 1 迎风""")
    return


@app.cell
def __(Solver, a, multi_diag, np, override):
    class SolverWind(Solver):
        @override
        def post_init(self) -> None:
            # to_next_u[#next_x, #current_x]
            self.to_next_u = (
                np.eye(self.x.size)
                + a * self.dt * multi_diag([1, -1, 0], size=self.x.size) / self.dx
            )

        @override
        def step(self, t) -> None:
            self.u[:, t] = self.to_next_u @ self.u[:, t - 1]
    return (SolverWind,)


@app.cell
def __(SolverWind, t, x):
    solver_wind = SolverWind(t=t, x=x)
    solver_wind.solve()
    solver_wind.to_next_u
    return (solver_wind,)


@app.cell
def __(plot_surface, solver_wind, t, x):
    plot_surface(t, x, solver_wind.u, title="近似解")
    return


@app.cell
def __(plot_surface, solver_wind, t, x):
    plot_surface(t, x, solver_wind.error(), title="误差")
    return


@app.cell
def __(solver_wind):
    solver_wind.max_error()
    return


@app.cell
def __(SolverWind, benchmark, benchmark_kwargs):
    benchmark(SolverWind, **benchmark_kwargs)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2 蛙跳

        需要虚拟网格。
        """
    )
    return


if __name__ == "__main__":
    app.run()
