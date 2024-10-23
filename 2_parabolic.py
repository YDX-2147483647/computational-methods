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
def __(mo, np):
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
    return (ref,)


@app.cell
def __(np):
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
    return (setup_conditions,)


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
def __(dt, dx, multi_diag, np, x):
    # to_next_u[#next_x, #current_x]
    to_next_u_ex = (
        np.eye(x.size) + dt * multi_diag([1, -2, 1], size=x.size) / dx**2
    )[1:-1, :]
    to_next_u_ex[:5, :5]
    return (to_next_u_ex,)


@app.cell
def __(setup_conditions, t, to_next_u_ex, x):
    # u[#x, #t]
    u_ex = setup_conditions(t, x)

    for _n, _t in enumerate(t):
        if _n == 0:
            continue

        u_ex[1:-1, _n] = to_next_u_ex @ u_ex[:, _n - 1]
    return (u_ex,)


@app.cell
def __(plot_surface, t, u_ex, x):
    plot_surface(t, x, u_ex, title="近似解")
    return


@app.cell
def __(plot_surface, ref, t, u_ex, x):
    plot_surface(t, x, u_ex - ref(t, x), title="误差")
    return


@app.cell
def __(np, ref, t, u_ex, x):
    np.abs(u_ex - ref(t, x)).max()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 1 最简隐格式（`im`）""")
    return


@app.cell
def __(dt, dx, multi_diag, np, x):
    # to_previous_u[#previous_x, #current_x] (without boundary)
    to_previous_u_im = (
        np.eye(x.size - 2) - dt * multi_diag([1, -2, 1], size=x.size - 2) / dx**2
    )
    to_previous_u_im
    return (to_previous_u_im,)


@app.cell
def __(
    dt,
    dx,
    linalg,
    multi_diag,
    np,
    setup_conditions,
    t,
    to_previous_u_im,
    x,
):
    # u[#x, #t]
    u_im = setup_conditions(t, x)

    _inv = linalg.inv(to_previous_u_im)

    _rhs = np.zeros(x.size - 2)

    for _n, _t in enumerate(t):
        if _n == 0:
            continue

        # RHS is derived from the “check” equation below
        _rhs[:] = u_im[1:-1, _n - 1]
        _rhs[0] += dt * u_im[0, _n] / dx**2
        _rhs[-1] += dt * u_im[-1, _n] / dx**2

        u_im[1:-1, _n] = _inv @ _rhs

    # Check

    # to_previous_u[#previous_x, #current_x] (only with current boundary)
    _to_previous_u = (
        np.eye(x.size) - dt * multi_diag([1, -2, 1], size=x.size) / dx**2
    )[1:-1, :]
    # to_previous_u (only with current boundary) @ current_x (with boundary) = previous_x (without boundary)
    assert np.abs(_to_previous_u @ u_im[:, _n] - u_im[1:-1, _n - 1]).max() < 1e-6
    return (u_im,)


@app.cell
def __(plot_surface, t, u_im, x):
    plot_surface(t, x, u_im, title="近似解")
    return


@app.cell
def __(plot_surface, ref, t, u_im, x):
    plot_surface(t, x, u_im - ref(t, x), title="误差")
    return


@app.cell
def __(np, ref, t, u_im, x):
    np.abs(u_im - ref(t, x)).max()
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
