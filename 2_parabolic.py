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


@app.cell(hide_code=True)
def __(mo):
    from functools import cache
    from subprocess import CalledProcessError, run
    from sys import stderr


    @cache
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
        """Write typst"""
        return mo.Html(_typst_compile(typ).decode())
    return CalledProcessError, cache, run, stderr, typst


@app.cell
def __(multi_diag):
    multi_diag([-1, 2, 3], size=5)
    return


@app.cell(hide_code=True)
def __(np):
    from collections.abc import Collection


    def multi_diag(coefficients: Collection[int], /, size: int) -> np.array:
        """三/多对角阵"""
        assert len(coefficients) % 2 == 1
        half_diag = (len(coefficients) - 1) // 2
        assert half_diag < size

        return sum(
            np.diagflat(c * np.ones(size - abs(k)), k=k)
            for k, c in enumerate(coefficients, start=-half_diag)
        )
    return Collection, multi_diag


@app.cell(hide_code=True)
def __(cm, np, plt):
    def plot_surface(
        t: np.array, x: np.array, u: np.array, title: str | None = None, **kwargs
    ) -> tuple[plt.Figure, plt.Axes3D]:
        """个人习惯版`Axes3D.plot_surface`

        Params:
            t[#t]
            x[#x]
            u[#x, #t]
        """
        assert t.ndim == 1
        assert x.ndim == 1
        assert u.shape == (x.size, t.size)

        fig, ax = plt.subplots(
            layout="constrained", subplot_kw={"projection": "3d"}
        )
        ax.plot_surface(
            t[np.newaxis, :], x[:, np.newaxis], u, cmap=cm.coolwarm, **kwargs
        )
        ax.xaxis.set_inverted(True)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        ax.set_zlabel("$u$")

        if title is not None:
            ax.set_title(title)

        return fig, ax
    return (plot_surface,)


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


@app.cell(hide_code=True)
def __(mo, np):
    @mo.cache
    def ref(t: np.array, x: np.array) -> np.array:
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
    mo.md(r"""一次求一行。""")
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
def __(np, t, to_next_u_ex, x):
    # u[#x, #t]
    u_ex = np.zeros((x.size, t.size))
    u_ex[0, :] = np.exp(t)
    u_ex[-1, :] = np.exp(t - 1)

    for _n, _t in enumerate(t):
        if _n == 0:
            u_ex[:, 0] = np.exp(-x)
        else:
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
    mo.md(r"""## 1 最简隐格式""")
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
