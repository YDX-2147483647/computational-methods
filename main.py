import marimo

__generated_with = "0.8.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import numpy as np
    from numpy import sin, linalg

    np.set_printoptions(precision=3)
    return linalg, np, sin


@app.cell
def __():
    from matplotlib import pyplot as plt

    plt.rcParams["mathtext.fontset"] = "cm"
    return (plt,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        问题：

        $$
        \mathcal{L} y = y'' + y = -x,\quad x \in (0,1).
        $$

        $$
        y(0) = y(1) = 0.
        $$
        """
    )
    return


@app.cell
def __():
    x_min = 0
    x_max = 1
    return x_max, x_min


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 解析解

        $0 = y'' + y + x = (y+x)'' + (y+x)$，解是三角函数。带入边界条件，得 $y+x = \sin x / \sin 1$。
        """
    )
    return


@app.cell
def __(sin):
    def ref(x):
        return sin(x) / sin(1) - x
    return (ref,)


@app.cell
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


    multi_diag([1, 2, 3], size=5)
    return Collection, multi_diag


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 1 差分法（`fin`: finite difference）

        间距 $\mathrm{d}x = h = \frac{1}{10}$。
        """
    )
    return


@app.cell
def __(np, x_max, x_min):
    dx_fin = 1 / 10
    x_fin = np.arange(x_min, x_max + dx_fin, dx_fin)
    x_fin
    return dx_fin, x_fin


@app.cell
def __(dx_fin, multi_diag, np, x_fin):
    # dv: derivative, l: ℒ

    # 中间部分用差商
    # y'' ≈ (y(+1) - 2 y + y(-1)) / dx²
    _dv_y_2 = multi_diag([1, -2, 1], x_fin.size) / dx_fin**2

    # ℒy = y'' + y
    l_by_y_fin = _dv_y_2 + np.eye(x_fin.size)

    # 两端直接用边界条件
    l_by_y_fin[[0, -1], :] = 0
    l_by_y_fin[0, 0] = 1
    l_by_y_fin[-1, -1] = 1

    l_by_y_fin
    return (l_by_y_fin,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""拿已知 $\mathcal{L}y$ 的 $y$ 测试一下。""")
    return


@app.cell
def __(l_by_y_fin, x_fin):
    l_by_y_fin @ x_fin**2
    return


@app.cell
def __(dx_fin, np, x_fin):
    np.diff(np.diff(x_fin**2)) / dx_fin**2
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        并无问题……（最后发现是下面的`_target`有误。）

        整体试试吧。
        """
    )
    return


@app.cell
def __(l_by_y_fin, linalg, x_fin):
    _target = -x_fin
    _target[[0, -1]] = 0
    y_fin = linalg.solve(l_by_y_fin, _target)
    y_fin
    return (y_fin,)


@app.cell
def __(plt, x_fin, y_fin):
    plt.plot(x_fin, y_fin, marker="+")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()
    plt.gcf()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2 配置法（`spl`: B-spline）

        采用三次B样条，配置 $11$ 点。
        """
    )
    return


@app.cell
def __(np, x_max, x_min):
    n_spl = 11
    x_spl = np.linspace(x_min, x_max, n_spl)
    dx_spl = np.diff(x_spl).mean()
    x_spl
    return dx_spl, n_spl, x_spl


@app.cell
def __(dx_spl, multi_diag, x_spl):
    # y ← 组合系数
    _y = multi_diag([1 / 6, 2 / 3, 1 / 6], x_spl.size)
    # y'' ← 组合系数
    _dv_y_2 = multi_diag([1, -2, 1], x_spl.size) / dx_spl**2

    # ℒy = y'' + y ← 组合系数 coefficients
    l_by_c_spl = _dv_y_2 + _y

    # 两端直接用边界条件
    l_by_c_spl[[0, -1], :] = 0
    l_by_c_spl[0, :2] = [2 / 3, 1 / 6]
    l_by_c_spl[-1, -2:] = [1 / 6, 2 / 3]

    l_by_c_spl
    return (l_by_c_spl,)


@app.cell
def __(l_by_c_spl, linalg, x_fin):
    _target = -x_fin
    _target[[0, -1]] = 0
    y_spl = linalg.solve(l_by_c_spl, _target)
    y_spl
    return (y_spl,)


@app.cell
def __(x_spl, y_spl):
    from scipy.interpolate import BSpline

    spl = BSpline(x_spl, y_spl, k=3)
    return BSpline, spl


@app.cell
def __(np, plt, spl):
    _x = np.linspace(0, 1, 123)
    _fig, _axs = plt.subplots(nrows=2, sharex=True)
    _axs[0].plot(_x, spl(_x))
    _axs[0].set_ylabel("$y$")
    _axs[1].plot(_x, spl.derivative(2)(_x))
    _axs[1].set_ylabel("$y''$")

    _axs[-1].set_xlabel("$x$")
    for ax in _axs:
        ax.grid()

    _fig
    return (ax,)


@app.cell
def __(mo):
    mo.md(r"""呃，形状对，倍数不对……""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3 最小二乘法

        采样幂函数 $1,x,\ldots, x^{10}$。
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 4 误差曲线""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 5 打靶法""")
    return


if __name__ == "__main__":
    app.run()
