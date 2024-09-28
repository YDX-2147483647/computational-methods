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
    from numpy import cos, linalg, sin

    np.set_printoptions(precision=3, suppress=True)
    return cos, linalg, np, sin


@app.cell
def __():
    from matplotlib import pyplot as plt

    plt.rcParams["font.family"] = "Source Han Serif CN"
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 若干工具函数""")
    return


@app.cell
def __(multi_diag):
    multi_diag([1, 2, 3], size=5)
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


@app.cell
def __(mo, typst):
    mo.md(r"$\frac12 y = \mathcal{L} y$"), typst(r"$1/2 y = cal(L) y$")
    return


@app.cell(hide_code=True)
def __(mo):
    from functools import cache
    from subprocess import CalledProcessError, run
    from sys import stderr
    from tempfile import TemporaryFile


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
        with TemporaryFile() as f:
            try:
                run(
                    # Support for stdout is implemented, but not released yet
                    # https://github.com/typst/typst/pull/3632
                    ["typst", "compile", "-", f.name, "--format", format],
                    input=(prelude + typ).encode(),
                    check=True,
                    capture_output=True,
                )
                return f.read()
            except CalledProcessError as err:
                stderr.write(err.stderr.decode())
                raise err


    def typst(typ: str) -> mo.Html:
        """Write typst"""
        return mo.Html(_typst_compile(typ).decode())
    return CalledProcessError, TemporaryFile, cache, run, stderr, typst


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


@app.cell(hide_code=True)
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
    c_spl = linalg.solve(l_by_c_spl, _target)
    c_spl
    return (c_spl,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""SciPy的B-样条左对齐，而我们中心对称，故需平移并在区间外两侧补结点。""")
    return


@app.cell(hide_code=True)
def __(np, plt):
    from scipy.interpolate import BSpline

    _spl = BSpline.basis_element(np.arange(5))
    assert _spl.k == 3

    for _l, _f in [
        ("y", _spl),
        ("y'", _spl.derivative()),
        ("y''", _spl.derivative(2)),
    ]:
        print(f"{_l:4} =", _f(np.arange(5)))

    _x = np.linspace(0, 4, 123)
    _fig, _axs = plt.subplots(nrows=3, sharex=True)

    _axs[0].plot(_x, _spl(_x))
    _axs[0].set_yticks(np.arange(6) / 6)
    _axs[0].set_ylabel("$y$")

    _axs[1].plot(_x, _spl.derivative()(_x))
    _axs[1].set_ylabel("$y'$")

    _axs[2].plot(_x, _spl.derivative(2)(_x))
    _axs[2].set_ylabel("$y''$")

    _axs[0].set_title("SciPy的B-样条")
    _axs[-1].set_xlabel("$x$")
    for _ax in _axs:
        _ax.set_xticks(np.arange(5))
        _ax.grid()

    _fig
    return (BSpline,)


@app.cell
def __(BSpline, c_spl, dx_spl, np, x_max, x_min, x_spl):
    spl = BSpline(
        np.concat(
            # 两侧各补两个结点，再外加一个空基（动机见下）
            [
                x_min + np.arange(-3, 0) * dx_spl,
                x_spl,
                x_max + np.arange(1, 4) * dx_spl,
            ]
        ),
        # 两侧各补一个空基，让SciPy把整个区间都看作内插
        np.concat([[0], c_spl, [0]]),
        k=3,
        # 让外插部分是 NaN，方便调试；其实不影响内插部分
        extrapolate=False,
    )
    return (spl,)


@app.cell(hide_code=True)
def __(np, plt, spl):
    _x = np.linspace(0, 1, 123)
    _fig, _axs = plt.subplots(nrows=2, sharex=True)
    _axs[0].plot(_x, spl(_x))
    _axs[0].set_ylabel("$y$")
    _axs[1].plot(_x, spl.derivative(2)(_x))
    _axs[1].set_ylabel("$y''$")

    _axs[-1].set_xlabel("$x$")
    for _ax in _axs:
        _ax.grid()

    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        边界处导数崩了，这很正常，因为方程根本没规定边界的导数。

        验证一下具体数字：
        """
    )
    return


@app.cell
def __(spl, x_spl):
    spl(x_spl)
    return


@app.cell
def __(c_spl, l_by_c_spl):
    l_by_c_spl @ c_spl
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3 最小二乘法（`ls`: least squares）

        采样幂函数 $1,x,\ldots, x^{10}$。
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        $\left<\square, \triangle \right> \coloneqq \int_0^1 \square \triangle \mathrm{d} x.$

        $$
        \left<x^n, x^m \right> = \int_0^1 x^n x^m \mathrm{d} x = \frac{1}{m+n+1}.
        $$

        $$
        \mathcal{L} x^n = \begin{cases}
            n(n-1) x^{n-2} + x^n & n \geq 2 \\
            0 & n \in \{0,1\} \\
        \end{cases}.
        $$

        $$
        \left<\mathcal{L} x^n, x^m \right> = \begin{cases}
            \frac{n(n-1)}{m+n-1} + \frac{1}{m+n+1} & n \geq 2 \\
            \frac{1}{m+n+1} & n \in \{0,1\} \\
        \end{cases}.
        $$
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        可以看到，$1,x$ 这俩基很讨厌。我们提前处理边界条件，化成齐次边界条件，顺带扔掉它们。

        - 由 $y(0) = 0$，$1$ 的组合系数为零，直接忽略。
        - 由 $y(1) = 1$，所有基的系数之和为 $1$。

        我们可把基换为 $x^2 - x, \ldots , x^{10} - x$，不过这和 Lagrange 乘数法没有本质区别，所以还是选成 $x, \ldots, x^{10}$ 吧，这也足够排除 $\frac10, 0^0$ 等问题了。
        """
    )
    return


@app.cell
def __(np):
    n_ls = 10
    ns_ls = np.arange(n_ls) + 1
    ns_ls
    return n_ls, ns_ls


@app.cell
def __(np, ns_ls):
    _m = ns_ls[np.newaxis, :]
    _n = ns_ls[:, np.newaxis]

    l_by_c_ls = _n * (_n - 1) / (_m + _n - 1) + 1 / (_m + _n + 1)
    l_by_c_ls
    return (l_by_c_ls,)


@app.cell
def __(l_by_c_ls, linalg, n_ls, np, ns_ls):
    # 补上λ
    _c_and_λ = linalg.solve(
        np.block(
            [
                [l_by_c_ls, -np.ones((n_ls, 1))],
                [np.ones(n_ls), 0],
            ]
        ),
        np.append(-1 / (2 + ns_ls), 0),
    )
    c_ls = _c_and_λ[:-1]
    c_ls, _c_and_λ[-1]
    return (c_ls,)


@app.cell
def __(c_ls, np, ns_ls):
    def ls(x: np.array) -> np.array:
        """Calculate y by least squares"""
        assert x.ndim == 1
        return c_ls @ x ** ns_ls[:, np.newaxis]
    return (ls,)


@app.cell
def __(c_ls, np, ns_ls):
    _n = ns_ls[:, np.newaxis]


    def ls_dv_2(x: np.array) -> np.array:
        """Calculate y'' by least squares"""
        assert x.ndim == 1

        # 0 * (0 ** -1) = 0 * inf → 0
        with np.errstate(divide="ignore", invalid="ignore"):
            _dv_y_2 = c_ls @ (_n * (_n - 1) * x ** (_n - 2))
        return np.nan_to_num(_dv_y_2, 0)
    return (ls_dv_2,)


@app.cell(hide_code=True)
def __(ls, ls_dv_2, np, plt):
    _x = np.linspace(0, 1, 123)

    _fig, _axs = plt.subplots(nrows=3, sharex=True)

    _axs[0].plot(_x, ls(_x))
    _axs[0].set_ylabel("$y$")
    _axs[1].plot(_x, ls_dv_2(_x))
    _axs[1].set_ylabel("$y''$")
    _axs[2].plot(_x, ls_dv_2(_x) + ls(_x) + _x)
    _axs[2].set_ylabel("$y'' + y + x$")

    _axs[-1].set_xlabel("$x$")
    for _ax in _axs:
        _ax.grid()

    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 4 误差曲线""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### $\hat y - y$""")
    return


@app.cell(hide_code=True)
def __(ls, np, plt, ref, spl, x_fin, y_fin):
    _x = np.linspace(0, 1, 123)

    plt.plot(_x, ref(_x), label="真解", linewidth=5, alpha=0.5)
    plt.plot(x_fin, y_fin, label="差分法", marker="+")
    plt.plot(_x, spl(_x), label="配置法")
    plt.plot(_x, ls(_x), label="最小二乘法")

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()
    plt.legend()
    return


@app.cell(hide_code=True)
def __(ls, mo, np, plt, ref, spl, x_fin, y_fin):
    _x = np.linspace(0, 1, 123)

    plt.plot(x_fin, y_fin - ref(x_fin), label="差分法", marker="+")
    plt.plot(_x, spl(_x) - ref(_x), label="配置法")
    plt.plot(_x, ls(_x) - ref(_x), label="最小二乘法")

    # 量级差太多了，缩放也救不过来……
    # plt.yscale("asinh")

    plt.xlabel("$x$")
    plt.ylabel(r"$\hat y - y$")
    plt.title("误差")
    plt.grid()
    plt.legend()

    mo.md(f"""
    | 方法 | 误差绝对值平均 |
    |:-:|:--|
    |差分法（fin）|{abs(y_fin - ref(x_fin)).mean():.3}|
    |配置法（spl）|{abs(spl(_x) - ref(_x)).mean():.3}|
    |最小二乘法（ls）|{abs(ls(_x) - ref(_x)).mean():.3}|

    {mo.as_html(plt.gcf())}
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {
            "误差“平均”方法": r"""
            - 差分法没有给出连续函数，平均误差只用列方程的那些采样点计算；
            - 其它方法给出了连续函数，平均误差是按重新采样的更密的点计算。
            """
        }
    )
    return


@app.cell(hide_code=True)
def __(ls, np, plt, ref, spl, x_fin, y_fin):
    _x = np.linspace(0, 1, 123)
    _c = plt.rcParams["axes.prop_cycle"]()

    _fig, _axs = plt.subplots(nrows=3, sharex=True)

    _axs[0].plot(x_fin, y_fin - ref(x_fin), marker="+", c=next(_c)["color"])
    _axs[0].set_ylabel("差分法")

    _axs[1].plot(_x, spl(_x) - ref(_x), c=next(_c)["color"])
    _axs[1].set_ylabel("配置法")

    _axs[2].plot(_x, ls(_x) - ref(_x), c=next(_c)["color"])
    _axs[2].set_ylabel("最小二乘法")

    _axs[-1].set_xlabel("$x$")

    for _ax in _axs:
        _ax.grid()

    _fig.suptitle(r"误差 $\hat y - y$")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### $\mathcal{L} \hat y - \mathcal{L} y$

        差分法没有构造出连续函数，所以忽略。
        """
    )
    return


@app.cell(hide_code=True)
def __(ls, ls_dv_2, np, plt, spl):
    _x = np.linspace(0, 1, 123)

    plt.plot(_x, -_x, label="真解", linewidth=5, alpha=0.5)
    plt.plot(_x, spl.derivative(2)(_x) + spl(_x), label="配置法")
    plt.plot(_x, ls_dv_2(_x) + ls(_x), label="最小二乘法")

    plt.xlabel("$x$")
    plt.ylabel(r"$\mathcal{L} y$")
    plt.grid()
    plt.legend()
    return


@app.cell(hide_code=True)
def __(ls, ls_dv_2, mo, np, plt, spl):
    _x = np.linspace(0, 1, 123)
    _c = plt.rcParams["axes.prop_cycle"]()

    _fig, _axs = plt.subplots(nrows=2, sharex=True)

    _error_spl = spl.derivative(2)(_x) + spl(_x) + _x
    _axs[0].plot(_x, _error_spl, c=next(_c)["color"])
    _axs[0].set_ylabel("配置法")

    _error_ls = ls_dv_2(_x) + ls(_x) + _x
    _axs[1].plot(_x, _error_ls, c=next(_c)["color"])
    _axs[1].set_ylabel("最小二乘法")

    _axs[-1].set_xlabel("$x$")

    for _ax in _axs:
        _ax.grid()

    _fig.suptitle(r"误差 $\mathcal{L} \hat y - \mathcal{L} y$")


    mo.md(f"""
    |方法|误差绝对值平均|
    |:-:|:--|
    |差分法（fin）|N/A|
    |配置法（spl）|{abs(_error_spl).mean():.3}|
    |最小二乘法（ls）|{abs(_error_ls).mean():.3}|

    {mo.as_html(_fig)}
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 5 打靶法""")
    return


@app.cell(hide_code=True)
def __(typst):
    typst(r"""
    #import "@preview/physica:0.9.3": eval, dv

    设 $z:=y'$，则

    $ dv(,x) mat(y;z) = mat(z; -y) + mat(0; -x). $
    $eval(y)_0 = 0$, $eval(y)_1 = 0$.
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 试一试""")
    return


@app.cell(hide_code=True)
def __(cos, np, plt, ref, sin):
    _x = np.linspace(0, 1, 123)

    plt.plot(ref(_x), cos(_x) / sin(1) - 1)
    plt.scatter([0], [1 / sin(1) - 1])

    plt.grid()
    plt.xlabel("$y$")
    plt.ylabel(r"$z = y'$")
    plt.title("真解")
    return


@app.cell
def __(mo):
    the_x = mo.ui.slider(0, 1, 0.1)
    the_x
    return (the_x,)


@app.cell(hide_code=True)
def __(np, plt, the_x):
    _y = np.linspace(-1, 1, 23)
    _z = np.linspace(-1, 1, _y.size)

    _ys, _zs = np.meshgrid(_y, _z)

    fig, ax = plt.subplots()
    ax.quiver(_y, _z, _zs, -_ys - the_x.value)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$z = y'$")
    ax.set_title(rf"$x = {the_x.value}$ 时的场")
    fig
    return ax, fig


@app.cell
def __(mo, sin):
    the_y_0 = mo.ui.slider(-0.5, 0.5, 0.1, value=0)
    the_z_0 = mo.ui.slider(-0.5, 0.5, 0.1, value=1 / sin(1) - 1)
    mo.md(f"$y_0$: {the_y_0}\n\n$z_0$: {the_z_0}")
    return the_y_0, the_z_0


@app.cell(hide_code=True)
def __(cos, np, plt, sin, the_y_0, the_z_0):
    _x = np.linspace(0, 1, 123)

    plt.plot(
        sin(_x) * (1 + the_z_0.value) + cos(_x) * the_y_0.value - _x,
        cos(_x) * (1 + the_z_0.value) - sin(_x) * the_y_0.value - 1,
    )
    plt.scatter([the_y_0.value], [the_z_0.value])

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.grid()
    plt.xlabel("$y$")
    plt.ylabel(r"$z = y'$")
    return


@app.cell
def __(mo):
    the_x_max = mo.ui.slider(0.1, 1, 0.1, value=0.8)
    the_x_max
    return (the_x_max,)


@app.cell(hide_code=True)
def __(cos, np, plt, sin, the_x_max):
    _x = np.arange(0, the_x_max.value, 0.04)[:, np.newaxis]

    _y_0, _z_0 = np.meshgrid(
        np.linspace(-0.5, 0.5, 14),
        np.linspace(-0.5, 0.5, 14),
    )
    _y_0 = _y_0.reshape(1, -1)
    _z_0 = _z_0.reshape(1, -1)

    plt.plot(
        sin(_x) * (1 + _z_0) + cos(_x) * _y_0 - _x,
        cos(_x) * (1 + _z_0) - sin(_x) * _y_0 - 1,
        alpha=0.6,
    )

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.grid()
    plt.xlabel("$y$")
    plt.ylabel(r"$z = y'$")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 正式试""")
    return


@app.cell
def __(np, sin, x_max, x_min):
    from collections import deque

    from scipy.integrate import ode


    def f_shoot(x: float, y_z: np.array) -> list[float]:
        """(y,z) → (y',z')"""
        return [y_z[1], -y_z[0] - x]


    _r = ode(f_shoot).set_integrator("vode")


    @np.vectorize
    def shoot(z_0: float, *, y_0=0.0, dx=0.02) -> float:
        """z₀ ↦ y₁"""
        _r.set_initial_value([y_0, z_0], x_min)

        while _r.successful():
            y_z = _r.integrate(_r.t + dx)
            if _r.t >= x_max:
                return y_z[0]


    shoot(1 / sin(1) - 1)
    return deque, f_shoot, ode, shoot


@app.cell
def __(np, plt, shoot):
    _z_0 = np.linspace(-0.5, 0.5, 14)
    _y_1 = shoot(_z_0)

    plt.plot(_z_0, _y_1, marker="+")
    plt.xlabel(r"$z_0$")
    plt.ylabel(r"$y_1$")
    plt.hlines(0, *plt.xlim(), "red")
    plt.grid()
    plt.gcf()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""回头一看，还真是一次函数呢……""")
    return


@app.cell
def __(shoot):
    from scipy.optimize import fsolve

    z_0_shoot = fsolve(shoot, 0)
    assert z_0_shoot.size == 1
    z_0_shoot = z_0_shoot[0]

    z_0_shoot, shoot(z_0_shoot)
    return fsolve, z_0_shoot


@app.cell
def __(mo, sin, z_0_shoot):
    mo.md(f"$z_0$ 相对误差：{z_0_shoot / (1 / sin(1) - 1) - 1:.3}")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""用与其它方法计算误差时相同的步长计算 $y$，算一下 $y$ 的误差。""")
    return


@app.cell(hide_code=True)
def __(deque, f_shoot, mo, np, ode, plt, ref, x_min, z_0_shoot):
    _r = ode(f_shoot).set_integrator("vode")
    _r.set_initial_value([0.0, z_0_shoot], x_min)

    _x = np.linspace(0, 1, 123)
    _y = deque()
    for _xx in _x:
        if _xx == 0.0:
            _y.append(0.0)
        else:
            assert _r.successful()
            _y.append(_r.integrate(_xx)[0])
    _y = np.array(list(_y))

    plt.plot(_x, _y - ref(_x))
    plt.xlabel("$x$")
    plt.ylabel(r"$\hat y - y$")
    plt.title("误差")
    plt.grid()

    mo.md(rf"""
    误差绝对值平均：{abs(_y - ref(_x)).mean():.3}
    {mo.as_html(plt.gcf())}
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""在最小二乘法和差分法之间，不过对打靶法对微分方程满足得更好。""")
    return


if __name__ == "__main__":
    app.run()
