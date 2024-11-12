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
    from numpy import linalg

    np.set_printoptions(precision=3, suppress=True)
    return linalg, np


@app.cell
def __():
    from matplotlib import pyplot as plt
    return (plt,)


@app.cell
def __():
    from util import typst
    return (typst,)


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


if __name__ == "__main__":
    app.run()
