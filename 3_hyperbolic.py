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
    #import "@preview/physica:0.9.3": pdv, eval

    $
    pdv(u,t) + 5 pdv(u,x) = 0, x in RR. \
    eval(u)_(t=0) = cases(
      2 &"if" x in (-1,0),
      1 &"if" x = 0,
      -2 &"if" x in (0,1),
      0 &"otherwise"
    ).
    $
    """)
    return


if __name__ == "__main__":
    app.run()
