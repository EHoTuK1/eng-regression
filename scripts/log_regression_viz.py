"""Visualizations for the logistic regression notebook. Every figure is saved to outputs/log_regression_*."""

# plotly ships no/incomplete type stubs, so these reports are noise for this file only
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

Arr = NDArray[np.float64]
History = dict[str, Arr]

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREFIX = "log_regression_"
Y_RANGE = [-0.08, 1.08]


def _path(name: str, ext: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{PREFIX}{name}.{ext}"


def _sigmoid(z: Arr) -> Arr:
    return 1 / (1 + np.exp(-z))


def _save_plotly(fig: go.Figure, name: str) -> None:
    html = _path(name, "html")
    fig.write_html(html, auto_play=False, include_plotlyjs=True)
    print("saved:", html)
    try:
        png = _path(name, "png")
        fig.write_image(png)
        print("saved:", png)
    except Exception as e:
        print(f"PNG export skipped ({type(e).__name__}). To enable it run: uv add kaleido")


def plot_data(hours: Arr, passed: Arr) -> None:
    plt.figure()
    plt.scatter(hours, passed, s=14, alpha=0.6)
    plt.xlabel("Время подготовки, ч")
    plt.ylabel("Сдал (1) / не сдал (0)")
    path = _path("data", "png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("saved:", path)
    plt.show()


def plot_init(X: Arr, Y: Arr, w: float, b: float, loss: float, acc: float) -> go.Figure:
    xs = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
    fig = go.Figure()
    fig.add_scatter(x=X, y=Y, mode="markers", name="data")
    fig.add_scatter(
        x=xs, y=_sigmoid(w * xs + b), mode="lines", name=f"random curve: w={w:.2f}, b={b:.1f}"
    )
    fig.update_layout(
        title=f"Random initialization (log loss = {loss:.2f}, accuracy = {acc:.0%})",
        xaxis=dict(title="Study time, h", range=[xs.min(), xs.max()]),
        yaxis=dict(title="Probability of passing", range=Y_RANGE),
        template="plotly_white",
        height=550,
        width=1000,
        legend=dict(orientation="h", y=-0.2),
    )
    _save_plotly(fig, "init")
    return fig


def _residual_xy(
    X: Arr, Y: Arr, w: float, b: float
) -> tuple[list[float | None], list[float | None]]:
    p = _sigmoid(w * X + b)
    rx: list[float | None] = []
    ry: list[float | None] = []
    for xi, yi, pi in zip(X, Y, p, strict=True):
        rx += [float(xi), float(xi), None]
        ry += [float(yi), float(pi), None]
    return rx, ry


def _padded(a: Arr, k: float = 0.08) -> list[float]:
    pad = float(a.max() - a.min()) * k or 1.0
    return [float(a.min()) - pad, float(a.max()) + pad]


def _frame_traces(X: Arr, Y: Arr, xs: Arr, hist: History, i: int) -> list[go.Scatter]:
    ep = hist["epoch"]
    w, b = float(hist["w"][i]), float(hist["b"][i])
    rx, ry = _residual_xy(X, Y, w, b)
    traces = [
        go.Scatter(x=xs, y=_sigmoid(w * xs + b), mode="lines", line=dict(color="crimson", width=3)),
        go.Scatter(x=rx, y=ry, mode="lines", line=dict(color="gray", width=1, dash="dot")),
    ]
    for key, color in [
        ("loss", "royalblue"),
        ("acc", "seagreen"),
        ("w", "darkorange"),
        ("b", "mediumpurple"),
    ]:
        traces.append(
            go.Scatter(x=ep[: i + 1], y=hist[key][: i + 1], mode="lines", line=dict(color=color))
        )
        traces.append(
            go.Scatter(
                x=[ep[i]], y=[hist[key][i]], mode="markers", marker=dict(color=color, size=10)
            )
        )
    return traces


def _save_gif(fig: go.Figure, name: str) -> None:
    try:
        from PIL import Image

        images: list[Any] = []
        for fr in cast("Any", fig.frames):
            f = go.Figure(fig)
            f.layout.updatemenus, f.layout.sliders = (), ()
            f.update_layout(margin=dict(b=50), title=f"Gradient descent, epoch {fr.name}")
            for tr_i, tr in zip(fr.traces, fr.data, strict=True):
                f.data[tr_i].update(tr)
            png = f.to_image(format="png", width=1200, height=520)
            images.append(Image.open(BytesIO(png)).convert("P", palette=Image.Palette.ADAPTIVE))
        images += [images[-1]] * 10
        path = _path(name, "gif")
        images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
        print("saved:", path)
    except Exception as e:
        print(f"GIF export skipped ({type(e).__name__}). To enable it run: uv add kaleido")


def plot_training(X: Arr, Y: Arr, hist: History, w_opt: float, b_opt: float) -> go.Figure:
    epochs = int(hist["epoch"][-1])
    xs = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
    idx = np.unique(np.round(np.geomspace(1, epochs + 1, 50)).astype(int) - 1)
    idx = np.unique(np.r_[0, idx, epochs])

    fig = make_subplots(
        rows=2,
        cols=3,
        column_widths=[0.5, 0.25, 0.25],
        horizontal_spacing=0.07,
        vertical_spacing=0.17,
        specs=[[{"rowspan": 2}, {}, {}], [None, {}, {}]],
        subplot_titles=(
            "Sigmoid fit and errors",
            "Loss: log loss",
            "Accuracy",
            "Weight w",
            "Bias b",
        ),
    )
    fig.add_scatter(x=X, y=Y, mode="markers", marker=dict(color="steelblue", size=7), row=1, col=1)
    cells = [(1, 1), (1, 1), (1, 2), (1, 2), (1, 3), (1, 3), (2, 2), (2, 2), (2, 3), (2, 3)]
    for trace, (r, c) in zip(_frame_traces(X, Y, xs, hist, 0), cells, strict=True):
        fig.add_trace(trace, row=r, col=c)
    fig.frames = [
        go.Frame(
            data=_frame_traces(X, Y, xs, hist, int(i)),
            traces=list(range(1, 11)),
            name=str(int(hist["epoch"][i])),
        )
        for i in idx
    ]

    fig.update_xaxes(range=[xs.min(), xs.max()], title_text="Study time, h", row=1, col=1)
    fig.update_yaxes(range=Y_RANGE, title_text="Probability of passing", row=1, col=1)
    curves = [
        ((1, 2), [0, float(hist["loss"].max()) * 1.05], "log loss"),
        ((1, 3), [0, 1.05], "accuracy"),
        ((2, 2), _padded(hist["w"]), "w, 1/h"),
        ((2, 3), _padded(hist["b"]), "b"),
    ]
    for (r, c), rng, ytitle in curves:
        fig.update_xaxes(range=[-3, epochs + 6], title_text="Epoch", row=r, col=c)
        fig.update_yaxes(range=rng, title_text=ytitle, row=r, col=c)
    for (r, c), v in [((2, 2), w_opt), ((2, 3), b_opt)]:
        fig.add_hline(y=v, line=dict(color="gray", dash="dash", width=1), row=r, col=c)  # pyright: ignore[reportArgumentType]
    fig.add_hline(y=0.5, line=dict(color="gray", dash="dot", width=1), row=1, col=1)  # pyright: ignore[reportArgumentType]
    fig.update_annotations(font_size=13)

    step = dict(
        frame=dict(duration=120, redraw=True), transition=dict(duration=0), mode="immediate"
    )
    pause = dict(frame=dict(duration=0, redraw=False), mode="immediate")
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=650,
        width=1200,
        title="Gradient descent: probability of passing ~ study time",
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=-0.22,
                xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, step]),
                    dict(label="⏸ Pause", method="animate", args=[[None], pause]),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.15,
                y=-0.16,
                len=0.85,
                currentvalue=dict(prefix="Epoch: "),
                steps=[
                    dict(
                        label=str(int(hist["epoch"][i])),
                        method="animate",
                        args=[[str(int(hist["epoch"][i]))], step],
                    )
                    for i in idx
                ],
            )
        ],
        margin=dict(l=55, r=15, t=60, b=110),
        title_font_size=16,
    )
    html = _path("training", "html")
    fig.write_html(html, auto_play=False, include_plotlyjs=True)
    print("saved:", html)
    _save_gif(fig, "training")
    return fig


def plot_prediction(X: Arr, Y: Arr, w: float, b: float, new_hours: float) -> go.Figure:
    prob = float(_sigmoid(np.array(w * new_hours + b)))
    x_lo, x_hi = min(X.min(), new_hours) - 0.5, max(X.max(), new_hours) + 0.5
    xs = np.linspace(x_lo, x_hi, 200)

    fig = go.Figure()
    fig.add_scatter(x=X, y=Y, mode="markers", marker=dict(color="steelblue", size=7), name="data")
    fig.add_scatter(
        x=xs,
        y=_sigmoid(w * xs + b),
        mode="lines",
        line=dict(color="crimson", width=3),
        name="fitted curve",
    )
    fig.add_scatter(
        x=[new_hours, new_hours, x_lo],
        y=[Y_RANGE[0], prob, prob],
        mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        showlegend=False,
    )
    fig.add_scatter(
        x=[new_hours],
        y=[prob],
        mode="markers",
        marker=dict(color="black", size=12, symbol="diamond"),
        name="prediction",
    )
    fig.add_annotation(
        x=new_hours,
        y=prob,
        ax=-110,
        ay=-70,
        showarrow=True,
        arrowhead=2,
        text=f"Study time {new_hours:g} h → pass probability {prob:.0%} ({'pass' if prob >= 0.5 else 'fail'})",
        bgcolor="white",
        bordercolor="black",
    )
    fig.update_layout(
        title="Prediction with the fitted curve",
        template="plotly_white",
        height=480,
        width=1000,
        xaxis=dict(title="Study time, h", range=[x_lo, x_hi]),
        yaxis=dict(title="Probability of passing", range=Y_RANGE),
        legend=dict(orientation="h", y=-0.2),
    )
    _save_plotly(fig, "prediction")
    return fig
