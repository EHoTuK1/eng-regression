"""Morphing visualization: probability space -> log-odds space. Saved to outputs/logit_transform_*."""

# plotly ships no/incomplete type stubs, so these reports are noise for this file only
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

Arr = NDArray[np.float64]

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREFIX = "logit_transform_"
LIMIT = 5.0
Y_RANGE = [-0.08, 1.08]
TICK_PROBS = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
HOLD = 8


def _path(name: str, ext: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{PREFIX}{name}.{ext}"


def _sigmoid(z: Arr) -> Arr:
    return 1 / (1 + np.exp(-z))


def _scaled_logit(p: Arr) -> Arr:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (np.clip(np.log(p / (1 - p)), -LIMIT, LIMIT) + LIMIT) / (2 * LIMIT)


def _warp(p: Arr, alpha: float) -> Arr:
    return (1 - alpha) * p + alpha * _scaled_logit(p)


def _bin_shares(X: Arr, Y: Arr, width: float = 1.0) -> tuple[Arr, Arr]:
    edges = np.arange(np.floor(X.min()), np.ceil(X.max()) + width, width)
    centers: list[float] = []
    shares: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (lo <= X) & (hi > X)
        if mask.sum() >= 3:
            centers.append(float((lo + hi) / 2))
            shares.append(float(np.clip(Y[mask].mean(), 0.02, 0.98)))
    return np.array(centers), np.array(shares)


def _ticks(alpha: float) -> dict[str, Any]:
    z = np.log(TICK_PROBS / (1 - TICK_PROBS))
    labels = [f"p={p:g} (z={zi:+.1f})" for p, zi in zip(TICK_PROBS, z, strict=True)]
    return dict(tickvals=_warp(TICK_PROBS, alpha).tolist(), ticktext=labels)


def _frame_layout(alpha: float) -> go.Layout:
    title = (
        "Probability space: p = sigmoid(w·x + b)"
        if alpha == 0
        else "Log-odds space: ln(p / (1 − p)) = w·x + b"
        if alpha == 1
        else f"Warping the y-axis: p → ln(p / (1 − p)), α = {alpha:.2f}"
    )
    y_title = "Probability of passing" if alpha < 0.5 else "Log-odds (probability labels)"
    return go.Layout(title=title, yaxis=dict(title=y_title, **_ticks(alpha)))


def _frame_traces(
    X: Arr, Y: Arr, p_hat: Arr, xs: Arr, curve_p: Arr, bins: tuple[Arr, Arr], alpha: float
) -> list[go.Scatter]:
    passed = Y == 1
    return [
        go.Scatter(
            x=xs, y=_warp(curve_p, alpha), mode="lines", line=dict(color="crimson", width=3)
        ),
        go.Scatter(
            x=X[passed],
            y=_warp(p_hat[passed], alpha),
            mode="markers",
            marker=dict(color="seagreen", size=8),
        ),
        go.Scatter(
            x=X[~passed],
            y=_warp(p_hat[~passed], alpha),
            mode="markers",
            marker=dict(color="indianred", size=8),
        ),
        go.Scatter(
            x=bins[0],
            y=_warp(bins[1], alpha),
            mode="markers",
            marker=dict(color="black", size=11, symbol="diamond-open", line=dict(width=2)),
        ),
    ]


def _save_gif(fig: go.Figure, name: str) -> None:
    try:
        from PIL import Image

        images: list[Any] = []
        for fr in cast("Any", fig.frames):
            f = go.Figure(fig)
            f.layout.updatemenus, f.layout.sliders = (), ()
            f.update_layout(fr.layout, margin=dict(b=50))
            for tr_i, tr in zip(fr.traces, fr.data, strict=True):
                f.data[tr_i].update(tr)
            png = f.to_image(format="png", width=1000, height=520)
            images.append(Image.open(BytesIO(png)).convert("P", palette=Image.Palette.ADAPTIVE))
        path = _path(name, "gif")
        images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
        print("saved:", path)
    except Exception as e:
        print(f"GIF export skipped ({type(e).__name__}). To enable it run: uv add kaleido")


def plot_logit_morph(X: Arr, Y: Arr, w: float, b: float) -> go.Figure:
    xs = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
    curve_p = _sigmoid(w * xs + b)
    p_hat = _sigmoid(w * X + b)
    bins = _bin_shares(X, Y)
    x_boundary = -b / w

    steps = np.linspace(0, 1, 36)
    alphas = (3 * steps**2 - 2 * steps**3).tolist()
    names = [f"{a:.3f}" for a in alphas]
    sequence = [(0.0, "hold0_" + str(k)) for k in range(HOLD)]
    sequence += list(zip(alphas, names, strict=True))
    sequence += [(1.0, "hold1_" + str(k)) for k in range(HOLD)]

    fig = go.Figure(data=_frame_traces(X, Y, p_hat, xs, curve_p, bins, 0.0))
    fig.add_hline(y=0.5, line=dict(color="gray", dash="dash", width=1))
    fig.add_vline(x=x_boundary, line=dict(color="gray", dash="dot", width=1))
    fig.frames = [
        go.Frame(
            data=_frame_traces(X, Y, p_hat, xs, curve_p, bins, a),
            traces=[0, 1, 2, 3],
            layout=_frame_layout(round(a, 6)),
            name=nm,
        )
        for a, nm in sequence
    ]
    fig.data[0].name, fig.data[1].name = "model", "passed"
    fig.data[2].name, fig.data[3].name = "failed", "observed pass rate per 1 h"
    for tr in fig.data:
        tr.showlegend = True

    step = dict(
        frame=dict(duration=100, redraw=True), transition=dict(duration=0), mode="immediate"
    )
    pause = dict(frame=dict(duration=0, redraw=False), mode="immediate")
    fig.update_layout(
        template="plotly_white",
        height=600,
        width=1000,
        xaxis=dict(title="Study time, h", range=[xs.min(), xs.max()]),
        yaxis=dict(range=Y_RANGE),
        legend=dict(orientation="h", y=-0.28),
        title_font_size=16,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=-0.42,
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
                y=-0.36,
                len=0.85,
                currentvalue=dict(prefix="α: "),
                steps=[
                    dict(label=f"{a:.2f}", method="animate", args=[[nm], step])
                    for a, nm in zip(alphas, names, strict=True)
                ],
            )
        ],
        margin=dict(l=130, r=20, t=60, b=170),
    )
    fig.update_layout(_frame_layout(0.0))

    html = _path("morph", "html")
    fig.write_html(html, auto_play=False, include_plotlyjs=True)
    print("saved:", html)
    _save_gif(fig, "morph")
    return fig
