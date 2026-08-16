"""
Generic prediction-evaluation plotting/metrics helpers.

Works on plain arrays (actual price, predicted price, a label per point) rather
than any specific data object, so it can score any of NeuralNetworkAgent,
SpecialistAgent, FrontierAgent, or the full EnsembleAgent the same way -
used by notebooks/04_model_comparison.ipynb.
"""
import math
from itertools import accumulate

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# Status palette (validated defaults - dataviz skill references/palette.md).
# Error-severity is a state, not a series identity, so it uses the fixed
# good/warning/critical status colors rather than a categorical hue.
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
COLOR_CRITICAL = "#d03b3b"
COLOR_MAP = {"good": COLOR_GOOD, "warning": COLOR_WARNING, "critical": COLOR_CRITICAL}


def _severity(error, truth):
    ratio = error / truth if truth else 1.0
    if error < 40 or ratio < 0.2:
        return "good"
    elif error < 80 or ratio < 0.4:
        return "warning"
    return "critical"


def scatter_chart(actual, predicted, labels, title, currency="₹"):
    """Actual vs. predicted scatter, points colored by error severity."""
    errors = [abs(p - a) for a, p in zip(actual, predicted)]
    severities = [_severity(e, a) for e, a in zip(errors, actual)]

    df = pd.DataFrame(
        {"truth": actual, "guess": predicted, "label": labels, "error": errors, "severity": severities}
    )
    df["hover"] = [
        f"{lbl}<br>Guess={currency}{g:,.2f}  Actual={currency}{y:,.2f}"
        for lbl, g, y in zip(df["label"], df["guess"], df["truth"])
    ]

    max_val = float(max(df["truth"].max(), df["guess"].max()))

    fig = px.scatter(
        df,
        x="truth",
        y="guess",
        color="severity",
        color_discrete_map=COLOR_MAP,
        category_orders={"severity": ["good", "warning", "critical"]},
        title=title,
        labels={"truth": f"Actual Price ({currency})", "guess": f"Predicted Price ({currency})"},
        width=900,
        height=700,
    )
    for tr in fig.data:
        mask = df["severity"] == tr.name
        tr.customdata = df.loc[mask, ["hover"]].to_numpy()
        tr.hovertemplate = "%{customdata[0]}<extra></extra>"
        tr.marker.update(size=6)

    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(width=2, dash="dash", color="#2a78d6"),
            name="y = x",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val])
    fig.update_layout(legend_title_text="Error severity")
    return fig


def error_trend_chart(actual, predicted, title, currency="₹"):
    """Cumulative mean absolute error with a 95% CI band, in evaluation order."""
    errors = [abs(p - a) for a, p in zip(actual, predicted)]
    n = len(errors)
    x = list(range(1, n + 1))

    running_sums = list(accumulate(errors))
    running_means = [s / i for s, i in zip(running_sums, x)]
    running_squares = list(accumulate(e * e for e in errors))
    running_stds = [
        math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
        for i, sq_sum, mean in zip(x, running_squares, running_means)
    ]
    ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
    upper = [m + c for m, c in zip(running_means, ci)]
    lower = [m - c for m, c in zip(running_means, ci)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(137,135,129,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="95% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=running_means,
            mode="lines",
            line=dict(width=3, color=COLOR_CRITICAL),
            name="Cumulative Avg Error",
            customdata=list(zip(ci)),
            hovertemplate=(
                "n=%{x}<br>"
                f"Avg Error={currency}" + "%{y:,.2f}<br>"
                f"±95% CI={currency}" + "%{customdata[0]:,.2f}<extra></extra>"
            ),
        )
    )

    final_mean = running_means[-1]
    final_ci = ci[-1]
    fig.update_layout(
        title=f"{title} — Error: {currency}{final_mean:,.2f} ± {currency}{final_ci:,.2f}",
        xaxis_title="Number of Datapoints",
        yaxis_title=f"Average Absolute Error ({currency})",
        width=900,
        height=320,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def report(actual, predicted, labels, title, currency="₹"):
    """
    Score a set of predictions against ground truth.
    Returns (scatter_fig, trend_fig, metrics) where metrics has mae/mse/rmse/r2.
    """
    errors = [abs(p - a) for a, p in zip(actual, predicted)]
    mae = sum(errors) / len(errors)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted) * 100
    metrics = {"mae": mae, "mse": mse, "rmse": mse**0.5, "r2": r2}

    full_title = (
        f"{title} results<br>"
        f"<b>MAE:</b> {currency}{mae:,.2f}  <b>RMSE:</b> {currency}{metrics['rmse']:,.2f}  <b>r²:</b> {r2:.1f}%"
    )
    trend_fig = error_trend_chart(actual, predicted, title, currency)
    scatter_fig = scatter_chart(actual, predicted, labels, full_title, currency)
    return scatter_fig, trend_fig, metrics
