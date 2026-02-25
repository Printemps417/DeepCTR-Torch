#!/usr/bin/env python3
"""
Stage-wise ratio vs latency plots for batch scheduler sweeps.

Creates one figure per ratio stage (e.g., 0-2500 and 2500-5000) to improve readability
when outliers or long tails compress the view. Focuses on latency-related metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Default metrics focused on latency and queuing.
DEFAULT_METRICS = ["avg_ms", "p50_ms", "p95_ms", "p99_ms", "avg_queue_ms"]
METRIC_LABELS = {
    "avg_ms": "Latency Avg (ms)",
    "p50_ms": "Latency P50 (ms)",
    "p95_ms": "Latency P95 (ms)",
    "p99_ms": "Latency P99 (ms)",
    "avg_queue_ms": "Queue Time Avg (ms)",
}


def parse_stages(stage_args: List[str]) -> List[Tuple[float, float]]:
    stages: List[Tuple[float, float]] = []
    for item in stage_args:
        if ":" not in item:
            raise ValueError(f"Stage must use start:end format, got '{item}'")
        start_str, end_str = item.split(":", 1)
        stages.append((float(start_str), float(end_str)))
    return stages


def build_stage_fig(df: pd.DataFrame, metrics: List[str], stage: Tuple[float, float], cols: int, line_width: float) -> go.Figure:
    start, end = stage
    sub_df = df[(df["arrival_rate"] >= start) & (df["arrival_rate"] < end)].copy()
    if sub_df.empty:
        raise ValueError(f"No data in stage range [{start}, {end})")

    used_metrics = [m for m in metrics if m in sub_df.columns and m != "arrival_rate"]
    rows = (len(used_metrics) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=[METRIC_LABELS.get(m, m) for m in used_metrics],
    )

    modes = list(sub_df["mode"].unique())
    for idx, metric in enumerate(used_metrics):
        row = idx // cols + 1
        col = idx % cols + 1
        for mode in modes:
            part = sub_df[sub_df["mode"] == mode]
            fig.add_trace(
                go.Scatter(
                    x=part["arrival_rate"],
                    y=part[metric],
                    mode="lines",
                    name=mode,
                    legendgroup=mode,
                    showlegend=(idx == 0),
                    line={"width": line_width},
                    hovertemplate="mode=%{text}<br>ratio=%{x}<br>value=%{y}<extra></extra>",
                    text=part["mode"],
                ),
                row=row,
                col=col,
            )
        fig.update_yaxes(title_text=METRIC_LABELS.get(metric, metric), row=row, col=col)

    for col in range(1, cols + 1):
        fig.update_xaxes(title_text="arrival_rate (ratio)", row=rows, col=col)

    fig.update_layout(
        title=f"Ratio stage [{start}, {end})",
        legend_title="mode",
        hovermode="x unified",
        template="plotly_white",
        height=max(360 * rows, 420),
        width=max(900, 450 * cols),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ratio vs latency per stage to reduce compression from outliers.")
    parser.add_argument(
        "--csv",
        default="examples/batchscheduler_outputs_iobound_4mode/batchscheduler_sweep.csv",
        help="Input CSV path (requires columns: mode, arrival_rate, latency metrics).",
    )
    parser.add_argument(
        "--output",
        default="examples/batchscheduler_ratio_segments.html",
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=["0:2500", "2500:5000"],
        help="Ratio stage ranges like '0:2500 2500:5000'.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to plot per stage (latency focused).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Subplot columns per figure.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.5,
        help="Line width for curves.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["mode", "arrival_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    stages = parse_stages(args.stages)
    figures = []
    for stage in stages:
        fig = build_stage_fig(df, args.metrics, stage, args.cols, args.line_width)
        figures.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # Compose HTML with a single PlotlyJS include for size efficiency.
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Ratio Segmented Latency</title>"
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
        "</head><body>"
        + "\n".join(figures)
        + "</body></html>"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML to {output_path}")


if __name__ == "__main__":
    main()
