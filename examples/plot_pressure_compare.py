#!/usr/bin/env python3
"""
Plot interactive comparison of model variants over a pressure-test sweep.

The script now renders three sections in a single HTML page:
1) Metrics versus arrival_rate (ratio)
2) Metrics versus p99 latency
3) Metrics versus average latency

This makes it easy to read "same ratio" and "same latency" views for throughput,
GPU utilization, and H2D metrics.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRIC_LABELS = {
    "throughput_qps": "Throughput (qps)",
    "avg_ms": "Latency Avg (ms)",
    "p50_ms": "Latency P50 (ms)",
    "p95_ms": "Latency P95 (ms)",
    "p99_ms": "Latency P99 (ms)",
    "avg_batch_size": "Average Batch Size",
    "max_batch_size": "Max Batch Size",
    "avg_queue_ms": "Queue Time Avg (ms)",
    "avg_h2d_only_ms": "H2D Only (ms)",
    "avg_h2d_pure_ms": "H2D Pure (ms)",
    "h2d_bubble_ms": "H2D Bubble (ms)",
    "h2d_bubble_ratio_pct": "H2D Bubble Ratio (%)",
    "avg_concat_ms": "Concat (ms)",
    "sm_util_avg_pct": "SM Util (%)",
    "gpu_util_avg_pct": "GPU Util (%)",
    "avg_h2d_total_numel": "H2D Total Elements",
    "avg_h2d_data_numel": "H2D Data Elements",
    "avg_h2d_meta_numel": "H2D Meta Elements",
}

# Default plotting order keeps commonly read metrics first.
DEFAULT_METRIC_ORDER = [
    "throughput_qps",
    "avg_ms",
    "p99_ms",
    "sm_util_avg_pct",
    "gpu_util_avg_pct",
    "avg_queue_ms",
    "avg_batch_size",
    "max_batch_size",
    "avg_h2d_only_ms",
    "avg_h2d_pure_ms",
    "h2d_bubble_ms",
    "h2d_bubble_ratio_pct",
    "avg_concat_ms",
    "avg_h2d_total_numel",
    "avg_h2d_data_numel",
    "avg_h2d_meta_numel",
    "p50_ms",
    "p95_ms",
]


def build_section(
    df: pd.DataFrame,
    metrics: list[str],
    x_field: str,
    title: str,
    cols: int,
    clip_pct: float | None,
    col_width: int,
    row_height: int,
    marker_mode: str,
    line_width: float,
    x_range: tuple[float | None, float | None] | None = None,
) -> go.Figure:
    usable_metrics = [m for m in metrics if m in df.columns and m != x_field]
    if not usable_metrics:
        raise ValueError(f"No plottable metrics for x={x_field}; requested={metrics}")

    x_vals = pd.to_numeric(df[x_field], errors="coerce")
    x_min = x_vals.min()
    x_max = x_vals.max()
    if x_range is not None:
        left = x_range[0] if x_range[0] is not None else x_min
        right = x_range[1] if x_range[1] is not None else x_max
        resolved_range = [left, right]
    else:
        resolved_range = None

    rows = math.ceil(len(usable_metrics) / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.1,
        subplot_titles=[METRIC_LABELS.get(m, m) for m in usable_metrics],
    )

    modes = list(df["mode"].unique())
    for idx, metric in enumerate(usable_metrics):
        row = idx // cols + 1
        col = idx % cols + 1
        y_values = []
        for mode in modes:
            sub = df[df["mode"] == mode]
            y_values.extend(sub[metric].tolist())
            fig.add_trace(
                go.Scatter(
                    x=sub[x_field],
                    y=sub[metric],
                    mode="lines+markers" if marker_mode == "auto" else "lines",
                    name=mode,
                    legendgroup=mode,
                    showlegend=(idx == 0),
                    hovertemplate=(
                        f"mode=%{{text}}<br>{x_field}=%{{x}}<br>{metric}=%{{y}}<extra></extra>"
                    ),
                    text=sub["mode"],
                    line={"width": line_width},
                    marker={"size": 6} if marker_mode == "auto" else {"size": 0},
                ),
                row=row,
                col=col,
            )
        if clip_pct:
            series = pd.Series(y_values).dropna()
            if not series.empty:
                upper = series.quantile(clip_pct / 100.0)
                lower = series.min()
                if pd.notna(upper):
                    fig.update_yaxes(range=[lower, upper], row=row, col=col)
        fig.update_yaxes(title_text=METRIC_LABELS.get(metric, metric), row=row, col=col)

    x_title = {
        "arrival_rate": "arrival_rate (ratio)",
        "p99_ms": "p99 latency (ms)",
        "avg_ms": "avg latency (ms)",
        "throughput_qps": "throughput (qps)",
    }.get(x_field, x_field)

    for col in range(1, cols + 1):
        fig.update_xaxes(title_text=x_title, row=rows, col=col, range=resolved_range)

    fig.update_layout(
        title=title,
        legend_title="mode",
        hovermode="x unified",
        template="plotly_white",
        height=max(row_height * rows, 400),
        width=max(col_width * cols, 800),
    )
    return fig


def parse_range(spec: str | None) -> tuple[float | None, float | None] | None:
    if spec is None or not spec.strip():
        return None
    if ":" not in spec:
        raise ValueError("Range must be in 'left:right' format, blanks allowed")
    left_str, right_str = spec.split(":", 1)
    left = float(left_str) if left_str.strip() else None
    right = float(right_str) if right_str.strip() else None
    return (left, right)


def render_html(figs: list[tuple[str, go.Figure]]) -> str:
    parts: list[str] = []
    for idx, (title, fig) in enumerate(figs):
        include_js = "cdn" if idx == 0 else False
        parts.append(f"<h2>{title}</h2>")
        parts.append(
            pio.to_html(
                fig,
                include_plotlyjs=include_js,
                full_html=False,
                config={"displaylogo": False},
            )
        )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Pressure Compare</title></head><body>"
        + "\n".join(parts)
        + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pressure-test metrics for model variants.")
    parser.add_argument(
        "--csv",
        default="examples/batchscheduler_outputs_iobound_0223/batchscheduler_sweep.csv",
        help=(
            "Input CSV path with columns: mode, arrival_rate, throughput_qps, "
            "latency metrics, GPU metrics, H2D metrics, etc."
        ),
    )
    parser.add_argument(
        "--output",
        default="examples/batchscheduler_pressure_compare.html",
        help="Output HTML file path (kept for backward compatibility).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRIC_ORDER,
        help="Metric columns to plot (defaults cover throughput, latency, GPU, and H2D).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of subplot columns per section.",
    )
    parser.add_argument(
        "--clip-pct",
        type=float,
        default=None,
        help="Optional percentile cap for y-axis per metric (e.g., 99 trims outliers).",
    )
    parser.add_argument(
        "--col-width",
        type=int,
        default=800,
        help="Pixels per subplot column for sizing the figure.",
    )
    parser.add_argument(
        "--row-height",
        type=int,
        default=420,
        help="Pixels per subplot row for sizing the figure.",
    )
    parser.add_argument(
        "--markers",
        choices=["auto", "none"],
        default="none",
        help="Show markers on lines ('auto') or hide them ('none') to reduce clutter.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Line width for plotted curves.",
    )
    parser.add_argument(
        "--ratio-range",
        default=None,
        help="Restrict arrival_rate x-axis as 'left:right'; leave empty for auto full range.",
    )
    parser.add_argument(
        "--png",
        help="Optional PNG output path for the arrival_rate section (requires kaleido).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="PNG width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="PNG height in pixels.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["mode", "arrival_rate", "p99_ms", "avg_ms", "throughput_qps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    requested_metrics = args.metrics or DEFAULT_METRIC_ORDER
    known_labels = set(METRIC_LABELS)
    unknown = [m for m in requested_metrics if m not in df.columns and m not in known_labels]
    if unknown:
        raise ValueError(f"Requested metrics not found in CSV: {unknown}")

    ratio_range = parse_range(args.ratio_range)
    if ratio_range is not None:
        ratio_df = df.copy()
        ratio_df["arrival_rate"] = pd.to_numeric(ratio_df["arrival_rate"], errors="coerce")
        left = ratio_range[0]
        right = ratio_range[1]
        if left is not None:
            ratio_df = ratio_df[ratio_df["arrival_rate"] >= left]
        if right is not None:
            ratio_df = ratio_df[ratio_df["arrival_rate"] <= right]
    else:
        ratio_df = df

    sections: list[tuple[str, go.Figure]] = []
    sections.append(
        (
            "Metrics vs arrival_rate (ratio)",
            build_section(
                ratio_df,
                requested_metrics,
                "arrival_rate",
                "Metrics vs arrival_rate (ratio)",
                args.cols,
                args.clip_pct,
                args.col_width,
                args.row_height,
                args.markers,
                args.line_width,
                ratio_range,
            ),
        )
    )
    sections.append(
        (
            "Metrics vs throughput",
            build_section(
                df,
                requested_metrics,
                "throughput_qps",
                "Metrics vs throughput",
                args.cols,
                args.clip_pct,
                args.col_width,
                args.row_height,
                args.markers,
                args.line_width,
            ),
        ),
    )
    sections.append(
        (
            "Metrics vs p99 latency",
            build_section(
                df,
                requested_metrics,
                "p99_ms",
                "Metrics vs p99 latency",
                args.cols,
                args.clip_pct,
                args.col_width,
                args.row_height,
                    args.markers,
                    args.line_width,
            ),
        )
    )
    sections.append(
        (
            "Metrics vs average latency",
            build_section(
                df,
                requested_metrics,
                "avg_ms",
                "Metrics vs average latency",
                args.cols,
                args.clip_pct,
                args.col_width,
                args.row_height,
                    args.markers,
                    args.line_width,
            ),
        )
    )

    html = render_html(sections)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML to {output_path}")

    if args.png:
        png_path = Path(args.png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            sections[0][1].write_image(
                str(png_path), format="png", width=args.width, height=args.height
            )
        except Exception as exc:  # plotly will raise if kaleido is missing
            msg = (
                "PNG export failed. Ensure kaleido is installed: "
                "`pip install -U kaleido` or `conda install -c conda-forge python-kaleido`."
            )
            raise RuntimeError(msg) from exc
        print(f"Saved PNG to {png_path} (width={args.width}, height={args.height})")


if __name__ == "__main__":
    main()
