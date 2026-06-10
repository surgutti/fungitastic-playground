from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ENTITY = "university-of-wroclaw-nntp-course"
DEFAULT_PROJECT = "FungiTasticSegmentation"

DEFAULT_SUMMARY_COLUMNS = [
    "Name",
    "State",
    "epoch",
    "trainer/global_step",
    "Runtime",
    "val/mean_iou.max",
    "val/loss.min",
    "val/pixel_acc.max",
    "train/mean_iou.max",
    "train/loss.min",
    "model.backbone.backbone_name",
    "data_module.batch_size",
    "training_cfg.max_epochs",
    "Commit",
]

CLASS_LABELS = {
    0: "background",
    1: "cap",
    2: "stem",
    3: "gills",
    4: "pores",
    5: "ring",
}


def _split_csv_values(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _higher_is_better(metric: str) -> bool:
    metric_lower = metric.lower()
    if "loss" in metric_lower or metric_lower in {"runtime", "_runtime"}:
        return False
    return True


def _sort_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return df
    return df.sort_values(metric, ascending=not _higher_is_better(metric), na_position="last")


def _safe_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def load_from_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_from_wandb(entity: str, project: str, run_names: list[str]) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    selected = []
    wanted = set(run_names)

    for run in runs:
        if wanted and run.name not in wanted:
            continue
        record = {
            "Name": run.name,
            "State": run.state,
            "Runtime": run.summary.get("_runtime"),
            "Commit": getattr(run, "commit", None),
        }
        record.update(dict(run.summary))
        record.update(_flatten_dict(dict(run.config)))
        selected.append(record)

    if not selected:
        raise click.ClickException("No matching W&B runs found.")

    return pd.DataFrame(selected)


def filter_runs(df: pd.DataFrame, run_names: list[str], regex: str | None, include_failed: bool) -> pd.DataFrame:
    if "Name" not in df.columns:
        raise click.ClickException("Input table has no 'Name' column.")

    filtered = df.copy()
    if run_names:
        wanted = set(run_names)
        filtered = filtered[filtered["Name"].isin(wanted)]

    if regex:
        filtered = filtered[filtered["Name"].astype(str).str.contains(regex, regex=True, na=False)]

    if not include_failed and "State" in filtered.columns:
        filtered = filtered[filtered["State"].isin(["finished", "running"])]

    if filtered.empty:
        raise click.ClickException("No runs left after filtering.")

    return filtered


def write_markdown_table(df: pd.DataFrame, path: Path, title: str, columns: list[str]) -> None:
    columns = _safe_columns(df, columns)
    table = df[columns].copy()

    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            table[col] = table[col].fillna("").astype(str)

    lines = [f"# {title}", ""]
    lines.append(f"Generated from {len(df)} run(s).")
    lines.append("")

    if len(table.columns) == 0:
        lines.append("No columns available.")
    else:
        header = "| " + " | ".join(table.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
        lines.extend([header, sep])
        for _, row in table.iterrows():
            values = [str(row[col]).replace("\n", " ") for col in table.columns]
            lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric_bar(df: pd.DataFrame, metric: str, output_path: Path, top_n: int) -> None:
    if metric not in df.columns:
        return

    data = df[["Name", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])
    if data.empty:
        return

    data = _sort_by_metric(data, metric).head(top_n)
    data = data.iloc[::-1]

    fig_h = max(4.0, 0.42 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(data["Name"], data[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"Top {len(data)} runs by {metric}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_iou(df: pd.DataFrame, output_path: Path, primary_metric: str, top_n: int) -> None:
    class_cols = [f"val/iou_class_{i}" for i in range(6) if f"val/iou_class_{i}" in df.columns]
    if not class_cols:
        return

    data = df[["Name", primary_metric, *class_cols]].copy()
    for col in [primary_metric, *class_cols]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=class_cols, how="all")
    if data.empty:
        return

    data = _sort_by_metric(data, primary_metric).head(top_n)

    x = np.arange(len(class_cols))
    width = 0.8 / max(1, len(data))

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, (_, row) in enumerate(data.iterrows()):
        values = [row[col] for col in class_cols]
        offset = (idx - (len(data) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=str(row["Name"]))

    labels = [CLASS_LABELS.get(int(col.rsplit("_", 1)[-1]), col) for col in class_cols]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class validation IoU")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fetch_histories(entity: str, project: str, run_names: list[str], metrics: list[str]) -> dict[str, pd.DataFrame]:
    import wandb

    api = wandb.Api()
    histories = {}
    wanted = set(run_names)
    runs = api.runs(f"{entity}/{project}")

    for run in runs:
        if wanted and run.name not in wanted:
            continue
        keys = ["epoch", "trainer/global_step", *metrics]
        rows = list(run.scan_history(keys=keys))
        if rows:
            histories[run.name] = pd.DataFrame(rows)

    return histories


def plot_history_curves(histories: dict[str, pd.DataFrame], metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    any_data = False

    for run_name, history in histories.items():
        if metric not in history.columns:
            continue
        x_col = "epoch" if "epoch" in history.columns else "trainer/global_step"
        data = history[[x_col, metric]].dropna()
        if data.empty:
            continue
        ax.plot(data[x_col], data[metric], marker="o", linewidth=1.5, label=run_name)
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@click.command()
@click.option("--csv", "csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="W&B CSV export. If omitted, W&B API is used.")
@click.option("--entity", type=str, default=DEFAULT_ENTITY, show_default=True)
@click.option("--project", type=str, default=DEFAULT_PROJECT, show_default=True)
@click.option("--run", "runs", multiple=True, help="Run name. Can be passed many times; comma-separated values are also accepted.")
@click.option("--regex", type=str, default=None, help="Regex filter on run names.")
@click.option("--include_failed/--only_finished", default=True, show_default=True)
@click.option("--primary_metric", type=str, default="val/mean_iou.max", show_default=True)
@click.option("--metric", "metrics", multiple=True, help="Metric to plot as a bar chart. Can be passed many times.")
@click.option("--top_n", type=int, default=16, show_default=True)
@click.option("--history", is_flag=True, default=False, help="Fetch metric histories via W&B API and plot curves.")
@click.option("--output_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("reports/wandb_comparison"), show_default=True)
def main(
    csv_path: Path | None,
    entity: str,
    project: str,
    runs: tuple[str, ...],
    regex: str | None,
    include_failed: bool,
    primary_metric: str,
    metrics: tuple[str, ...],
    top_n: int,
    history: bool,
    output_dir: Path,
) -> None:
    run_names = _split_csv_values(runs)
    metrics_to_plot = _split_csv_values(metrics) or [
        primary_metric,
        "val/loss.min",
        "val/pixel_acc.max",
        "train/mean_iou.max",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_from_csv(csv_path) if csv_path is not None else load_from_wandb(entity, project, run_names)
    df = filter_runs(df, run_names, regex, include_failed)
    df = _sort_by_metric(df, primary_metric)

    summary_csv = output_dir / "run_summary.csv"
    summary_md = output_dir / "run_summary.md"
    df.to_csv(summary_csv, index=False)
    write_markdown_table(
        df,
        summary_md,
        title=f"W&B run comparison sorted by {primary_metric}",
        columns=DEFAULT_SUMMARY_COLUMNS,
    )

    for metric in metrics_to_plot:
        safe_name = metric.replace("/", "_").replace(".", "_")
        plot_metric_bar(df, metric, output_dir / f"bar_{safe_name}.png", top_n=top_n)

    plot_per_class_iou(df, output_dir / "per_class_val_iou.png", primary_metric=primary_metric, top_n=min(top_n, 8))

    if history:
        histories = fetch_histories(entity, project, run_names, metrics_to_plot)
        for metric in metrics_to_plot:
            safe_name = metric.replace("/", "_").replace(".", "_")
            plot_history_curves(histories, metric, output_dir / f"history_{safe_name}.png")

    click.echo(f"Saved summary CSV: {summary_csv}")
    click.echo(f"Saved markdown report: {summary_md}")
    click.echo(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
