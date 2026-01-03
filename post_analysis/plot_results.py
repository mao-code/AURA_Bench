#!/usr/bin/env python3
"""Generate AuthBench visualizations for model performance and dataset composition."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BG_COLOR = "#ffffff"
BAR_COLOR = "#4a90e2"
BASELINE_COLOR = "#d62728"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "mathtext.fontset": "stix",
            "font.serif": ["Times New Roman"],
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_reference_grid(ax: plt.Axes, axis: str = "x") -> None:
    """Add dashed reference lines behind bars for easier reading."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, linestyle="--", alpha=0.35, linewidth=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def language_order_from_csv(csv_dir: Path) -> List[str]:
    lang_path = csv_dir / "languages_overall.csv"
    if lang_path.exists():
        df = pd.read_csv(lang_path)
        return df.sort_values("docs", ascending=False)["lang"].tolist()
    return []


def language_order_from_results(results_dir: Path) -> List[str]:
    langs: set[str] = set()
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        if not data:
            continue
        _, payload = next(iter(data.items()))
        block = payload.get("representation") or payload.get("attribution")
        if not block:
            continue
        langs.update(block.get("by_language", {}).keys())
    return sorted(langs)


def make_macro_barh(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    higher_better: bool,
    outpath: Path,
    langs: Sequence[str],
    baseline_value: float | None = None,
    baseline_label: str | None = None,
) -> None:
    d = df.copy()
    d[list(langs)] = d[list(langs)].apply(pd.to_numeric, errors="coerce")
    d["macro_avg"] = d[list(langs)].mean(axis=1)
    d = d.dropna(subset=["macro_avg"])
    if d.empty:
        return
    d = d.sort_values("macro_avg", ascending=not higher_better).reset_index(drop=True)

    plt.rcParams.update({"axes.titlesize": 18, "axes.labelsize": 15})

    n = len(d)
    fig_h = max(10.0, 0.22 * n)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y = np.arange(n)
    bars = ax.barh(y, d["macro_avg"].values, color=BAR_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(d["model"].tolist())
    ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    add_reference_grid(ax, axis="x")

    xmax = float(d["macro_avg"].max())
    if baseline_value is not None:
        xmax = max(xmax, float(baseline_value))
    pad = (xmax - float(d["macro_avg"].min()) + 1e-9) * 0.01
    for i, v in enumerate(d["macro_avg"].values):
        ax.text(v + pad, i, f"{v:.3f}", va="center", ha="left", fontsize=10)

    xmin = min(0.0, float(d["macro_avg"].min()) - 0.02)
    if baseline_value is not None:
        xmin = min(xmin, float(baseline_value) - 0.02)
    ax.set_xlim(xmin, xmax + 0.06)

    if baseline_value is not None:
        ax.axvline(
            baseline_value,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=2,
            label=baseline_label or "baseline",
        )
        ax.legend(loc="lower right")

    fig.tight_layout()
    ensure_dir(outpath)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def collect_performance_dataframe(
    results_dir: Path, metric: str, task: str, langs: Sequence[str]
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    metric_key = "eer" if "eer" in metric.lower() else metric
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        if not data:
            continue
        model, payload = next(iter(data.items()))
        block = payload.get(task)
        if not block:
            continue
        by_lang = block.get("by_language", {})
        row: Dict[str, float | str] = {"model": model}
        for lang in langs:
            row[lang] = by_lang.get(lang, {}).get(metric_key)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_column_for_overall(metric: str, columns: Sequence[str]) -> str | None:
    if metric in columns:
        return metric
    metric_lower = metric.lower()
    if "eer" in metric_lower:
        for candidate in ("eer@10", "eer"):
            if candidate in columns:
                return candidate
    return None


def collect_overall_metrics_dataframe(csv_path: Path, metric: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()
    metric_col = metric_column_for_overall(metric, df.columns)
    if metric_col is None:
        return pd.DataFrame()
    df = df[["model", metric_col]].rename(columns={metric_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def make_overall_barh(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    higher_better: bool,
    outpath: Path,
    baseline_value: float | None = None,
    baseline_label: str | None = None,
) -> None:
    d = df.copy()
    if d.empty:
        return
    d = d.sort_values("value", ascending=not higher_better).reset_index(drop=True)

    plt.rcParams.update({"axes.titlesize": 18, "axes.labelsize": 15})

    n = len(d)
    fig_h = max(10.0, 0.22 * n)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y = np.arange(n)
    ax.barh(y, d["value"].values, color=BAR_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(d["model"].tolist())
    ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    add_reference_grid(ax, axis="x")

    xmax = float(d["value"].max())
    if baseline_value is not None:
        xmax = max(xmax, float(baseline_value))
    pad = (xmax - float(d["value"].min()) + 1e-9) * 0.01
    for i, v in enumerate(d["value"].values):
        ax.text(v + pad, i, f"{v:.3f}", va="center", ha="left", fontsize=10)

    xmin = min(0.0, float(d["value"].min()) - 0.02)
    if baseline_value is not None:
        xmin = min(xmin, float(baseline_value) - 0.02)
    ax.set_xlim(xmin, xmax + 0.06)

    if baseline_value is not None:
        ax.axvline(
            baseline_value,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=2,
            label=baseline_label or "baseline",
        )
        ax.legend(loc="lower right")

    fig.tight_layout()
    ensure_dir(outpath)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def metric_direction(task: str, metric: str, override_higher: bool | None) -> bool:
    if override_higher is not None:
        return override_higher
    metric_lower = metric.lower()
    if task == "attribution" and "eer" in metric_lower:
        return False
    return True


def infer_task(metric: str) -> str:
    metric_lower = metric.lower()
    if "eer" in metric_lower:
        return "attribution"
    return "representation"


def make_genre_pie_grid(
    csv_path: Path,
    outpath: Path,
    langs: Sequence[str],
    top_per_lang: int = 6,
) -> None:
    df = pd.read_csv(csv_path)
    order = [lang for lang in langs if lang in df["lang"].unique()]
    n = len(order)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 6.0, rows * 4.5), subplot_kw={"aspect": "equal"}
    )
    fig.patch.set_facecolor(BG_COLOR)
    cmap = plt.get_cmap("tab20")

    axes = np.array(axes).reshape(rows, cols)
    for idx, lang in enumerate(order):
        ax = axes.flat[idx]
        subset = df[df["lang"] == lang].sort_values("docs", ascending=False)
        top = subset.head(top_per_lang)
        other = subset.iloc[top_per_lang:]
        if not other.empty:
            other_row = pd.DataFrame({"genre": ["Other"], "docs": [other["docs"].sum()]})
            plot_df = pd.concat([top[["genre", "docs"]], other_row], ignore_index=True)
        else:
            plot_df = top[["genre", "docs"]]

        wedges, texts, autotexts = ax.pie(
            plot_df["docs"],
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 1 else "",
            pctdistance=0.72,
            colors=cmap.colors[: len(plot_df)],
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 9},
        )
        for t in autotexts:
            t.set_color("black")
        ax.legend(
            wedges,
            plot_df["genre"],
            title="Genre",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
        )
        ax.set_title(f"{lang} genre mix", fontsize=14, pad=8)
        ax.set_facecolor(BG_COLOR)

    for j in range(n, rows * cols):
        axes.flat[j].axis("off")

    fig.suptitle("Genre distribution within each language", fontsize=18)
    fig.tight_layout()
    ensure_dir(outpath)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def make_language_distribution_bar(csv_path: Path, outpath: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    df = df.sort_values("docs", ascending=False)

    fig_w = max(10.0, 0.9 * len(df))
    fig_h = 6.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(len(df))
    bars = ax.bar(x, df["docs"], color=BAR_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(df["lang"], rotation=30, ha="right")

    ax.set_ylabel("Documents")
    ax.set_xlabel("Language")
    ax.set_title("Language distribution (all docs)")
    add_reference_grid(ax, axis="y")

    xmax = float(df["docs"].max())
    pad = xmax * 0.01 + 1e-6
    for i, v in enumerate(df["docs"]):
        ax.text(i, v + pad, f"{int(v):,}", va="bottom", ha="center", fontsize=10, rotation=0)

    ax.set_ylim(0, xmax * 1.1)
    fig.tight_layout()
    ensure_dir(outpath)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def make_token_length_boxplot(
    csv_path: Path,
    outpath: Path,
    langs: Sequence[str],
) -> None:
    df = pd.read_csv(csv_path)
    df = df[df["lang"].isin(langs)]
    df = df.set_index("lang").loc[list(langs)]

    stats = []
    for lang in langs:
        row = df.loc[lang]
        stats.append(
            {
                "label": lang,
                "whislo": float(row.get("p10", row["min"])),
                "q1": float(row.get("p25", row["min"])),
                "med": float(row.get("p50", row.get("median", row["mean"]))),
                "q3": float(row.get("p75", row.get("max"))),
                "whishi": float(row.get("p90", row.get("max"))),
                "fliers": [],
            }
        )

    fig_h = max(6.0, 0.6 * len(stats))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    box = ax.bxp(stats, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], plt.cm.Blues(np.linspace(0.4, 0.9, len(stats)))):
        patch.set_facecolor(color)
        patch.set_edgecolor("#1f4e79")
        patch.set_linewidth(1.2)
    for line in box["medians"]:
        line.set_color("#0f3057")
        line.set_linewidth(1.5)
    for whisker in box["whiskers"]:
        whisker.set_color("#1f4e79")
        whisker.set_linewidth(1.0)
    for cap in box["caps"]:
        cap.set_color("#1f4e79")
        cap.set_linewidth(1.0)

    ax.set_ylabel("Token length")
    ax.set_xlabel("Language")
    ax.set_title("Token length distribution by language")
    add_reference_grid(ax, axis="y")

    fig.tight_layout()
    ensure_dir(outpath)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AuthBench evaluation and dataset visuals.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory containing per-model evaluation JSON files.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("post_analysis/outputs/csv"),
        help="Directory containing dataset summary CSVs.",
    )
    parser.add_argument(
        "--performance-metric",
        type=str,
        default=None,
        help="Single metric to plot from evaluation results (deprecated if --performance-metrics is set).",
    )
    parser.add_argument(
        "--performance-metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metrics to plot; defaults to recall@10, success@10, ndcg@10, eer.",
    )
    parser.add_argument(
        "--performance-task",
        type=str,
        choices=["representation", "attribution", "auto"],
        default="auto",
        help="Which evaluation task block to read metrics from. 'auto' infers task per metric.",
    )
    parser.add_argument(
        "--higher-better",
        action="store_true",
        help="Force interpret metric as higher-is-better.",
    )
    parser.add_argument(
        "--lower-better",
        action="store_true",
        help="Force interpret metric as lower-is-better.",
    )
    parser.add_argument(
        "--genre-top-k",
        type=int,
        default=6,
        help="Genres to keep per language before merging the rest into 'Other'.",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=None,
        help="Optional output path for the performance bar chart PDF.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="tfidf",
        help="Model name to render as a red reference line instead of a bar.",
    )
    parser.add_argument(
        "--genre-out",
        type=Path,
        default=None,
        help="Optional output path for the genre pie chart PDF.",
    )
    parser.add_argument(
        "--token-out",
        type=Path,
        default=None,
        help="Optional output path for the token-length box plot PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()

    metrics: List[str]
    if args.performance_metrics:
        metrics = args.performance_metrics
    elif args.performance_metric:
        metrics = [args.performance_metric]
    else:
        metrics = ["recall@10", "success@10", "ndcg@10", "eer"]

    custom_single_out = (
        args.performance_out is not None and len(metrics) == 1 and args.performance_out.suffix
    )
    if args.performance_out is not None:
        if args.performance_out.suffix and not custom_single_out:
            perf_dir = args.performance_out.parent
        elif not args.performance_out.suffix:
            perf_dir = args.performance_out
        else:
            perf_dir = args.results_dir / "figures"
    else:
        perf_dir = args.results_dir / "figures"

    genre_out = (
        args.genre_out
        if args.genre_out is not None
        else args.csv_dir.parent / "figures" / "genre_distribution_by_language_pies.pdf"
    )
    token_out = (
        args.token_out
        if args.token_out is not None
        else args.csv_dir.parent / "figures" / "token_length_boxplot_by_language.pdf"
    )

    higher_override = True if args.higher_better else False if args.lower_better else None

    overall_metrics_path = args.results_dir / "overall_metrics.csv"
    use_overall_metrics = overall_metrics_path.exists()

    genre_csv = args.csv_dir / "genre_distribution_by_language.csv"
    token_csv = args.csv_dir / "token_lengths_by_language.csv"
    language_csv = args.csv_dir / "languages_overall.csv"

    need_langs = (
        not use_overall_metrics
        or genre_csv.exists()
        or token_csv.exists()
        or language_csv.exists()
    )
    langs: Sequence[str] = []
    if need_langs:
        langs = language_order_from_csv(args.csv_dir)
        if not langs:
            langs = language_order_from_results(args.results_dir)
        if not langs and not use_overall_metrics:
            raise ValueError("Could not determine language order from CSVs or results.")

    for metric in metrics:
        task = args.performance_task
        if task == "auto":
            task = infer_task(metric)

        if use_overall_metrics:
            perf_df = collect_overall_metrics_dataframe(overall_metrics_path, metric)
        else:
            perf_df = collect_performance_dataframe(args.results_dir, metric, task, langs)
        if perf_df.empty:
            continue

        baseline_value = None
        baseline_label = None
        if args.baseline_model:
            baseline_rows = perf_df[perf_df["model"] == args.baseline_model]
            if not baseline_rows.empty:
                baseline_label = args.baseline_model
                if use_overall_metrics:
                    baseline_value = float(baseline_rows["value"].iloc[0])
                else:
                    baseline_values = baseline_rows[list(langs)].apply(pd.to_numeric, errors="coerce")
                    baseline_value = float(baseline_values.mean(axis=1).iloc[0])
                if np.isnan(baseline_value):
                    baseline_value = None
                    baseline_label = None
                else:
                    perf_df = perf_df[perf_df["model"] != args.baseline_model]

        higher_better = metric_direction(task, metric, higher_override)
        if custom_single_out:
            outpath = args.performance_out
        else:
            outpath = perf_dir / f"{task}_{metric}_macro_bar.pdf"

        if use_overall_metrics:
            make_overall_barh(
                perf_df,
                title=f"{task.title()} performance ({metric})",
                xlabel=metric,
                higher_better=higher_better,
                outpath=outpath,
                baseline_value=baseline_value,
                baseline_label=baseline_label,
            )
        else:
            make_macro_barh(
                perf_df,
                title=f"{task.title()} performance ({metric})",
                xlabel=metric,
                higher_better=higher_better,
                outpath=outpath,
                langs=langs,
                baseline_value=baseline_value,
                baseline_label=baseline_label,
            )

    if genre_csv.exists():
        make_genre_pie_grid(genre_csv, genre_out, langs=langs, top_per_lang=args.genre_top_k)

    if token_csv.exists():
        make_token_length_boxplot(token_csv, token_out, langs=langs)

    if language_csv.exists():
        lang_bar_out = args.csv_dir.parent / "figures" / "language_distribution_bar.pdf"
        make_language_distribution_bar(language_csv, lang_bar_out)


if __name__ == "__main__":
    main()
