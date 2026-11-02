"""Plots and a self-contained HTML report for a single experiment.

All plotting functions return the written path, or ``None`` when matplotlib is
not installed (install the ``plots`` extra) or required columns are missing.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import pandas as pd

from prunestack.io.results import ExperimentResults

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_MATPLOTLIB = False

_DPI = 150


def _save(fig, output_path: Path, description: str) -> Path:
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s to %s", description, output_path)
    return output_path


def _has_columns(df: pd.DataFrame, columns: set[str], plot_name: str) -> bool:
    missing = columns - set(df.columns)
    if missing:
        logger.warning("Cannot draw %s: missing columns %s", plot_name, sorted(missing))
        return False
    return True


def plot_optimization_convergence(
    convergence: list[dict],
    output_path: str | Path,
) -> Path | None:
    """Plot best fitness per iteration, one line per alpha.

    Parameters
    ----------
    convergence : list of dict
        Entries ``{"alpha": float, "convergence_history": list[float]}`` as
        written to ``optimization/optimization_convergence.json``.
    output_path : str or Path
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for entry in convergence:
        history = entry.get("convergence_history") or []
        if history:
            ax.plot(range(len(history)), history, label=f"α={entry['alpha']:.1f}", alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title("EHO convergence per alpha")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    return _save(fig, Path(output_path), "convergence plot")


def plot_alpha_tradeoff(
    optimization_results: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    """Plot best fitness and selected ensemble size as functions of alpha.

    Parameters
    ----------
    optimization_results : DataFrame
        Columns ``alpha``, ``best_fitness`` and ``n_models_selected``.
    output_path : str or Path
    """
    if not HAS_MATPLOTLIB:
        return None
    required = {"alpha", "best_fitness", "n_models_selected"}
    if not _has_columns(optimization_results, required, "alpha trade-off plot"):
        return None

    df = optimization_results.sort_values("alpha")
    fig, ax_fitness = plt.subplots(figsize=(9, 5))
    ax_fitness.plot(df["alpha"], df["best_fitness"], "o-", color="tab:blue", label="Best fitness")
    ax_fitness.set_xlabel("α (performance weight)")
    ax_fitness.set_ylabel("Best fitness", color="tab:blue")
    ax_fitness.grid(True, alpha=0.3)

    ax_size = ax_fitness.twinx()
    ax_size.plot(df["alpha"], df["n_models_selected"], "s--", color="tab:orange", label="Selected models")
    ax_size.set_ylabel("Selected models", color="tab:orange")

    ax_fitness.set_title("Performance–diversity trade-off")
    return _save(fig, Path(output_path), "alpha trade-off plot")


def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    """Bar chart of mean performance per ensemble strategy.

    Parameters
    ----------
    comparison_df : DataFrame
        Columns ``strategy_name`` and ``performance``.
    output_path : str or Path
    """
    if not HAS_MATPLOTLIB:
        return None
    if not _has_columns(comparison_df, {"strategy_name", "performance"}, "strategy comparison"):
        return None

    mean_performance = comparison_df.groupby("strategy_name")["performance"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(mean_performance)), mean_performance.values, color="tab:blue")
    ax.set_xticks(range(len(mean_performance)))
    ax.set_xticklabels(mean_performance.index, rotation=30, ha="right")
    ax.set_ylabel("Performance")
    ax.set_title("Strategy comparison (mean over subsets)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, value in zip(bars, mean_performance.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}",
                ha="center", va="bottom", fontsize=9)
    return _save(fig, Path(output_path), "strategy comparison plot")


def plot_ensemble_size_vs_fitness(
    optimization_results: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    """Scatter plot of selected ensemble size against best fitness, coloured by alpha.

    Parameters
    ----------
    optimization_results : DataFrame
        Columns ``alpha``, ``n_models_selected`` and ``best_fitness``.
    output_path : str or Path
    """
    if not HAS_MATPLOTLIB:
        return None
    required = {"alpha", "n_models_selected", "best_fitness"}
    if not _has_columns(optimization_results, required, "size vs fitness plot"):
        return None

    df = optimization_results
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["n_models_selected"], df["best_fitness"], c=df["alpha"],
                         cmap="viridis", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Selected models")
    ax.set_ylabel("Best fitness")
    ax.set_title("Ensemble size vs fitness")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="α")
    return _save(fig, Path(output_path), "size vs fitness plot")


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, float_format="%.4f", border=0)


def _section(title: str, body: list[str]) -> list[str]:
    return [f"<section><h2>{html.escape(title)}</h2>", *body, "</section>"]


_CSS = """
body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #222; }
h1 { border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }
h2 { margin-top: 28px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
th { background: #1f77b4; color: white; }
tr:nth-child(even) { background: #f5f5f5; }
img { max-width: 100%; margin: 10px 0; border: 1px solid #ddd; }
section { margin: 20px 0; padding: 12px 16px; background: #fafafa; border-radius: 4px; }
"""


def generate_experiment_report(
    results: ExperimentResults,
    output_path: str | Path,
    max_meta_features_shown: int = 20,
) -> Path:
    """Write an HTML report with summary tables and plots.

    Plots are written to ``<report dir>/plots``. The convergence plot is read
    from ``<report dir>/optimization/optimization_convergence.json`` when present.
    """
    output_path = Path(output_path)
    report_dir = output_path.parent
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plots: dict[str, str] = {}

    def add_plot(key: str, path: Path | None) -> None:
        if path is not None:
            plots[key] = f"plots/{path.name}"

    opt = results.optimization_results
    if opt is not None:
        add_plot("tradeoff", plot_alpha_tradeoff(opt, plots_dir / "alpha_tradeoff.png"))
        add_plot("size", plot_ensemble_size_vs_fitness(opt, plots_dir / "size_vs_fitness.png"))
        convergence_path = report_dir / "optimization" / "optimization_convergence.json"
        if convergence_path.exists():
            convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
            add_plot("convergence", plot_optimization_convergence(convergence, plots_dir / "convergence.png"))
    if results.strategy_comparison is not None:
        add_plot("strategy", plot_strategy_comparison(results.strategy_comparison,
                                                      plots_dir / "strategy_comparison.png"))

    def img(key: str, alt: str) -> list[str]:
        return [f"<img src='{plots[key]}' alt='{alt}'>"] if key in plots else []

    parts = ["<!DOCTYPE html>", "<html><head><meta charset='utf-8'>",
             "<title>PruneStack experiment report</title>", f"<style>{_CSS}</style>",
             "</head><body>", "<h1>PruneStack experiment report</h1>"]

    info = results.dataset_info
    parts += _section("Dataset", [
        f"<p><b>{label}:</b> {html.escape(str(info.get(key, 'n/a')))}</p>"
        for label, key in [("Path", "dataset_path"), ("Task", "task_type"), ("Samples", "n_samples"),
                           ("Features", "n_features"), ("Models in pool", "n_models")]
    ])

    if results.strategy_comparison is not None:
        parts += _section("Strategy comparison",
                          [_table(results.strategy_comparison), *img("strategy", "Strategy comparison")])

    if opt is not None:
        columns = [c for c in ["alpha", "best_fitness", "n_models_selected", "n_iterations", "execution_time"]
                   if c in opt.columns]
        parts += _section("Subset optimization", [
            _table(opt[columns]),
            *img("tradeoff", "Alpha trade-off"),
            *img("size", "Ensemble size vs fitness"),
            *img("convergence", "Convergence"),
        ])

    if results.stacking_results is not None:
        parts += _section("Stacking", [_table(results.stacking_results)])

    if results.meta_features:
        top = sorted(results.meta_features.items(), key=lambda kv: abs(kv[1]), reverse=True)
        rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{value:.6g}</td></tr>"
                       for name, value in top[:max_meta_features_shown])
        body = [f"<p><b>Number of meta-features:</b> {len(top)}</p>",
                f"<table><tr><th>Meta-feature (largest |value|)</th><th>Value</th></tr>{rows}</table>"]
        parts += _section("Meta-features", body)

    meta = results.execution_metadata
    body = [f"<p><b>prunestack version:</b> {html.escape(str(meta.get('package_version', 'n/a')))}</p>",
            f"<p><b>Seed:</b> {meta.get('seed', 'n/a')}</p>",
            f"<p><b>Total time:</b> {meta.get('total_time', 0.0):.2f} s</p>"]
    if meta.get("phase_times"):
        rows = "".join(f"<tr><td>{html.escape(phase)}</td><td>{seconds:.2f}</td></tr>"
                       for phase, seconds in meta["phase_times"].items())
        body.append(f"<table><tr><th>Phase</th><th>Time (s)</th></tr>{rows}</table>")
    parts += _section("Execution", body)
    parts.append("</body></html>")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote experiment report to %s", output_path)
    return output_path
