#!/usr/bin/env python3
"""Generate all manuscript figures (Figures 1 to 4) for the JACT submission.

Every figure is written in four formats into OUT_DIR:
  * .eps  vector, preferred by Sage for graphs and line art (separate upload)
  * .tif  raster, grayscale, LZW compressed, TIFF_DPI (separate upload)
  * .pdf  vector, convenient for LaTeX
  * .png  raster at 300 dpi, for embedding in the Word manuscript

Design rules (Sage artwork guidelines, JACT: 300 dpi minimum; Sage line art: 800 dpi):
  * grayscale only, groups are separated by marker shape and line style, not colour;
  * no transparency (EPS cannot store it), light grey is used instead;
  * Liberation Sans (metric-compatible with Arial), 8 pt, fonts embedded as TrueType;
  * widths follow common Sage page geometry (single column 84 mm, full width 174 mm).

All numbers are recomputed from the raw per-dataset result files, with the same
definitions as final21.py, and a short consistency report is printed so that the
figures can be checked against the values quoted in Section 4.

Usage:
  python3 make_figures.py [--out DIR] [--tiff-dpi 800] [--only 1 2 3 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------- configuration
PAPER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PAPER_DIR / "data"
OUT_DIR = PAPER_DIR / "figures"
CLS = [1, 12, 49, 50, 52, 53, 87, 106]
REG = [25, 38, 39, 51, 66, 73, 74, 82, 83, 85, 90, 95, 96]
IDS = CLS + REG
TASK = {i: ("classification" if i in CLS else "regression") for i in IDS}
MARKER = {"classification": "o", "regression": "^"}
FILL = {"classification": "white", "regression": "black"}

MM = 1 / 25.4
SINGLE_COL = 84 * MM
FULL_WIDTH = 174 * MM
PNG_DPI = 300

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------- data loading
def load_results(data_dir: Path = DATA_DIR) -> tuple[dict, dict, dict, dict]:
    """Return per-dataset alpha-curves, reference scores, run-level scores and n."""
    curves, refs, runs, n_rows = {}, {}, {}, {}
    for i in IDS:
        h = pd.read_csv(data_dir / str(i) / "results" / "heldout_performance.csv")
        refs[i] = (h[h.strategy.isin(["full_vote", "full_stacked", "best_single"])]
                   .set_index("strategy").score.to_dict())
        p = h[h.strategy.isin(["pruned_vote", "pruned_stacked"])]
        curves[i] = p.groupby(["strategy", "alpha"]).score.mean().unstack(0)
        runs[i] = p.pivot_table(index=["run", "alpha"], columns="strategy", values="score")
        n_rows[i] = json.loads((data_dir / str(i) / "results" / "summary.json").read_text())["n_samples"]
    return curves, refs, runs, n_rows


def save(fig: plt.Figure, name: str, out: Path, tiff_dpi: int) -> list[Path]:
    """Write one figure in all four formats; the TIFF is converted to 8-bit grayscale."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("eps", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, format=ext)
        paths.append(p)
    png = out / f"{name}.png"
    fig.savefig(png, dpi=PNG_DPI)
    paths.append(png)
    tif = out / f"{name}.tif"
    tmp = out / f".{name}_tmp.png"
    fig.savefig(tmp, dpi=tiff_dpi)
    Image.open(tmp).convert("L").save(tif, compression="tiff_lzw", dpi=(tiff_dpi, tiff_dpi))
    tmp.unlink()
    paths.append(tif)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- Figure 1
def _box(ax, x, y, w, h, text, style="solid", weight="normal", fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec="black", lw=0.7, ls=style))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7,
            fontweight=weight, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, style="solid"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color="black", ls=style, shrinkA=0, shrinkB=0))


def figure1() -> plt.Figure:
    """Nested evaluation protocol (Section 3.6)."""
    fig = plt.figure(figsize=(FULL_WIDTH, 112 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # dataset and outer split
    _box(ax, 0.30, 0.915, 0.40, 0.06, "Dataset (n rows)", weight="bold")
    _arrow(ax, 0.50, 0.915, 0.50, 0.875)
    _box(ax, 0.22, 0.815, 0.56, 0.06,
         "Outer 5-fold split (stratified for classification, seed 1985)\n"
         "for k = 1, ..., 5:  training rows $T_k$,  held-out rows $V_k$")

    # outer fold frame: left = T_k (inner data), right = V_k (held out)
    ax.add_patch(Rectangle((0.02, 0.165), 0.96, 0.625, fill=False, lw=0.7, ls=(0, (4, 2))))
    ax.text(0.04, 0.185, "Repeated for every outer fold k", fontsize=7, style="italic",
            va="bottom")
    ax.plot([0.66, 0.66], [0.18, 0.765], color="black", lw=0.6, ls=(0, (1, 1.5)))
    ax.text(0.30, 0.745, "Uses labels of $T_k$ only", fontsize=7, ha="center",
            fontweight="bold")
    ax.text(0.82, 0.745, "Held-out rows $V_k$", fontsize=7, ha="center",
            fontweight="bold")
    _arrow(ax, 0.55, 0.815, 0.55, 0.712)
    _arrow(ax, 0.60, 0.815, 0.82, 0.715)

    # left column: inner work on T_k
    _box(ax, 0.05, 0.635, 0.56, 0.075,
         "120 base models (6 algorithms × 20 variants), inner 5-fold CV on $T_k$\n"
         "→ out-of-fold predictions; models refitted on all of $T_k$")
    _arrow(ax, 0.33, 0.635, 0.33, 0.6)
    _box(ax, 0.05, 0.535, 0.56, 0.065,
         "Normalisation of P and D: 500 random subsets,\n"
         "1st and 99th percentiles on the inner predictions")
    _arrow(ax, 0.33, 0.535, 0.33, 0.5)
    _box(ax, 0.05, 0.405, 0.56, 0.095,
         "EHO subset search, for every run r = 1, ..., 30 and every\n"
         r"$\alpha \in \{0, 0.1, ..., 1\}$:  maximise "
         r"$F_\alpha(S) = \alpha\,\tilde{P}(S) + (1-\alpha)\,\tilde{D}(S)$" "\n"
         r"on the inner predictions  →  subset $S_{k,r,\alpha}$")
    _arrow(ax, 0.33, 0.405, 0.33, 0.37)
    _box(ax, 0.05, 0.27, 0.56, 0.10,
         "Fitted on inner predictions of $T_k$:\n"
         "stacking meta-learner (NNLS-MLR or NNLS) for $S_{k,r,\\alpha}$\n"
         "and for the full pool; best single model chosen by inner score")

    # right column: evaluation on V_k
    _box(ax, 0.69, 0.555, 0.27, 0.155,
         "Base-model predictions\non $V_k$ (refitted models)")
    _box(ax, 0.69, 0.27, 0.27, 0.255,
         "Evaluated on $V_k$:\n\n"
         "pruned ensemble, vote\npruned ensemble, stacking\n\n"
         "full pool, vote\nfull pool, stacking\nbest single model")
    _arrow(ax, 0.825, 0.555, 0.825, 0.525)
    _arrow(ax, 0.61, 0.32, 0.69, 0.32)
    _arrow(ax, 0.61, 0.45, 0.69, 0.45)

    # after the loop
    _arrow(ax, 0.825, 0.27, 0.825, 0.12)
    _box(ax, 0.42, 0.06, 0.54, 0.06,
         "Held-out predictions of the 5 folds concatenated:\n"
         "one score per strategy, combiner, run and $\\alpha$ (macro-F1 or skill)")
    _arrow(ax, 0.42, 0.09, 0.40, 0.09)
    _box(ax, 0.02, 0.03, 0.38, 0.12,
         "Dataset-level analysis (N = 21):\n"
         r"mean over 30 runs → $\alpha$-curve $\bar{s}_c(\alpha)$;" "\n"
         r"fixed $\alpha$, LODO and $\alpha^*$ comparisons;" "\n"
         "Wilcoxon, $d_z$, W/T/L, Holm", fc="white")
    return fig


# ---------------------------------------------------------------- Figure 2
def figure2(curves, refs, n_rows) -> tuple[plt.Figure, pd.DataFrame]:
    """Per-dataset difference pruned stacked (alpha = 1) minus full pool with vote."""
    df = pd.DataFrame({
        "ds": IDS,
        "task": [TASK[i] for i in IDS],
        "n": [n_rows[i] for i in IDS],
        "diff": [curves[i].loc[1.0, "pruned_stacked"] - refs[i]["full_vote"] for i in IDS],
    }).sort_values("n").reset_index(drop=True)

    # long tick labels widen the tight bounding box, so the axes box is drawn narrower
    fig, ax = plt.subplots(figsize=(SINGLE_COL - 4 * MM, 105 * MM))
    y = np.arange(len(df))
    ax.axvline(0, color="black", lw=0.6)
    for yy in y:
        ax.axhline(yy, color="0.88", lw=0.4, zorder=0)
    for task in ("classification", "regression"):
        m = df.task == task
        ax.scatter(df.loc[m, "diff"], y[m], marker=MARKER[task], s=22, facecolor=FILL[task],
                   edgecolor="black", linewidth=0.7, label=task.capitalize(), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} (n = {n:,})".replace(",", " ") for d, n in zip(df.ds, df.n)])
    ax.set_xlabel("Score difference:\npruned stacking (α = 1) minus full pool with vote")
    ax.set_ylabel("Dataset ID, ordered by number of rows")
    ax.legend(loc="center right", frameon=False, handletextpad=0.3, scatterpoints=1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig, df


# ---------------------------------------------------------------- Figure 3
def figure3(curves) -> tuple[plt.Figure, pd.DataFrame]:
    """alpha-curves relative to alpha = 1, full range (top) and zoom (bottom)."""
    combs = [("pruned_stacked", "Stacking"), ("pruned_vote", "Vote")]
    rel = {c: pd.DataFrame({i: curves[i][c] - curves[i].loc[1.0, c] for i in IDS}) for c, _ in combs}
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 120 * MM))
    zoom = (0.5, 1.0, -0.03, 0.03)
    lo = min(r.values.min() for r in rel.values())
    hi = max(r.values.max() for r in rel.values())
    pad = 0.05 * (hi - lo)
    for col, (c, title) in enumerate(combs):
        r = rel[c]
        for row in range(2):
            ax = axes[row, col]
            ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
            for i in IDS:
                ax.plot(r.index, r[i], color="0.7", lw=0.5,
                        ls="-" if TASK[i] == "regression" else (0, (2, 1)))
            q1, q3 = r.quantile(0.25, axis=1), r.quantile(0.75, axis=1)
            ax.plot(r.index, r.median(axis=1), color="black", lw=1.6, marker="o", ms=2.5)
            ax.plot(r.index, q1, color="black", lw=0.7, ls=(0, (1, 1)))
            ax.plot(r.index, q3, color="black", lw=0.7, ls=(0, (1, 1)))
            if row == 0:
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_title(f"({'ab'[col]}) {title}, full range", loc="left")
                ax.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                                       fill=False, lw=0.6, ls=(0, (1, 1))))
            else:
                ax.set_xlim(zoom[0] - 0.02, zoom[1] + 0.02)
                ax.set_ylim(zoom[2], zoom[3])
                ax.set_title(f"({'cd'[col]}) {title}, zoom on α ≥ 0.5", loc="left")
                ax.set_xlabel("α (weight of performance in the fitness)")
            if col == 0:
                ax.set_ylabel("Score difference to α = 1")
            ax.set_xticks(np.round(np.arange(0, 1.01, 0.1), 1) if row == 0
                          else np.round(np.arange(0.5, 1.01, 0.1), 1))
    handles = [
        plt.Line2D([], [], color="black", lw=1.6, marker="o", ms=2.5, label="Median over datasets"),
        plt.Line2D([], [], color="black", lw=0.7, ls=(0, (1, 1)), label="Interquartile range"),
        plt.Line2D([], [], color="0.6", lw=0.6, ls=(0, (2, 1)), label="Classification dataset"),
        plt.Line2D([], [], color="0.6", lw=0.6, ls="-", label="Regression dataset"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, pd.concat({t: rel[c] for c, t in combs}, axis=1)


# ---------------------------------------------------------------- Figure 4
def figure4(runs) -> tuple[plt.Figure, pd.DataFrame]:
    """Run-level consistency of alpha = 0.8 versus alpha = 1 under stacking."""
    rows = []
    for i in IDS:
        r = runs[i]["pruned_stacked"].unstack(1)
        d = r[0.8] - r[1.0]
        rows.append(dict(ds=i, task=TASK[i], share=(d > 0).mean(), mean=d.mean(), sd=d.std()))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 70 * MM))
    ax.axvline(0.5, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.axhline(0, color="black", lw=0.6)
    for task in ("classification", "regression"):
        m = df.task == task
        ax.scatter(df.loc[m, "share"], df.loc[m, "mean"], marker=MARKER[task], s=22,
                   facecolor=FILL[task], edgecolor="black", linewidth=0.7,
                   label=task.capitalize(), zorder=3)
    for _, row in df.loc[[df.share.idxmin(), df.share.idxmax()]].iterrows():
        ax.annotate(str(row.ds), (row.share, row["mean"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of runs in which α = 0.8 beats α = 1")
    ax.set_ylabel("Mean score difference (α = 0.8 minus α = 1)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, scatterpoints=1)
    return fig, df


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--tiff-dpi", type=int, default=800,
                    help="TIFF resolution; JACT requires >= 300, Sage line art >= 800 (default)")
    ap.add_argument("--only", type=int, nargs="*", default=[1, 2, 3, 4])
    args = ap.parse_args()

    curves, refs, runs, n_rows = load_results(args.data_root)
    written = []
    if 1 in args.only:
        written += save(figure1(), "Figure1", args.out, args.tiff_dpi)
    if 2 in args.only:
        fig, d2 = figure2(curves, refs, n_rows)
        written += save(fig, "Figure2", args.out, args.tiff_dpi)
        w, l = (d2["diff"] > 0.001).sum(), (d2["diff"] < -0.001).sum()
        print(f"[Fig 2] W/T/L = {w}/{len(d2) - w - l}/{l}, median diff = {d2['diff'].median():+.3f}"
              "   (Section 4: 18/0/3, +0.066)")
    if 3 in args.only:
        fig, d3 = figure3(curves)
        written += save(fig, "Figure3", args.out, args.tiff_dpi)
        for t in ("Stacking", "Vote"):
            med = d3[t].median(axis=1)
            print(f"[Fig 3] {t}: median diff at alpha 0 / 0.8 = {med.loc[0.0]:+.3f} / {med.loc[0.8]:+.3f}"
                  f", min = {d3[t].values.min():+.3f}")
        print("        (Section 4: stacking -0.013 / +0.001, vote -0.050 / -0.005)")
    if 4 in args.only:
        fig, d4 = figure4(runs)
        written += save(fig, "Figure4", args.out, args.tiff_dpi)
        print(f"[Fig 4] share range {d4.share.min():.2f} (ds {d4.loc[d4.share.idxmin(), 'ds']}) to "
              f"{d4.share.max():.2f} (ds {d4.loc[d4.share.idxmax(), 'ds']}), median {d4.share.median():.2f}"
              f"; datasets with sd > |mean|: {(d4.sd > d4['mean'].abs()).sum()}/21"
              "   (Section 4: 0.23 (83) to 0.80 (73), median about 0.53)")
    for p in written:
        extra = ""
        if p.suffix == ".tif":
            im = Image.open(p)
            extra = f"  {im.size[0]}x{im.size[1]} px, mode {im.mode}, dpi {im.info.get('dpi')}"
        print(f"wrote {p}{extra}")


if __name__ == "__main__":
    main()
