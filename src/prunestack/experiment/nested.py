"""Nested evaluation protocol of the manuscript (``protocol: nested``).

Why nested
----------
If a subset is selected on a set of predictions and its score is reported on the
same predictions, the score is optimistically biased (selection bias), and the
bias grows with the freedom of the search. The nested protocol keeps the rows on
which a subset is *selected* apart from the rows on which it is *evaluated*:

for every outer fold k (stratified k-fold for classification, seed ``seed``)
    T_k = outer training rows, V_k = held-out rows
    1. inner out-of-fold predictions of the 120 models on T_k (inner k-fold on T_k);
    2. every model refitted on all of T_k predicts V_k;
    3. P/D normalization bounds (FitnessScaler) from the inner predictions on T_k;
    4. for every run r and every alpha: EHO selects S on the inner predictions of T_k;
    5. S is evaluated on V_k: majority vote / mean of S, and a stacking meta-learner
       fitted on the inner predictions of S on T_k;
    6. reference strategies on V_k: full pool (vote / stacking) and the best single
       model chosen by its inner score on T_k.

Neither the base models, the scaler, the optimizer, the meta-learner nor the choice
of the best single model ever sees the labels of V_k. The held-out predictions of
the K folds are concatenated, so every strategy gets one cross-validated prediction
per row and the metric (macro-F1 / RMSE) is computed once on all n rows.

Meta-target
-----------
For each run, the held-out score of the pruned voting ensemble is a curve over the
alpha grid. ``alpha_star`` is the arg-max of the mean curve over the runs (the
oracle trade-off of the dataset); the per-run arg-max values, the set of alphas
within one standard error of the maximum and the full curve are saved as well.
Because alpha_star is chosen with the held-out rows it is a *target* for the
meta-analysis, not an unbiased performance estimate; honest comparisons of the
strategies use an alpha fixed without the dataset's own held-out rows
(``scripts/meta_analysis.py``).

Outputs (``output_dir``)
------------------------
``selection.csv``            one row per (run, fold, alpha): subset, inner P/D/fitness, held-out scores
``heldout_performance.csv``  pooled held-out metric per strategy (and run / alpha)
``alpha_curve.csv``          mean / sd over runs of the pooled held-out score per alpha
``alpha_star.json``          meta-targets of the dataset
``meta_features.json``       123 meta-features (pool features from the held-out predictions)
``summary.json``             configuration, dropped columns, timings (written last)
``base_models/``             held-out predictions of every model, model pool
``folds/``                   per-fold checkpoints; an interrupted run resumes from them
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from prunestack.config import ExperimentConfig
from prunestack.data.loader import DatasetBundle, load_dataset
from prunestack.data.validation import run_all_validations
from prunestack.ensemble.stacking import build_meta_learner
from prunestack.experiment.common import build_optimizer, stacking_settings
from prunestack.io.meta_features import extract_meta_features
from prunestack.metrics.diversity import average_diversity
from prunestack.metrics.fitness import FitnessScaler, fit_fitness_scaler
from prunestack.metrics.performance import (
    calculate_metric,
    ensemble_predictions_classification,
    ensemble_predictions_regression,
    normalize_performance_score,
    resolve_metric,
)
from prunestack.models.factory import ModelConfig, create_model_pool
from prunestack.models.hyperparameters import load_hyperparameters
from prunestack.models.trainer import train_base_models
from prunestack.optimization.runner import optimize_alpha_grid

logger = logging.getLogger(__name__)

Folds = list[tuple[np.ndarray, np.ndarray]]
REFERENCE_STRATEGIES = ("full_vote", "full_stacked", "best_single")
PRUNED_STRATEGIES = ("pruned_vote", "pruned_stacked")


@dataclass
class FoldPredictions:
    """Base-model predictions of one outer fold."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_predictions: np.ndarray            # (|T|, M) inner out-of-fold predictions
    test_predictions: np.ndarray             # (|V|, M) predictions of models refitted on T
    inner_probabilities: np.ndarray | None   # (|T|, M, K), classification only
    test_probabilities: np.ndarray | None    # (|V|, M, K), classification only


def inner_folds(y_train: np.ndarray, task_type: str, n_folds: int, seed: int) -> Folds:
    """Inner k-fold split of the outer training rows (positions within T)."""
    if task_type == "classification":
        min_class = int(np.unique(y_train, return_counts=True)[1].min())
        splitter = StratifiedKFold(n_splits=max(2, min(n_folds, min_class)), shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y_train)), y_train))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y_train))))


def alpha_targets(alphas: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Meta-targets from held-out scores of shape (n_runs, n_alphas), higher is better.

    Ties of the mean curve are resolved towards the smaller alpha (``np.argmax``).
    """
    mean = scores.mean(axis=0)
    n_runs = scores.shape[0]
    se = scores.std(axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean)
    best = int(np.argmax(mean))
    per_run = alphas[np.argmax(scores, axis=1)]
    out: dict[str, Any] = {
        "alpha_star": float(alphas[best]),
        "score_at_alpha_star": float(mean[best]),
        "alpha_star_per_run": [float(a) for a in per_run],
        "alpha_star_median_over_runs": float(np.median(per_run)),
        "alphas_within_1se": [float(a) for a in alphas[mean >= mean[best] - se[best]]],
        "curve_range": float(mean.max() - mean.min()),
    }
    if np.any(np.isclose(alphas, 1.0)):
        out["gain_over_alpha_1"] = float(mean[best] - mean[np.isclose(alphas, 1.0)][0])
    return out


class NestedExperiment:
    """Nested pruning experiment on one dataset (see the module docstring).

    Parameters
    ----------
    config : ExperimentConfig
        ``n_outer_folds`` outer folds; ``n_folds`` inner folds on every outer training part.
    output_dir : str, Path or None
        ``None`` -> ``<dataset_path>/results``.
    strict : bool
        Treat validation warnings as errors.
    n_jobs : int
        Base models trained in parallel (``-1`` = all cores); does not change the results.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str | Path | None = None,
        strict: bool = False,
        n_jobs: int = 1,
    ) -> None:
        if n_jobs == 0:
            raise ValueError("n_jobs must be a positive integer or -1 (all cores), got 0")
        self.config = config
        self.strict = strict
        self.n_jobs = n_jobs
        self.output_dir = Path(output_dir) if output_dir is not None else config.dataset_path / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric = ""
        self.bundle: DatasetBundle | None = None
        self.model_pool: list[ModelConfig] = []
        self.phase_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Data and base models
    # ------------------------------------------------------------------

    @property
    def task_type(self) -> str:
        return self.bundle.task_type

    @property
    def X(self) -> np.ndarray:
        return np.asarray(self.bundle.X)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.bundle.y)

    def load_data(self) -> None:
        """Dataset (outer folds), fixed hyperparameters, model pool and input validation."""
        cfg = self.config
        self.bundle = load_dataset(
            cfg.dataset_path, task_type=cfg.task_type, n_folds=cfg.n_outer_folds,
            seed=cfg.seed, drop_columns=cfg.drop_columns,
        )
        hyperparameters = load_hyperparameters(self.task_type, cfg.seed, cfg.hyperparameters_path)
        run_all_validations(X=self.bundle.X, y=self.bundle.y, hyperparameters=hyperparameters,
                            config=cfg, task_type=self.task_type, strict=self.strict)
        self.model_pool = create_model_pool(
            n_features=self.bundle.n_features, hyperparameters=hyperparameters,
            task_type=self.task_type, base_seed=cfg.seed,
            max_models_per_algo=cfg.max_models_per_algo,
            feature_subset_min_ratio=cfg.feature_subset_min_ratio,
            feature_subset_max_ratio=cfg.feature_subset_max_ratio,
        )
        self.metric = resolve_metric(cfg.performance_metric, self.task_type)
        logger.info("Loaded %d samples x %d features (%s); %d outer folds; pool: %d models",
                    self.bundle.n_samples, self.bundle.n_features, self.task_type,
                    len(self.bundle.folds), len(self.model_pool))

    def _train(self, rows: np.ndarray, folds: Folds) -> tuple[np.ndarray, np.ndarray | None]:
        """Predictions (and probabilities) of every model for ``folds`` on ``X[rows]``."""
        preds_df, _, proba = train_base_models(
            X=self.X[rows], y=self.y[rows], model_pool=self.model_pool, folds=folds,
            task_type=self.task_type, performance_metric=self.metric,
            return_probabilities=True, n_jobs=self.n_jobs,
        )
        return preds_df.to_numpy(dtype=float), proba

    def fold_predictions(self, k: int) -> FoldPredictions:
        """Inner out-of-fold and held-out predictions of outer fold ``k`` (cached on disk)."""
        train_idx, test_idx = self.bundle.folds[k]
        path = self.output_dir / "folds" / f"fold_{k}_base_models.npz"
        if path.exists():
            data = np.load(path)
            if np.array_equal(data["train_idx"], train_idx) and data["inner_predictions"].shape[1] == len(self.model_pool):
                logger.info("Fold %d: base-model predictions loaded from %s", k, path)
                return FoldPredictions(
                    train_idx, test_idx, data["inner_predictions"], data["test_predictions"],
                    data["inner_probabilities"] if "inner_probabilities" in data else None,
                    data["test_probabilities"] if "test_probabilities" in data else None,
                )

        folds = inner_folds(self.y[train_idx], self.task_type, self.config.n_folds, self.config.seed)
        inner_pred, inner_proba = self._train(train_idx, folds)
        # One "fold" on the union T + V: fit on T (positions 0..|T|-1), predict V.
        rows = np.concatenate([train_idx, test_idx])
        refit = [(np.arange(len(train_idx)), np.arange(len(train_idx), len(rows)))]
        test_pred, test_proba = self._train(rows, refit)
        test_pred = test_pred[len(train_idx):]
        test_proba = test_proba[len(train_idx):] if test_proba is not None else None

        fp = FoldPredictions(train_idx, test_idx, inner_pred, test_pred, inner_proba, test_proba)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"train_idx": train_idx, "test_idx": test_idx,
                  "inner_predictions": inner_pred, "test_predictions": test_pred}
        if inner_proba is not None:
            arrays |= {"inner_probabilities": inner_proba, "test_probabilities": test_proba}
        np.savez_compressed(path, **arrays)
        return fp

    # ------------------------------------------------------------------
    # Ensembles on the held-out rows
    # ------------------------------------------------------------------

    def _vote(self, predictions: np.ndarray, subset: list[int] | np.ndarray) -> np.ndarray:
        if self.task_type == "classification":
            return ensemble_predictions_classification(predictions, subset)
        return ensemble_predictions_regression(predictions, subset)

    def _stack(self, fp: FoldPredictions, subset: list[int] | np.ndarray) -> np.ndarray:
        """Meta-learner fitted on the inner predictions of ``subset`` on T, applied to V."""
        meta_learner, params = stacking_settings(self.config, self.task_type)
        idx = np.asarray(subset, dtype=int)
        use_proba = (self.task_type == "classification" and fp.inner_probabilities is not None
                     and self.config.stacking_inputs == "probabilities" and meta_learner != "categorical_nb")
        if use_proba:
            train_in, test_in = fp.inner_probabilities[:, idx, :], fp.test_probabilities[:, idx, :]
        else:
            train_in, test_in = fp.inner_predictions[:, idx], fp.test_predictions[:, idx]
        n_classes = int(self.y.max()) + 1 if self.task_type == "classification" else 0
        learner = build_meta_learner(meta_learner, self.task_type, n_classes, params, self.config.seed)
        return learner.fit(train_in, self.y[fp.train_idx]).predict(test_in)

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """Raw metric and its [0, 1] score (macro-F1; RMSE skill 1 - RMSE/sd(y))."""
        if self.task_type == "classification":
            y_pred = np.asarray(y_pred).astype(int)
        raw = calculate_metric(y_true, y_pred, self.metric)
        return raw, normalize_performance_score(raw, self.metric, y_true)

    def _reference_predictions(self, fp: FoldPredictions) -> dict[str, np.ndarray]:
        """Held-out predictions of the full pool and of the best single model."""
        y_train = self.y[fp.train_idx]
        all_models = np.arange(fp.inner_predictions.shape[1])
        inner_scores = [self._score(y_train, fp.inner_predictions[:, m])[1] for m in all_models]
        best = int(np.argmax(inner_scores))
        return {
            "full_vote": self._vote(fp.test_predictions, all_models),
            "full_stacked": self._stack(fp, all_models),
            "best_single": fp.test_predictions[:, best],
        }

    # ------------------------------------------------------------------
    # Pruning on one outer fold
    # ------------------------------------------------------------------

    def _prune_fold(self, k: int, fp: FoldPredictions, seeds: list[int], alphas: list[float],
                    scaler: FitnessScaler) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """EHO selection on T for every run and alpha; held-out predictions on V.

        Returns the selection table and the vote / stacking predictions of shape
        (n_runs, n_alphas, |V|). Cached in ``folds/`` for the same seeds and alphas.
        """
        cache = self.output_dir / "folds" / f"fold_{k}_selection.npz"
        table_path = self.output_dir / "folds" / f"fold_{k}_selection.csv"
        if cache.exists() and table_path.exists():
            data = np.load(cache)
            if list(data["seeds"]) == seeds and np.allclose(data["alphas"], alphas):
                logger.info("Fold %d: selection loaded from %s", k, cache)
                return pd.read_csv(table_path), data["vote"], data["stacked"]

        cfg = self.config
        y_train, y_test = self.y[fp.train_idx], self.y[fp.test_idx]
        optimizer = build_optimizer(cfg)
        vote = np.zeros((len(seeds), len(alphas), len(fp.test_idx)))
        stacked = np.zeros_like(vote)
        rows: list[dict[str, Any]] = []
        for r, seed in enumerate(seeds):
            logger.info("Fold %d/%d, run %d/%d (optimization_seed=%d)",
                        k + 1, len(self.bundle.folds), r + 1, len(seeds), seed)
            grid = optimize_alpha_grid(
                y_train, fp.inner_predictions, alphas, optimizer=optimizer, task_type=self.task_type,
                seed=seed, performance_metric=self.metric, diversity_metric=cfg.diversity_metric,
                scaler=scaler,
            )
            for a, row in enumerate(grid.itertuples(index=False)):
                subset = json.loads(row.best_subset_indices)
                vote[r, a] = self._vote(fp.test_predictions, subset)
                stacked[r, a] = self._stack(fp, subset)
                rows.append({
                    "run": r + 1, "optimization_seed": seed, "fold": k, "alpha": float(row.alpha),
                    "n_models_selected": row.n_models_selected,
                    "subset_indices": row.best_subset_indices,
                    "inner_fitness": row.best_fitness,
                    "inner_performance": row.performance,
                    "inner_performance_scaled": scaler.scale_performance(row.performance_score),
                    "inner_diversity": row.diversity,
                    "inner_diversity_scaled": scaler.scale_diversity(row.diversity),
                    "heldout_vote_performance": self._score(y_test, vote[r, a])[0],
                    "heldout_stacked_performance": self._score(y_test, stacked[r, a])[0],
                    "heldout_diversity": average_diversity(
                        fp.test_predictions, subset, self.task_type, y_test,
                        diversity_metric=cfg.diversity_metric),
                    "n_evaluations": row.n_evaluations,
                    "execution_time": row.execution_time,
                })
        table = pd.DataFrame(rows)
        table.to_csv(table_path, index=False)
        np.savez_compressed(cache, seeds=np.array(seeds), alphas=np.array(alphas), vote=vote, stacked=stacked)
        return table, vote, stacked

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _timed(self, name: str, start: float) -> None:
        self.phase_times[name] = round(self.phase_times.get(name, 0.0) + time.perf_counter() - start, 2)

    def run(self, n_runs: int = 1, first_seed: int = 1) -> dict[str, Any]:
        """Run the protocol with ``n_runs`` optimization seeds ``first_seed, first_seed + 1, ...``."""
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        if first_seed < 0:
            raise ValueError(f"first_seed must be non-negative, got {first_seed}")
        cfg = self.config
        seeds = list(range(first_seed, first_seed + n_runs))
        alphas = cfg.resolved_alpha_grid
        logger.info("Nested protocol | dataset: %s | output: %s | %d runs",
                    cfg.dataset_path, self.output_dir, n_runs)

        start = time.perf_counter()
        self.load_data()
        self._timed("load_data", start)
        n, n_models = self.bundle.n_samples, len(self.model_pool)
        y = self.y

        heldout_models = np.full((n, n_models), np.nan)
        reference = {name: np.zeros(n) for name in REFERENCE_STRATEGIES}
        pooled = {name: np.zeros((n_runs, len(alphas), n)) for name in PRUNED_STRATEGIES}
        tables, scalers = [], []
        for k in range(len(self.bundle.folds)):
            start = time.perf_counter()
            fp = self.fold_predictions(k)
            self._timed("train_base_models", start)
            heldout_models[fp.test_idx] = fp.test_predictions

            start = time.perf_counter()
            scaler = fit_fitness_scaler(
                y[fp.train_idx], fp.inner_predictions, self.task_type,
                performance_metric=self.metric, diversity_metric=cfg.diversity_metric,
                method=cfg.fitness_normalization, n_subsets=cfg.fitness_reference_subsets, seed=cfg.seed,
            )
            scalers.append({"fold": k, **scaler.to_dict()})
            for name, pred in self._reference_predictions(fp).items():
                reference[name][fp.test_idx] = pred
            table, vote, stacked = self._prune_fold(k, fp, seeds, alphas, scaler)
            pooled["pruned_vote"][:, :, fp.test_idx] = vote
            pooled["pruned_stacked"][:, :, fp.test_idx] = stacked
            tables.append(table)
            self._timed("prune_and_evaluate", start)

        start = time.perf_counter()
        selection = pd.concat(tables, ignore_index=True)
        mean_size = selection.groupby(["run", "alpha"])["n_models_selected"].mean()
        records: list[dict[str, Any]] = []
        for name, pred in reference.items():
            raw, score = self._score(y, pred)
            records.append({"strategy": name, "run": None, "alpha": None,
                            "performance": raw, "score": score,
                            "mean_n_models": 1 if name == "best_single" else n_models})
        pruned_scores = {name: np.zeros((n_runs, len(alphas))) for name in PRUNED_STRATEGIES}
        for name, preds in pooled.items():
            for r in range(n_runs):
                for a, alpha in enumerate(alphas):
                    raw, score = self._score(y, preds[r, a])
                    pruned_scores[name][r, a] = score
                    records.append({"strategy": name, "run": r + 1, "alpha": alpha,
                                    "performance": raw, "score": score,
                                    "mean_n_models": float(mean_size.loc[(r + 1, alpha)])})
        heldout = pd.DataFrame(records)

        alpha_arr = np.asarray(alphas, dtype=float)
        curve = pd.DataFrame({"alpha": alpha_arr})
        for name, scores in pruned_scores.items():
            curve[f"{name}_mean"] = scores.mean(axis=0)
            curve[f"{name}_sd"] = scores.std(axis=0, ddof=1) if n_runs > 1 else 0.0
        curve["mean_n_models"] = selection.groupby("alpha")["n_models_selected"].mean().reindex(alphas).to_numpy()
        ref_scores = {name: float(heldout.loc[heldout.strategy == name, "score"].iloc[0])
                      for name in REFERENCE_STRATEGIES}
        targets = {
            "task_type": self.task_type,
            "metric": self.metric,
            "score_definition": "macro-F1" if self.metric == "f1" else f"1 - {self.metric.upper()} / baseline",
            "n_runs": n_runs,
            "n_outer_folds": len(self.bundle.folds),
            "reference_scores": ref_scores,
            "pruned_vote": alpha_targets(alpha_arr, pruned_scores["pruned_vote"]),
            "pruned_stacked": alpha_targets(alpha_arr, pruned_scores["pruned_stacked"]),
        }
        self._timed("aggregate", start)

        start = time.perf_counter()
        meta_features = extract_meta_features(
            dataset_path=cfg.dataset_path, base_models_df=None, y_true=y, X=self.X,
            task_type=self.task_type, predictions_matrix=heldout_models,
            performance_metric=self.metric, folds=self.bundle.folds, seed=cfg.seed,
        )
        self._timed("extract_meta_features", start)

        self._save(selection, heldout, curve, targets, meta_features, heldout_models, scalers, seeds)
        logger.info("alpha* (pruned vote) = %.1f; held-out score %.4f vs full vote %.4f",
                    targets["pruned_vote"]["alpha_star"], targets["pruned_vote"]["score_at_alpha_star"],
                    ref_scores["full_vote"])
        return targets

    def _save(self, selection: pd.DataFrame, heldout: pd.DataFrame, curve: pd.DataFrame,
              targets: dict[str, Any], meta_features: dict[str, float], heldout_models: np.ndarray,
              scalers: list[dict[str, Any]], seeds: list[int]) -> None:
        from prunestack import __version__

        out = self.output_dir
        base_dir = out / "base_models"
        base_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(heldout_models, columns=[mc.name for mc in self.model_pool]).to_csv(
            base_dir / "heldout_predictions.csv", index=False)
        with open(base_dir / "model_pool.json", "w", encoding="utf-8") as f:
            json.dump([mc.to_dict() for mc in self.model_pool], f, indent=2)
        selection.to_csv(out / "selection.csv", index=False)
        heldout.to_csv(out / "heldout_performance.csv", index=False)
        curve.to_csv(out / "alpha_curve.csv", index=False)
        dumps = {
            "alpha_star.json": targets,
            "meta_features.json": meta_features,
            "fitness_normalization.json": scalers,
        }
        for name, obj in dumps.items():
            with open(out / name, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        summary = {
            "package_version": __version__,
            "protocol": "nested",
            "dataset_path": str(self.config.dataset_path),
            "task_type": self.task_type,
            "n_samples": self.bundle.n_samples,
            "n_features": self.bundle.n_features,
            "n_models": len(self.model_pool),
            "dropped_columns": self.bundle.dropped_columns,
            "optimization_seeds": seeds,
            "config": self.config.to_dict(),
            "phase_times": self.phase_times,
            "total_time": round(sum(self.phase_times.values()), 2),
        }
        # Written last: its presence marks a completed dataset (scripts/run_all.sh).
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", out)
