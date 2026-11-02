"""In-sample experiment pipeline for a single dataset (``protocol: in_sample``).

Subsets are selected AND evaluated on the same out-of-fold predictions, so the
reported pruned-ensemble scores are optimistically biased. Use it for exploration
and optimizer benchmarks; the manuscript protocol is
:class:`prunestack.experiment.nested.NestedExperiment`.

:class:`ExperimentRunner` executes the phases below in order and writes every
intermediate artefact to ``output_dir``:

1. ``load_data``              load the dataset, the fixed hyperparameters and build the model pool
2. ``train_base_models``      k-fold out-of-fold predictions of every model in the pool
3. ``evaluate_baseline``      performance and diversity of the full ensemble
4. ``optimize_alpha_grid``    P/D normalization bounds, then EHO subset selection for every alpha
5. ``train_stacking``         stacked generalization on the full pool and on each selected subset
6. ``compare_strategies``     majority vote / averaging vs. stacking, full vs. pruned
7. ``extract_meta_features``  dataset, pool and ensemble meta-features (+ separate optimization summary)
8. ``save_results``           CSV/JSON results and an HTML report
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prunestack.config import ExperimentConfig
from prunestack.data.loader import DatasetBundle, load_dataset
from prunestack.data.validation import run_all_validations
from prunestack.ensemble.evaluator import evaluate_baseline
from prunestack.ensemble.stacking import StackedEnsemble
from prunestack.ensemble.strategies import compare_ensemble_strategies
from prunestack.experiment.common import build_optimizer, stacking_settings
from prunestack.io.meta_features import extract_meta_features, extract_optimization_summary
from prunestack.io.results import ExperimentResults, save_results
from prunestack.io.visualization import generate_experiment_report
from prunestack.models.factory import ModelConfig, create_model_pool
from prunestack.models.hyperparameters import load_hyperparameters
from prunestack.metrics.fitness import FitnessScaler, fit_fitness_scaler
from prunestack.models.trainer import train_base_models
from prunestack.optimization.runner import optimize_alpha_grid
from prunestack.utils.exceptions import (
    DataValidationError,
    ModelTrainingError,
    OptimizationError,
    PruneStackError,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run the full pipeline on one dataset.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.
    skip_optimization : bool
        Skip the EHO optimization phase (stacking is then evaluated on the full pool only).
    skip_stacking : bool
        Skip the stacking and strategy-comparison phases.
    output_dir : str, Path or None
        Output directory; ``None`` -> ``<dataset_path>/results``.
    strict : bool
        Treat validation warnings as errors.
    n_jobs : int
        Base models trained in parallel (``-1`` = all cores); does not change the results.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        skip_optimization: bool = False,
        skip_stacking: bool = False,
        output_dir: str | Path | None = None,
        strict: bool = False,
        n_jobs: int = 1,
    ) -> None:
        if n_jobs == 0:
            raise ValueError("n_jobs must be a positive integer or -1 (all cores), got 0")
        self.config = config
        self.n_jobs = n_jobs
        self.skip_optimization = skip_optimization
        self.skip_stacking = skip_stacking
        self.strict = strict
        self.output_dir = Path(output_dir) if output_dir is not None else config.dataset_path / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bundle: DatasetBundle | None = None
        self.hyperparameters: dict[str, dict[str, Any]] = {}
        self.model_pool: list[ModelConfig] = []
        self.predictions_df: pd.DataFrame | None = None
        # Out-of-fold class probabilities (n_samples, n_models, K); classification only.
        self.probabilities: np.ndarray | None = None
        self.base_models_df: pd.DataFrame | None = None
        self.optimization_df: pd.DataFrame | None = None
        self.selected_subsets: dict[float, list[int]] = {}
        self.stacking_df: pd.DataFrame | None = None
        self.strategy_comparison_df: pd.DataFrame | None = None
        self.meta_features: dict[str, Any] = {}
        self.optimization_summary: dict[str, Any] = {}
        self.fitness_scaler: FitnessScaler | None = None
        self.phase_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def task_type(self) -> str:
        return self.bundle.task_type

    @property
    def y_true(self) -> np.ndarray:
        return np.asarray(self.bundle.y)

    @property
    def predictions_matrix(self) -> np.ndarray:
        return self.predictions_df.to_numpy(dtype=float)

    @property
    def stacking_probabilities(self) -> np.ndarray | None:
        """Level-0 probabilities for stacking, or ``None`` when labels are stacked."""
        if self.config.stacking_inputs == "probabilities":
            return self.probabilities
        return None

    def _stacking_settings(self) -> tuple[str, dict[str, Any]]:
        return stacking_settings(self.config, self.task_type)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Phase 1: dataset, fixed hyperparameters, model pool and input validation."""
        self.bundle = load_dataset(
            self.config.dataset_path,
            task_type=self.config.task_type,
            n_folds=self.config.n_folds,
            seed=self.config.seed,
            drop_columns=self.config.drop_columns,
        )
        self.hyperparameters = load_hyperparameters(
            self.task_type, self.config.seed, self.config.hyperparameters_path,
        )
        run_all_validations(
            X=self.bundle.X,
            y=self.bundle.y,
            hyperparameters=self.hyperparameters,
            config=self.config,
            task_type=self.task_type,
            strict=self.strict,
        )
        self.model_pool = create_model_pool(
            n_features=self.bundle.n_features,
            hyperparameters=self.hyperparameters,
            task_type=self.task_type,
            base_seed=self.config.seed,
            max_models_per_algo=self.config.max_models_per_algo,
            feature_subset_min_ratio=self.config.feature_subset_min_ratio,
            feature_subset_max_ratio=self.config.feature_subset_max_ratio,
        )
        logger.info(
            "Loaded %d samples x %d features (%s); model pool: %d models",
            self.bundle.n_samples, self.bundle.n_features, self.task_type, len(self.model_pool),
        )

    def train_base_models(self) -> None:
        """Phase 2: out-of-fold predictions of every model in the pool."""
        self.predictions_df, self.base_models_df, self.probabilities = train_base_models(
            X=self.bundle.X,
            y=self.bundle.y,
            model_pool=self.model_pool,
            folds=self.bundle.folds,
            task_type=self.task_type,
            performance_metric=self.config.performance_metric,
            return_probabilities=True,
            n_jobs=self.n_jobs,
        )
        base_dir = self.output_dir / "base_models"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_df.to_csv(base_dir / "base_models_predictions.csv", index=False)
        if self.probabilities is not None:
            n_classes = self.probabilities.shape[2]
            columns = [f"{name}__p{k}" for name in self.predictions_df.columns for k in range(n_classes)]
            pd.DataFrame(
                self.probabilities.reshape(self.probabilities.shape[0], -1), columns=columns,
            ).to_csv(base_dir / "base_models_probabilities.csv", index=False)
        with open(base_dir / "model_pool.json", "w", encoding="utf-8") as f:
            json.dump([mc.to_dict() for mc in self.model_pool], f, indent=2)
        logger.info("Out-of-fold predictions: %s", self.predictions_df.shape)

    def evaluate_baseline(self) -> None:
        """Phase 3: performance and diversity of the full (unpruned) ensemble."""
        baseline = evaluate_baseline(
            self.y_true,
            self.predictions_matrix,
            self.task_type,
            performance_metric=self.config.performance_metric,
            diversity_metric=self.config.diversity_metric,
        )
        logger.info(
            "Full ensemble: performance=%.4f, diversity=%.4f",
            baseline["ensemble_performance"], baseline["ensemble_diversity"],
        )

    def _get_fitness_scaler(self) -> FitnessScaler:
        """P/D normalization bounds; depend only on the data and ``config.seed``, so they are
        estimated once and shared by all runs and alpha values."""
        if self.fitness_scaler is None:
            self.fitness_scaler = fit_fitness_scaler(
                self.y_true,
                self.predictions_matrix,
                self.task_type,
                performance_metric=self.config.performance_metric,
                diversity_metric=self.config.diversity_metric,
                method=self.config.fitness_normalization,
                n_subsets=self.config.fitness_reference_subsets,
                seed=self.config.seed,
            )
        return self.fitness_scaler

    def optimize_alpha_grid(self) -> None:
        """Phase 4: P/D normalization bounds and EHO subset selection for every alpha."""
        scaler = self._get_fitness_scaler()
        optimizer = build_optimizer(self.config)
        self.optimization_df = optimize_alpha_grid(
            self.y_true,
            self.predictions_matrix,
            self.config.resolved_alpha_grid,
            optimizer=optimizer,
            task_type=self.task_type,
            seed=self.config.resolved_optimization_seed,
            performance_metric=self.config.performance_metric,
            diversity_metric=self.config.diversity_metric,
            scaler=scaler,
        )

        convergence = []
        self.selected_subsets = {}
        for row in self.optimization_df.itertuples(index=False):
            alpha = float(row.alpha)
            self.selected_subsets[alpha] = json.loads(row.best_subset_indices)
            convergence.append({"alpha": alpha, "convergence_history": json.loads(row.convergence_history)})

        opt_dir = self.output_dir / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        with open(opt_dir / "optimization_convergence.json", "w", encoding="utf-8") as f:
            json.dump(convergence, f, indent=2)
        with open(opt_dir / "fitness_normalization.json", "w", encoding="utf-8") as f:
            json.dump(scaler.to_dict(), f, indent=2)
        logger.info("Optimized %d alpha values", len(self.optimization_df))

    def train_stacking(self) -> None:
        """Phase 5: stacked ensemble on the full pool and on every selected subset."""
        meta_learner, params = self._stacking_settings()
        stacker = StackedEnsemble(
            self.predictions_matrix,
            self.y_true,
            self.task_type,
            cv_folds=self.config.n_folds,
            meta_learner=meta_learner,
            meta_learner_params=params,
            seed=self.config.seed,
            base_probabilities=self.stacking_probabilities,
        )
        stacker.train()

        metric = self.config.performance_metric
        rows = [{
            "subset": "full",
            "alpha": None,
            "n_models": stacker.n_models,
            "meta_learner": meta_learner,
            "stacking_inputs": stacker.input_kind,
            "stacking_performance": stacker.evaluate(performance_metric=metric),
        }]
        for alpha, indices in self.selected_subsets.items():
            rows.append({
                "subset": f"alpha_{alpha:.1f}",
                "alpha": alpha,
                "n_models": len(indices),
                "meta_learner": meta_learner,
                "stacking_inputs": stacker.input_kind,
                "stacking_performance": stacker.evaluate(indices, performance_metric=metric),
            })
        self.stacking_df = pd.DataFrame(rows)
        logger.info("Stacking (%s) on full pool: %.4f", meta_learner, rows[0]["stacking_performance"])

    def compare_strategies(self) -> None:
        """Phase 6: combination strategies on the full pool and on the selected subsets."""
        meta_learner, params = self._stacking_settings()
        self.strategy_comparison_df = compare_ensemble_strategies(
            y_true=self.y_true,
            predictions_matrix=self.predictions_matrix,
            optimized_subsets=self.selected_subsets,
            task_type=self.task_type,
            cv_folds=self.config.n_folds,
            seed=self.config.seed,
            performance_metric=self.config.performance_metric,
            diversity_metric=self.config.diversity_metric,
            meta_learner=meta_learner,
            meta_learner_params=params,
            base_probabilities=self.stacking_probabilities,
            scaler=self._get_fitness_scaler(),
        )
        logger.info("Compared %d strategy configurations", len(self.strategy_comparison_df))

    def extract_meta_features(self) -> None:
        """Phase 7: meta-features of the dataset and the model pool; optimization summary kept separate."""
        self.meta_features = extract_meta_features(
            dataset_path=self.config.dataset_path,
            base_models_df=self.predictions_df,
            y_true=self.y_true,
            X=np.asarray(self.bundle.X),
            task_type=self.task_type,
            predictions_matrix=self.predictions_matrix,
            performance_metric=self.config.performance_metric,
            folds=self.bundle.folds,
            seed=self.config.seed,
        )
        # Derived from the optimization outcome -> kept out of the meta-features (leakage).
        self.optimization_summary = (
            extract_optimization_summary(self.optimization_df) if self.optimization_df is not None else {}
        )
        logger.info("Extracted %d meta-features", len(self.meta_features))

    def save_results(self) -> ExperimentResults:
        """Phase 8: write all results and the HTML report."""
        from prunestack import __version__

        results = ExperimentResults(
            dataset_info={
                "dataset_path": str(self.config.dataset_path),
                "task_type": self.task_type,
                "n_samples": self.bundle.n_samples,
                "n_features": self.bundle.n_features,
                "n_folds": len(self.bundle.folds),
                "n_models": self.predictions_df.shape[1],
            },
            base_models_performance=self.base_models_df,
            optimization_results=self.optimization_df,
            stacking_results=self.stacking_df,
            strategy_comparison=self.strategy_comparison_df,
            meta_features=self.meta_features,
            optimization_summary=self.optimization_summary,
            execution_metadata={
                "package_version": __version__,
                "seed": self.config.seed,
                "config": self.config.to_dict(),
                "fitness_normalization": (
                    self.fitness_scaler.to_dict() if self.fitness_scaler is not None else None
                ),
                "phase_times": self.phase_times,
                "total_time": round(sum(self.phase_times.values()), 2),
            },
        )
        save_results(results, self.output_dir)
        generate_experiment_report(results, self.output_dir / "experiment_report.html")
        logger.info("Results written to %s", self.output_dir)
        return results

    def run_repeated_experiment(self, n_runs: int, first_seed: int = 1) -> list[ExperimentResults]:
        """Repeat the stochastic part of the pipeline ``n_runs`` times.

        The base models depend only on ``config.seed`` (CV folds, feature subsets and
        model seeds), so they are trained ONCE. Only the EHO optimizer is stochastic:
        run ``r`` (1-based) uses ``optimization_seed = first_seed + r - 1`` and writes its
        results to ``<output_dir>/run_<r>``; the shared out-of-fold predictions are kept in
        ``<output_dir>/base_models``. Stacking, strategy comparison and meta-features are
        recomputed per run (stacking/comparison depend on the selected subsets).
        """
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        if first_seed < 0:
            raise ValueError(f"first_seed must be non-negative, got {first_seed}")
        logger.info("Dataset: %s | output: %s | %d runs", self.config.dataset_path, self.output_dir, n_runs)

        shared = [p for p in self._phases() if p[0] in ("load_data", "train_base_models", "evaluate_baseline")]
        for name, phase in shared:
            self._run_phase(name, phase)
        shared_times = dict(self.phase_times)

        root_dir, base_config = self.output_dir, self.config
        width = max(2, len(str(n_runs)))
        all_results: list[ExperimentResults] = []
        try:
            for run in range(1, n_runs + 1):
                seed = first_seed + run - 1
                self.config = replace(base_config, optimization_seed=seed)
                self.output_dir = root_dir / f"run_{run:0{width}d}"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.phase_times = dict(shared_times)
                logger.info("=== Run %d/%d (optimization_seed=%d) ===", run, n_runs, seed)
                for name, phase in self._phases():
                    if name in shared_times:
                        continue
                    output = self._run_phase(name, phase)
                    if name == "save_results":
                        all_results.append(output)
        finally:
            self.output_dir, self.config = root_dir, base_config
        return all_results

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _phases(self) -> list[tuple[str, Callable[[], Any]]]:
        phases: list[tuple[str, Callable[[], Any]]] = [
            ("load_data", self.load_data),
            ("train_base_models", self.train_base_models),
            ("evaluate_baseline", self.evaluate_baseline),
        ]
        if not self.skip_optimization:
            phases.append(("optimize_alpha_grid", self.optimize_alpha_grid))
        if not self.skip_stacking:
            phases.append(("train_stacking", self.train_stacking))
            phases.append(("compare_strategies", self.compare_strategies))
        phases.append(("extract_meta_features", self.extract_meta_features))
        phases.append(("save_results", self.save_results))
        return phases

    def _run_phase(self, name: str, phase: Callable[[], Any]) -> Any:
        """Run one phase, record its time and map unexpected errors to PruneStack errors."""
        logger.info("[START] %s", name)
        start = time.perf_counter()
        try:
            output = phase()
        except PruneStackError:
            raise
        except FileNotFoundError as e:
            raise DataValidationError(
                f"Phase '{name}' failed: {e}",
                suggestion="The dataset directory must contain X_test.csv, y_test.csv and metrics.csv.",
            ) from e
        except Exception as e:
            logger.error("[FAIL] %s: %s", name, e)
            if name in ("train_base_models", "train_stacking", "compare_strategies"):
                raise ModelTrainingError(
                    f"Phase '{name}' failed: {e}",
                    suggestion="Check the hyperparameter configuration and the dataset.",
                ) from e
            if name == "optimize_alpha_grid":
                raise OptimizationError(
                    f"Phase '{name}' failed: {e}",
                    suggestion="Check the EHO settings (eho_variant, eho_n_clans, eho_clan_size, "
                               "eho_max_iterations, eho_max_evaluations).",
                ) from e
            raise
        self.phase_times[name] = round(time.perf_counter() - start, 2)
        logger.info("[DONE] %s in %.2fs", name, self.phase_times[name])
        return output

    def run_full_experiment(self) -> ExperimentResults:
        """Run all enabled phases and return the collected results."""
        logger.info("Dataset: %s | output: %s", self.config.dataset_path, self.output_dir)
        results: ExperimentResults | None = None

        for name, phase in self._phases():
            output = self._run_phase(name, phase)
            if name == "save_results":
                results = output

        return results
