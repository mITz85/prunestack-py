"""Command-line interface: ``prunestack --dataset-path DIR [options]``.

Every configuration option defaults to ``None`` so that only the values given on
the command line override the YAML file (``--config``) and ``config.DEFAULTS``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from prunestack.config import DEFAULTS, VALID_DIVERSITY_METRICS, VALID_PERFORMANCE_METRICS, ExperimentConfig, build_config
from prunestack.ensemble.stacking import META_LEARNERS


def _float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats, e.g. ``'0,0.5,1'``."""
    try:
        return [float(x) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated floats, got '{value}'") from e


def _str_list(value: str) -> list[str]:
    """Parse a comma-separated list of column names, e.g. ``'name,bio'``."""
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``prunestack`` command."""
    from prunestack import __version__

    parser = argparse.ArgumentParser(
        prog="prunestack",
        description=(
            "Performance-diversity ensemble pruning with Elephant Herding Optimization "
            "and stacked generalization on a heterogeneous model pool."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    io = parser.add_argument_group("input / output")
    io.add_argument("--dataset-path", required=True, metavar="DIR",
                    help="directory with X_test.csv, y_test.csv and metrics.csv")
    io.add_argument("--config", metavar="YAML", help="YAML file with configuration overrides")
    io.add_argument("--hyperparameters", dest="hyperparameters_path", metavar="JSON",
                    help="fixed base-model hyperparameters and stacking settings "
                         "(default: bundled configs/hyperparameters.json)")
    io.add_argument("--output-dir", metavar="DIR", help="results directory (default: DATASET_PATH/results)")

    exp = parser.add_argument_group("experiment")
    exp.add_argument("--seed", type=int, help=f"random seed (default: {DEFAULTS['seed']})")
    exp.add_argument("--n-folds", type=int, help=f"cross-validation folds (default: {DEFAULTS['n_folds']})")
    exp.add_argument("--task-type", choices=["auto", "classification", "regression"],
                     help="task type (default: auto-detect)")
    exp.add_argument("--protocol", choices=["nested", "in_sample"],
                     help="nested: subsets selected on the outer training part and evaluated on "
                          "the held-out outer fold (manuscript); in_sample: selection and "
                          f"evaluation on the same predictions, exploratory (default: {DEFAULTS['protocol']})")
    exp.add_argument("--n-outer-folds", type=int,
                     help=f"outer folds of the nested protocol (default: {DEFAULTS['n_outer_folds']})")
    exp.add_argument("--drop-columns", type=_str_list, metavar="C1,C2,...",
                     help="feature columns to remove; constant and identifier columns are "
                          "always removed")
    exp.add_argument("--max-models-per-algo", type=int,
                     help="models per algorithm: 1 original + feature-subset variants "
                          f"(default: {DEFAULTS['max_models_per_algo']})")
    exp.add_argument("--feature-subset-min-ratio", type=float,
                     help=f"minimum share of features in a variant (default: {DEFAULTS['feature_subset_min_ratio']})")
    exp.add_argument("--feature-subset-max-ratio", type=float,
                     help=f"maximum share of features in a variant (default: {DEFAULTS['feature_subset_max_ratio']})")
    exp.add_argument("--performance-metric", choices=sorted(VALID_PERFORMANCE_METRICS),
                     help="P(S) in the fitness (default: auto = macro-F1 / RMSE)")
    exp.add_argument("--diversity-metric", choices=sorted(VALID_DIVERSITY_METRICS),
                     help="D(S) in the fitness (default: auto = disagreement / ambiguity)")
    exp.add_argument("--alpha-grid", type=_float_list, metavar="A1,A2,...",
                     help="trade-off weights alpha (default: 0.0,0.1,...,1.0)")
    exp.add_argument("--fitness-normalization", choices=["minmax", "none"],
                     help="rescale P(S) and D(S) to robust [0, 1] bounds from random reference "
                          f"subsets before mixing (default: {DEFAULTS['fitness_normalization']})")
    exp.add_argument("--fitness-reference-subsets", type=int, metavar="N",
                     help="random subsets used to estimate the normalization bounds "
                          f"(default: {DEFAULTS['fitness_reference_subsets']})")
    exp.add_argument("--stacking-method", choices=sorted(META_LEARNERS),
                     help="stacking meta-learner (default: from the hyperparameters JSON, "
                          "nnls_mlr / nnls)")
    exp.add_argument("--stacking-inputs", choices=["probabilities", "labels"],
                     help="level-0 stacking inputs for classification: out-of-fold class "
                          f"probabilities or predicted labels (default: {DEFAULTS['stacking_inputs']})")

    eho = parser.add_argument_group("Elephant Herding Optimization")
    eho.add_argument("--eho-variant", choices=["standard", "legacy"],
                     help="operator set; legacy reproduces PruneStack <= 2.0.0 "
                          f"(default: {DEFAULTS['eho_variant']})")
    eho.add_argument("--eho-n-clans", type=int, help=f"number of clans (default: {DEFAULTS['eho_n_clans']})")
    eho.add_argument("--eho-clan-size", type=int, help=f"elephants per clan (default: {DEFAULTS['eho_clan_size']})")
    eho.add_argument("--eho-max-iterations", type=int,
                     help=f"maximum iterations (default: {DEFAULTS['eho_max_iterations']})")
    eho.add_argument("--eho-alpha", type=float,
                     help=f"matriarch influence alpha_EHO (default: {DEFAULTS['eho_alpha']})")
    eho.add_argument("--eho-beta", type=float,
                     help=f"clan-centre influence beta_EHO (default: {DEFAULTS['eho_beta']})")
    eho.add_argument("--eho-early-stop-patience", type=int,
                     help=f"iterations without improvement before stopping (default: {DEFAULTS['eho_early_stop_patience']})")
    eho.add_argument("--eho-init-inclusion-min", type=float,
                     help="minimum initial inclusion probability per elephant "
                          f"(default: {DEFAULTS['eho_init_inclusion_min']})")
    eho.add_argument("--eho-init-inclusion-max", type=float,
                     help="maximum initial inclusion probability per elephant "
                          f"(default: {DEFAULTS['eho_init_inclusion_max']})")
    eho.add_argument("--eho-mutation-rate", type=float,
                     help="bit-flip probability per bit (default: 1 / number of models)")
    eho.add_argument("--eho-max-evaluations", type=int,
                     help="fitness-evaluation budget per optimization (default: unlimited)")
    eho.add_argument("--optimization-seed", type=int, help="optimizer seed (default: same as --seed)")

    run = parser.add_argument_group("run control")
    run.add_argument("--n-runs", type=int, default=1, metavar="N",
                     help="repeat the EHO optimization N times with optimization seeds "
                          "S, S+1, ..., S+N-1 (S = --optimization-seed, default 1); base models "
                          "are trained once (per outer fold). in_sample: results go to "
                          "OUTPUT_DIR/run_XX (default: 1 = single run)")
    run.add_argument("--n-jobs", type=int, default=1, metavar="J",
                     help="base models trained in parallel, -1 = all cores; results do not "
                          "depend on J (default: 1)")
    run.add_argument("--skip-optimization", action="store_true", help="skip the EHO optimization phase")
    run.add_argument("--skip-stacking", action="store_true", help="skip stacking and strategy comparison")
    run.add_argument("--strict", action="store_true", help="treat validation warnings as errors")
    verbosity = run.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="DEBUG-level console logging")
    verbosity.add_argument("--quiet", action="store_true", help="WARNING-level console logging")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> tuple[ExperimentConfig, argparse.Namespace]:
    """Parse ``argv`` into a validated :class:`ExperimentConfig` and the raw namespace."""
    args = build_parser().parse_args(argv)
    overrides = {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key)}
    config = build_config(args.dataset_path, args.config, **overrides)
    return config, args


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point; returns the process exit code."""
    from prunestack.experiment.nested import NestedExperiment
    from prunestack.experiment.runner import ExperimentRunner
    from prunestack.utils.exceptions import PruneStackError
    from prunestack.utils.logging import setup_logging

    config, args = parse_args(argv)
    if args.n_runs < 1:
        print(f"error: --n-runs must be >= 1, got {args.n_runs}", file=sys.stderr)
        return 2
    if args.n_jobs == 0 or args.n_jobs < -1:
        print(f"error: --n-jobs must be a positive integer or -1, got {args.n_jobs}", file=sys.stderr)
        return 2
    if config.protocol == "nested" and (args.skip_optimization or args.skip_stacking):
        print("error: --skip-optimization / --skip-stacking require --protocol in_sample", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir) if args.output_dir else config.dataset_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=output_dir / "experiment.log")

    first_seed = 1 if config.optimization_seed is None else config.optimization_seed
    try:
        if config.protocol == "nested":
            NestedExperiment(config, output_dir=output_dir, strict=args.strict, n_jobs=args.n_jobs).run(
                args.n_runs, first_seed=first_seed)
            return 0
        runner = ExperimentRunner(
            config,
            skip_optimization=args.skip_optimization,
            skip_stacking=args.skip_stacking,
            output_dir=output_dir,
            strict=args.strict,
            n_jobs=args.n_jobs,
        )
        if args.n_runs > 1:
            runner.run_repeated_experiment(args.n_runs, first_seed=first_seed)
        else:
            runner.run_full_experiment()
    except (PruneStackError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
