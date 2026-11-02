"""Tests for prunestack.cli."""

import argparse
import json

import pytest

from prunestack.cli import _float_list, build_parser, main, parse_args


class TestFloatList:
    def test_basic(self):
        assert _float_list("0.0,0.5,1.0") == [0.0, 0.5, 1.0]

    def test_spaces_and_trailing_comma(self):
        assert _float_list("0, 0.5, 1,") == [0.0, 0.5, 1.0]

    def test_invalid(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _float_list("a,b")


class TestParseArgs:
    def test_minimal_uses_defaults(self, tiny_dataset_dir):
        cfg, args = parse_args(["--dataset-path", str(tiny_dataset_dir)])
        assert cfg.dataset_path == tiny_dataset_dir
        assert cfg.seed == 1985
        assert not args.skip_optimization
        assert (cfg.protocol, cfg.n_outer_folds, cfg.drop_columns) == ("nested", 5, None)

    def test_options_are_mapped_to_config(self, tiny_dataset_dir):
        cfg, _ = parse_args([
            "--dataset-path", str(tiny_dataset_dir), "--seed", "99", "--alpha-grid", "0,0.5,1",
            "--task-type", "classification", "--stacking-method", "ridge",
            "--eho-alpha", "0.3", "--eho-beta", "0.2", "--max-models-per-algo", "4",
        ])
        assert cfg.seed == 99
        assert cfg.alpha_grid == [0.0, 0.5, 1.0]
        assert cfg.task_type == "classification"
        assert cfg.stacking_method == "ridge"
        assert (cfg.eho_alpha, cfg.eho_beta) == (0.3, 0.2)
        assert cfg.max_models_per_algo == 4

    def test_eho_and_normalization_options(self, tiny_dataset_dir):
        cfg, _ = parse_args([
            "--dataset-path", str(tiny_dataset_dir), "--eho-variant", "legacy",
            "--eho-init-inclusion-min", "0.1", "--eho-init-inclusion-max", "0.4",
            "--eho-mutation-rate", "0.02", "--eho-max-evaluations", "3000",
            "--fitness-normalization", "none", "--fitness-reference-subsets", "200",
        ])
        assert cfg.eho_variant == "legacy"
        assert (cfg.eho_init_inclusion_min, cfg.eho_init_inclusion_max) == (0.1, 0.4)
        assert (cfg.eho_mutation_rate, cfg.eho_max_evaluations) == (0.02, 3000)
        assert (cfg.fitness_normalization, cfg.fitness_reference_subsets) == ("none", 200)

    def test_cli_overrides_yaml(self, tiny_dataset_dir, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("seed: 77\nn_folds: 3\n")
        cfg, _ = parse_args(["--dataset-path", str(tiny_dataset_dir), "--config", str(path), "--seed", "1"])
        assert (cfg.seed, cfg.n_folds) == (1, 3)

    def test_unknown_stacking_method_rejected(self, tiny_dataset_dir):
        with pytest.raises(SystemExit):
            parse_args(["--dataset-path", str(tiny_dataset_dir), "--stacking-method", "magic"])

    def test_verbose_and_quiet_are_exclusive(self, tiny_dataset_dir):
        with pytest.raises(SystemExit):
            parse_args(["--dataset-path", str(tiny_dataset_dir), "--verbose", "--quiet"])

    @pytest.mark.parametrize("removed", ["--beta-grid", "--resume", "--full-pipeline", "--use-precomputed"])
    def test_removed_options(self, tiny_dataset_dir, removed):
        with pytest.raises(SystemExit):
            parse_args(["--dataset-path", str(tiny_dataset_dir), removed, "x"])

    def test_every_dest_is_known(self):
        from prunestack.config import DEFAULTS

        run_control = {"help", "version", "dataset_path", "config", "output_dir", "skip_optimization",
                       "skip_stacking", "strict", "verbose", "quiet", "n_runs", "n_jobs"}
        dests = {a.dest for a in build_parser()._actions}
        assert dests - run_control <= set(DEFAULTS)


class TestMain:
    def test_end_to_end_classification(self, classification_dataset_dir, tmp_path):
        out = tmp_path / "out"
        code = main([
            "--dataset-path", str(classification_dataset_dir), "--output-dir", str(out),
            "--max-models-per-algo", "2", "--alpha-grid", "0,1", "--eho-max-iterations", "3",
            "--eho-n-clans", "2", "--eho-clan-size", "3", "--quiet", "--protocol", "in_sample",
        ])
        assert code == 0
        summary = json.loads((out / "experiment_summary.json").read_text())
        assert summary["has_optimization"] and summary["has_stacking"]
        assert (out / "experiment_report.html").exists()
        assert (out / "experiment.log").exists()

    def test_nested_protocol_is_default(self, classification_dataset_dir, tmp_path):
        out = tmp_path / "nested"
        code = main([
            "--dataset-path", str(classification_dataset_dir), "--output-dir", str(out),
            "--max-models-per-algo", "2", "--alpha-grid", "0,1", "--eho-max-iterations", "3",
            "--eho-n-clans", "2", "--eho-clan-size", "3", "--n-outer-folds", "3", "--n-folds", "3",
            "--drop-columns", "feat_5", "--quiet",
        ])
        assert code == 0
        summary = json.loads((out / "summary.json").read_text())
        assert summary["protocol"] == "nested"
        assert summary["dropped_columns"] == {"feat_5": "user"}
        assert (out / "alpha_star.json").exists()

    def test_nested_rejects_skip_options(self, classification_dataset_dir, tmp_path):
        assert main(["--dataset-path", str(classification_dataset_dir), "--output-dir", str(tmp_path),
                     "--skip-stacking", "--quiet"]) == 2

    def test_unknown_drop_column_returns_1(self, classification_dataset_dir, tmp_path):
        assert main(["--dataset-path", str(classification_dataset_dir), "--output-dir", str(tmp_path),
                     "--drop-columns", "nope", "--quiet"]) == 1

    def test_returns_1_on_invalid_dataset(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert main(["--dataset-path", str(tmp_path / "empty"), "--quiet"]) == 1


class TestRepeatedRunsCli:
    def test_n_runs_and_n_jobs(self, classification_dataset_dir, tmp_path):
        out = tmp_path / "out"
        code = main([
            "--dataset-path", str(classification_dataset_dir), "--output-dir", str(out),
            "--max-models-per-algo", "2", "--alpha-grid", "0,1", "--eho-max-iterations", "3",
            "--eho-n-clans", "2", "--eho-clan-size", "3", "--quiet", "--n-runs", "2", "--n-jobs", "2",
            "--protocol", "in_sample",
        ])
        assert code == 0
        for run, seed in ((1, 1), (2, 2)):
            summary = json.loads((out / f"run_{run:02d}" / "experiment_summary.json").read_text())
            assert summary["execution_metadata"]["config"]["optimization_seed"] == seed
        assert (out / "base_models" / "base_models_predictions.csv").exists()

    @pytest.mark.parametrize("option", [["--n-runs", "0"], ["--n-jobs", "0"], ["--n-jobs", "-3"]])
    def test_invalid_values(self, classification_dataset_dir, tmp_path, option):
        assert main(["--dataset-path", str(classification_dataset_dir),
                     "--output-dir", str(tmp_path), "--quiet", *option]) == 2
