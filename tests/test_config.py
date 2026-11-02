"""Tests for prunestack.config."""

import pytest

from prunestack.config import DEFAULT_ALPHA_GRID, DEFAULTS, ExperimentConfig, build_config, load_yaml_config


class TestDefaults:
    def test_defaults_match_protocol(self, tiny_dataset_dir):
        cfg = ExperimentConfig(dataset_path=tiny_dataset_dir)
        assert cfg.seed == 1985
        assert cfg.n_folds == 5
        assert cfg.max_models_per_algo == 20
        assert (cfg.feature_subset_min_ratio, cfg.feature_subset_max_ratio) == (0.6, 0.9)
        assert (cfg.eho_n_clans, cfg.eho_clan_size, cfg.eho_max_iterations) == (10, 10, 100)
        assert (cfg.eho_alpha, cfg.eho_beta, cfg.eho_early_stop_patience) == (0.5, 0.1, 25)
        assert cfg.eho_variant == "standard"
        assert (cfg.eho_init_inclusion_min, cfg.eho_init_inclusion_max) == (0.05, 0.5)
        assert cfg.eho_mutation_rate is None and cfg.eho_max_evaluations is None
        assert (cfg.fitness_normalization, cfg.fitness_reference_subsets) == ("minmax", 500)
        assert cfg.stacking_method is None

    def test_default_alpha_grid(self, tiny_dataset_dir):
        cfg = ExperimentConfig(dataset_path=tiny_dataset_dir)
        assert cfg.resolved_alpha_grid == DEFAULT_ALPHA_GRID
        assert DEFAULT_ALPHA_GRID == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def test_optimization_seed_falls_back_to_seed(self, tiny_dataset_dir):
        assert ExperimentConfig(dataset_path=tiny_dataset_dir, seed=7).resolved_optimization_seed == 7
        assert ExperimentConfig(dataset_path=tiny_dataset_dir, optimization_seed=3).resolved_optimization_seed == 3

    def test_to_dict_is_complete_and_serialisable(self, tiny_dataset_dir):
        import json

        d = ExperimentConfig(dataset_path=tiny_dataset_dir).to_dict()
        assert set(d) == {"dataset_path", *DEFAULTS}
        json.dumps(d)


class TestValidation:
    @pytest.mark.parametrize("kwargs, match", [
        ({"seed": -1}, "seed"),
        ({"n_folds": 1}, "n_folds"),
        ({"task_type": "invalid"}, "task_type"),
        ({"performance_metric": "bad"}, "performance_metric"),
        ({"diversity_metric": "bad"}, "diversity_metric"),
        ({"stacking_method": "bad"}, "stacking_method"),
        ({"alpha_grid": [0.0, 1.5]}, "alpha_grid"),
        ({"alpha_grid": []}, "alpha_grid"),
        ({"feature_subset_min_ratio": 0.9, "feature_subset_max_ratio": 0.5}, "must be <"),
        ({"eho_clan_size": 1}, "eho_clan_size"),
        ({"eho_beta": 2.0}, "eho_beta"),
        ({"optimizer": "ga"}, "optimizer"),
        ({"eho_variant": "bad"}, "eho_variant"),
        ({"eho_init_inclusion_min": 0.6, "eho_init_inclusion_max": 0.5}, "eho_init_inclusion"),
        ({"eho_mutation_rate": 1.5}, "eho_mutation_rate"),
        ({"eho_max_evaluations": 0}, "eho_max_evaluations"),
        ({"fitness_normalization": "zscore"}, "fitness_normalization"),
        ({"fitness_reference_subsets": 1}, "fitness_reference_subsets"),
    ])
    def test_invalid_values(self, tiny_dataset_dir, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ExperimentConfig(dataset_path=tiny_dataset_dir, **kwargs)

    def test_nonexistent_dataset_path(self):
        with pytest.raises(FileNotFoundError):
            ExperimentConfig(dataset_path="/nonexistent/path")

    def test_nonexistent_hyperparameters_path(self, tiny_dataset_dir):
        with pytest.raises(FileNotFoundError):
            ExperimentConfig(dataset_path=tiny_dataset_dir, hyperparameters_path="/nope.json")

    def test_valid_stacking_method(self, tiny_dataset_dir):
        assert ExperimentConfig(dataset_path=tiny_dataset_dir, stacking_method="ridge").stacking_method == "ridge"


class TestLoadYamlConfig:
    def test_load_valid_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("seed: 99\nn_folds: 3\n")
        assert load_yaml_config(path) == {"seed": 99, "n_folds": 3}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/config.yaml")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_yaml_config(path)

    def test_non_mapping(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_yaml_config(path)


class TestBuildConfig:
    def test_defaults(self, tiny_dataset_dir):
        assert build_config(dataset_path=tiny_dataset_dir).seed == DEFAULTS["seed"]

    def test_yaml_overrides_defaults(self, tiny_dataset_dir, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("seed: 99\nn_folds: 3\n")
        cfg = build_config(dataset_path=tiny_dataset_dir, config_path=path)
        assert (cfg.seed, cfg.n_folds) == (99, 3)

    def test_keyword_overrides_yaml(self, tiny_dataset_dir, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("seed: 99\n")
        assert build_config(dataset_path=tiny_dataset_dir, config_path=path, seed=123).seed == 123

    def test_none_override_is_ignored(self, tiny_dataset_dir):
        assert build_config(dataset_path=tiny_dataset_dir, seed=None).seed == DEFAULTS["seed"]

    def test_unknown_yaml_key(self, tiny_dataset_dir, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("beta_grid: [0.1]\n")
        with pytest.raises(ValueError, match="Unknown"):
            build_config(dataset_path=tiny_dataset_dir, config_path=path)

    def test_unknown_override(self, tiny_dataset_dir):
        with pytest.raises(ValueError, match="Unknown"):
            build_config(dataset_path=tiny_dataset_dir, use_precomputed=True)
