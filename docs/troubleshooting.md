# Troubleshooting Guide

## Data Loading Errors

### `FileNotFoundError: Missing required files`
**Cause**: Dataset directory is missing required files.
**Solution**: Ensure these files exist: `X_test.csv`, `y_test.csv`, and `metrics.csv`.

### `ValueError: X and y have different lengths`
**Cause**: Feature matrix and target have different numbers of samples.
**Solution**: Regenerate your data files ensuring row counts match.

### `ValueError: ambiguous task type`
**Cause**: metrics.csv contains both classification and regression indicators.
**Solution**: Use `--task-type classification` or `--task-type regression`.

## Configuration Errors

### `ValueError: Seed must be non-negative`
**Solution**: Use a non-negative integer for `--seed`.

### `ValueError: alpha_grid values must be in [0, 1]`
**Solution**: Alpha values must be between 0 and 1.

### `FileNotFoundError: Dataset path does not exist`
**Solution**: Check the path passed to `--dataset-path`.

## Training Errors

### Run or tests hang on a many-core machine (containers, HPC nodes)

`HistGradientBoosting*` uses OpenMP with one thread per visible core. If the
process is CPU-limited (cgroup/Slurm) but sees many cores, the threads compete
and training can stall. Set `OMP_NUM_THREADS` to the allocated cores, e.g.
`OMP_NUM_THREADS=1 prunestack ...`. Results do not depend on this setting.

### `ModelTrainingError` during base model training
**Possible causes**:
- Incompatible hyperparameters for the algorithm (hyperparameters JSON)
- Data contains values that cause numerical issues
- Insufficient samples for certain algorithms

**Solutions**:
1. Check the hyperparameters JSON (`configs/hyperparameters.json` or the file passed with `--hyperparameters`) for invalid values
2. Enable `--verbose` logging to identify the failing model
3. Use `--strict-params false` to skip unsupported parameters

### Training takes very long
**Solutions**:
1. Reduce `--max-models-per-algo` (e.g., 5 instead of 20)
2. Reduce `--n-folds` (e.g., 3 instead of 5)
3. Use `--skip-optimization` for testing

## Optimization Errors

### `OptimizationError`: Optimization stuck
**Cause**: EHO cannot improve fitness.
**Solutions**:
1. Increase `eho_max_iterations`
2. Increase population size (`eho_n_clans * eho_clan_size`)
3. Try different α-β grid values

### All models have same performance
**Cause**: Base models are too similar.
**Solutions**:
1. Use more diverse algorithms
2. Increase feature subset diversity (`feature_subset_min_ratio`, `feature_subset_max_ratio`)
3. Check if the dataset has enough variance

## Memory Issues

### Out of memory
**Solutions**:
1. Reduce model pool size
2. Process one α-β combination at a time
3. Use a machine with more RAM

## Logging & Debugging

### Enable detailed logging
```bash
prunestack --dataset-path ./data --verbose
```

### Check the log file
The experiment log is saved to `<output_dir>/experiment.log`.

### Run with strict validation
```bash
prunestack --dataset-path ./data --strict
```
This will stop on any data quality warning, helping identify potential issues early.
