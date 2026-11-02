# Frequently Asked Questions

## General

### What is PruneStack?
PruneStack is a research framework for ensemble pruning — selecting an optimal subset of base models from a larger pool to achieve better or comparable performance with fewer models.

### What is the α-β fitness function?
The fitness function `F(S) = α·Perf(S) + (1-α)·Div(S) - β·|S|/|M|` balances three objectives: maximizing performance (α), maximizing diversity (1-α), and minimizing ensemble size (β).

### Why 120 base models?
6 algorithms × 20 models (1 original model + 19 feature-subset variants) = 120 models, all with fixed hyperparameters (`configs/hyperparameters.json`). The pool is heterogeneous in both inductive bias and input space.

## Data

### What data format does PruneStack expect?
CSV files: `X_test.csv` (features), `y_test.csv` (target), and `metrics.csv` (algorithm metrics, used for task detection). Hyperparameters are fixed and read from `configs/hyperparameters.json`. See [Quick Start](quickstart.md).

### Can I use my own datasets?
Yes. Prepare your data in the expected format. The framework auto-detects classification vs regression from the metrics.csv columns.

### How does task type auto-detection work?
PruneStack looks at column names in metrics.csv. Classification indicators: accuracy, precision, recall, f1, kappa. Regression indicators: mae, mse, rmse, r2.

### What if my dataset is very small?
PruneStack automatically reduces the number of CV folds for small datasets. You'll see a warning but the pipeline will continue.

## Optimization

### How long does EHO optimization take?
It depends on the number of α-β combinations and EHO iterations. A typical 5×5 grid with 50 iterations takes 30-60 seconds.

### Can I add a custom optimizer?
Yes. Implement the `OptimizerProtocol` with an `optimize(objective_fn, n_models, seed)` method.

### What if optimization doesn't converge?
The early stopping mechanism prevents wasted computation. You can increase `eho_max_iterations` or `eho_early_stop_patience`.

## Results

### What are meta-features used for?
Meta-features characterize the dataset-experiment combination. In Layer 2 of the research, they train a meta-learner to predict which ensemble strategy works best for unseen datasets.

### How do I interpret the fitness values?
Higher fitness = better ensemble. Compare across α-β combinations to understand the trade-off between performance, diversity, and parsimony.

### What's the difference between voting and stacking?
**Voting**: Each model's prediction gets equal weight (majority for classification, mean for regression). **Stacking**: A meta-learner is trained to optimally combine predictions.
