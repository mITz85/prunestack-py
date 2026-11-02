# Metrics Reference

## Performance Metrics

### Classification
| Metric | Formula | Range | Description |
|--------|---------|-------|-------------|
| accuracy | correct / total | [0, 1] | Fraction of correct predictions |
| f1 | 2·P·R / (P+R) | [0, 1] | Harmonic mean of precision and recall |
| precision | TP / (TP+FP) | [0, 1] | Positive predictive value |
| recall | TP / (TP+FN) | [0, 1] | True positive rate |
| balanced_accuracy | mean of per-class recall | [0, 1] | Balanced across classes |

### Regression
| Metric | Formula | Range | Description |
|--------|---------|-------|-------------|
| r2 | 1 - SS_res/SS_tot | (-∞, 1] | Coefficient of determination |
| mae | mean(|y - ŷ|) | [0, ∞) | Mean absolute error |
| mse | mean((y - ŷ)²) | [0, ∞) | Mean squared error |
| rmse | √mse | [0, ∞) | Root mean squared error |

## Diversity Metrics

### Classification Diversity

**Disagreement Measure**:
For model pair (i, j):
```
D(i,j) = (N_01 + N_10) / N
```
Where N_01 = samples where i is correct and j is wrong, etc.

**Q-Statistic**:
```
Q(i,j) = (N_11·N_00 - N_01·N_10) / (N_11·N_00 + N_01·N_10)
```
Q ∈ [-1, 1]. Lower Q = more diverse.

**Double-Fault Measure**:
```
DF(i,j) = N_00 / N
```
Fraction of samples where both models fail. Lower = better.

### Regression Diversity

**Prediction Correlation**: Pearson correlation between model predictions.
Lower correlation = more diverse.

**RMSE Difference**: Absolute difference in RMSE between models.

## Fitness Function

The multi-objective fitness function balances performance, diversity, and parsimony:

```
F(S) = α · Performance(S) + (1 - α) · Diversity(S) - β · |S|/|M|
```

### Parameters
- **α** ∈ [0, 1]: Performance weight. α=1 → pure performance, α=0 → pure diversity
- **β** ≥ 0: Size penalty. β=0 → no penalty, higher β → prefer smaller subsets
- **|S|**: Number of selected models
- **|M|**: Total number of models

### Ensemble Performance
For classification: majority vote accuracy of the subset.
For regression: mean prediction performance of the subset.

### Ensemble Diversity
Mean pairwise diversity across all model pairs in the subset.

### Normalization
Both performance and diversity are normalized to [0, 1] before combining to ensure balanced trade-off.
