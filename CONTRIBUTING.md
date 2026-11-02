# Contributing to PruneStack

Thank you for your interest in contributing to PruneStack!

## Development Setup

```bash
git clone https://github.com/mITz85/prunestack-py.git
cd prunestack-py
pip install -e ".[dev]"
```

## Code Style

- Follow **PEP 8** conventions
- Use **type hints** for all function signatures
- Add **docstrings** (NumPy style) to all public functions and classes
- Maximum line length: 100 characters
- Use `from __future__ import annotations` for modern type annotation syntax

## Testing

### Running Tests

```bash
# Full test suite
python -m pytest tests/ -v --tb=short

# With coverage
python -m pytest tests/ --cov=prunestack --cov-report=term-missing

# Skip slow integration tests
python -m pytest tests/
```

### Writing Tests

- Place tests in `tests/` mirroring the source structure
- Use pytest fixtures for shared setup
- Test edge cases and error conditions
- Aim for >95% coverage

## How to Add New Optimizers

1. Create a new file in `src/prunestack/optimization/`, e.g., `my_optimizer.py`
2. Implement the `OptimizerProtocol`:

```python
from prunestack.optimization.protocol import OptimizerProtocol, OptimizationResult

class MyOptimizer:
    def optimize(self, objective_function, n_models, seed=42) -> OptimizationResult:
        # Your optimization logic here
        ...
```

3. Register it in `VALID_OPTIMIZERS` in `config.py`
4. Add tests in `tests/optimization/`

## How to Add New Metrics

1. Add the metric function to the appropriate module:
   - `src/prunestack/metrics/performance.py` for performance metrics
   - `src/prunestack/metrics/diversity.py` for diversity metrics
2. Register the metric name in `VALID_PERFORMANCE_METRICS` or `VALID_DIVERSITY_METRICS` in `config.py`
3. Add tests

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Submit PR with a clear description of changes

## Commit Messages

Use conventional commit format:
- `feat: add new optimizer`
- `fix: handle empty dataset edge case`
- `docs: update API reference`
- `test: add edge case tests`
- `refactor: simplify fitness calculation`
