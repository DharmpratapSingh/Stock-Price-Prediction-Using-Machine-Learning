# Contributing to Quantitative Trading System

Thank you for your interest in contributing to this quantitative trading system! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## 🤝 Code of Conduct

This project adheres to professional standards of conduct:

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize code quality and testing
- Document your changes thoroughly
- Follow quantitative finance best practices

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Stock-Price-Prediction-Using-Machine-Learning.git
cd Stock-Price-Prediction-Using-Machine-Learning
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies including dev tools
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## 💻 Development Process

### Areas for Contribution

1. **Statistical Methods**
   - New stationarity tests
   - Additional cointegration methods
   - Advanced time series diagnostics

2. **Risk Metrics**
   - Alternative VaR methods (e.g., Cornish-Fisher)
   - Additional risk-adjusted metrics
   - Stress testing frameworks

3. **Portfolio Optimization**
   - Black-Litterman model
   - Robust optimization methods
   - Transaction cost optimization

4. **Factor Models**
   - Additional factor models (e.g., momentum, quality)
   - Custom factor construction
   - Factor timing strategies

5. **Machine Learning**
   - New model architectures
   - Ensemble methods
   - Feature selection techniques

6. **Backtesting**
   - Market microstructure modeling
   - Execution simulation
   - Performance attribution enhancements

## 📝 Coding Standards

### Python Style Guide

Follow PEP 8 with these specifics:

```python
# Use Black for formatting (line length 120)
black src/ tests/ --line-length 120

# Check with Flake8
flake8 src/ tests/ --max-line-length=120

# Type hints for all public functions
def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio with proper documentation.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 0.02)
        periods_per_year: Trading periods per year (default 252)

    Returns:
        Annualized Sharpe ratio

    Raises:
        ValueError: If returns array is empty
    """
    pass
```

### Documentation Standards

All code must be well-documented:

1. **Module Docstrings**
   ```python
   """
   Portfolio Optimization Module

   Implements modern portfolio theory including:
   - Mean-Variance Optimization (Markowitz)
   - Risk Parity
   - Hierarchical Risk Parity (HRP)
   """
   ```

2. **Class Docstrings**
   ```python
   class MeanVarianceOptimizer:
       """
       Mean-Variance Portfolio Optimizer

       Implements Markowitz portfolio optimization to find efficient
       portfolios that maximize return for given risk or minimize risk
       for given return.

       Attributes:
           returns: DataFrame of asset returns
           risk_free_rate: Annual risk-free rate
           periods_per_year: Trading periods per year

       Example:
           >>> optimizer = MeanVarianceOptimizer(returns_df)
           >>> max_sharpe = optimizer.max_sharpe_ratio()
           >>> print(f"Sharpe: {max_sharpe.sharpe_ratio:.4f}")
       """
   ```

3. **Function Docstrings** (Google style)
   ```python
   def calculate_var(
       returns: np.ndarray,
       confidence_level: float = 0.95,
       method: str = 'historical'
   ) -> float:
       """
       Calculate Value at Risk.

       Args:
           returns: Array of portfolio returns
           confidence_level: Confidence level (default 0.95)
           method: Calculation method ('historical', 'parametric', 'monte_carlo')

       Returns:
           Value at Risk as positive percentage

       Raises:
           ValueError: If confidence_level not in (0, 1)
           ValueError: If method is not recognized

       Example:
           >>> returns = np.random.randn(1000) * 0.01
           >>> var_95 = calculate_var(returns, confidence_level=0.95)
           >>> print(f"95% VaR: {var_95:.2%}")
       """
   ```

### Mathematical Formulas

Document formulas using LaTeX in docstrings or comments:

```python
def sharpe_ratio(returns, rf_rate):
    """
    Calculate Sharpe Ratio:

    .. math::
        SR = \\frac{E[R_p - R_f]}{\\sigma_p}

    Where:
        - R_p: Portfolio returns
        - R_f: Risk-free rate
        - σ_p: Portfolio volatility
    """
```

## 🧪 Testing Guidelines

### Test Requirements

All contributions must include tests:

1. **Unit Tests**
   ```python
   # tests/test_risk_metrics.py
   def test_sharpe_ratio_positive_returns():
       """Test Sharpe ratio with positive drift"""
       np.random.seed(42)
       returns = np.random.randn(1000) * 0.01 + 0.0005

       analyzer = RiskAnalyzer(returns)
       sharpe = analyzer.sharpe_ratio()

       assert sharpe > 0, "Sharpe should be positive for positive drift"
       assert 0 < sharpe < 5, "Sharpe should be reasonable"
   ```

2. **Integration Tests**
   ```python
   def test_full_optimization_pipeline():
       """Test complete optimization workflow"""
       # Load data
       returns_df = create_sample_returns()

       # Optimize
       optimizer = MeanVarianceOptimizer(returns_df)
       weights = optimizer.max_sharpe_ratio()

       # Validate
       assert abs(weights.weights.sum() - 1.0) < 1e-6
       assert all(weights.weights >= 0)
       assert weights.sharpe_ratio > 0
   ```

3. **Edge Cases**
   ```python
   def test_var_with_zero_volatility():
       """Test VaR calculation with constant returns"""
       constant_returns = np.ones(100) * 0.01

       var_calc = ValueAtRisk(constant_returns)
       result = var_calc.historical_var()

       assert result.var == 0.0
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_risk_metrics.py -v

# Run specific test
pytest tests/test_risk_metrics.py::test_sharpe_ratio_positive_returns -v
```

### Coverage Requirements

- Minimum coverage: 80% for new code
- All public functions must be tested
- Critical paths (VaR, portfolio optimization) require >90% coverage

## 📚 Documentation

### Code Comments

```python
# Good: Explain WHY, not WHAT
# Adjust for survivorship bias in historical data
adjusted_returns = returns * survivorship_factor

# Bad: Obvious comment
# Calculate mean
mean = np.mean(returns)
```

### README Updates

If your contribution adds features, update the README:

1. Add to "Key Features" section
2. Provide usage example
3. Update methodology section if applicable
4. Add to results section if relevant

### Academic References

For new quantitative methods, cite academic papers:

```python
def black_litterman_optimization(self, views):
    """
    Black-Litterman portfolio optimization.

    References:
        Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
        Financial Analysts Journal, 48(5), 28-43.

        He, G., & Litterman, R. (1999). The Intuition Behind Black-Litterman
        Model Portfolios. Goldman Sachs Investment Management Research.
    """
```

## 📤 Submitting Changes

### Pull Request Process

1. **Ensure Quality**
   ```bash
   # Format code
   black src/ tests/

   # Check linting
   flake8 src/ tests/

   # Run tests
   pytest tests/ -v --cov=src

   # Type checking
   mypy src/
   ```

2. **Commit Messages**

   Follow conventional commits:
   ```
   feat: Add Black-Litterman portfolio optimization
   fix: Correct VaR calculation for small samples
   docs: Update README with new risk metrics
   test: Add tests for cointegration analysis
   refactor: Simplify portfolio optimization interface
   perf: Optimize rolling statistics calculation
   ```

3. **Pull Request Template**

   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] New feature
   - [ ] Bug fix
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Refactoring

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Coverage maintained/improved

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Tests pass locally
   - [ ] No breaking changes (or documented)

   ## Mathematical Validation
   For quantitative features:
   - [ ] Formulas validated against literature
   - [ ] Numerical results verified
   - [ ] Edge cases considered
   ```

4. **Review Process**

   - Maintainers will review within 3-5 business days
   - Address feedback promptly
   - Maintain professional communication
   - Be open to suggestions

## 🔍 Code Review Checklist

Reviewers will check:

- [ ] Code quality and style
- [ ] Test coverage and quality
- [ ] Documentation completeness
- [ ] Mathematical correctness
- [ ] Performance considerations
- [ ] No data leakage in features
- [ ] Proper time-series handling
- [ ] Edge case handling
- [ ] Breaking changes documented

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Load data with...
2. Call function...
3. Observe error...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- Python version:
- OS:
- Package versions:

**Additional Context**
Any other relevant information
```

## 💡 Suggesting Enhancements

### Enhancement Proposal Template

```markdown
**Feature Request**
Clear description of proposed feature

**Motivation**
Why this feature would be valuable

**Proposed Implementation**
High-level approach to implementation

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Academic papers, existing implementations, etc.
```

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an Issue
- **Security**: Email maintainers directly (not public issues)

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Recognized in project documentation

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to making this the best open-source quantitative trading system! 🚀
