"""
Deep Dive: Time Series Forecasting - Numerical Examples

This script demonstrates key concepts from the Time Series Deep Dive
with concrete calculations.
Run with: pixi run python course-companion-computations/deep_dive_timeseries_examples.py

References in Course Companion:
- demo_time_series_decomposition()  -> Deep Dive Time Series, Decomposition section
- demo_arima_parameter_intuition()  -> Deep Dive Time Series, ARIMA section
- demo_walk_forward_vs_kfold()      -> Deep Dive Time Series, Validation section

Last updated: 2026-02-22
"""

import numpy as np


def demo_time_series_decomposition():
    """
    Demonstrate time series decomposition into trend, seasonality, and residual.
    Uses synthetic data to show each component clearly.
    """
    print("=" * 60)
    print("TIME SERIES DECOMPOSITION")
    print("=" * 60)

    np.random.seed(42)

    # Create 3 years of monthly data (36 months)
    n_months = 36
    t = np.arange(n_months)

    # Components:
    # 1. Trend: Linear growth of $1000/month
    trend = 50000 + 1000 * t

    # 2. Seasonality: Monthly pattern (peaks in December, trough in February)
    # Using a simple seasonal pattern
    month_effects = np.array([
        -5000,   # Jan
        -8000,   # Feb (lowest)
        -3000,   # Mar
        0,       # Apr
        2000,    # May
        3000,    # Jun
        2000,    # Jul
        1000,    # Aug
        -1000,   # Sep
        0,       # Oct
        5000,    # Nov
        10000,   # Dec (highest - holiday shopping)
    ])
    seasonality = np.tile(month_effects, 3)  # Repeat for 3 years

    # 3. Noise: Random variation
    noise = np.random.normal(loc=0, scale=2000, size=n_months)

    # Combined series
    sales = trend + seasonality + noise

    print("\nScenario: Monthly retail sales over 3 years")
    print("We'll decompose into: Trend + Seasonality + Residual")
    print()

    # Show first year of data
    print("OBSERVED DATA (Year 1):")
    print("-" * 50)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"{'Month':<6} {'Sales':>12} {'Trend':>12} {'Seasonal':>12} {'Noise':>10}")
    print("-" * 50)
    for i in range(12):
        print(f"{months[i]:<6} ${sales[i]:>10,.0f} ${trend[i]:>10,.0f} "
              f"${seasonality[i]:>+10,.0f} ${noise[i]:>+8,.0f}")

    # Summary statistics
    print("\nCOMPONENT SUMMARY:")
    print("-" * 40)
    print(f"{'Component':<15} {'Mean':>12} {'Std Dev':>12} {'Range':>15}")
    print("-" * 40)
    print(f"{'Trend':<15} ${trend.mean():>10,.0f} ${trend.std():>10,.0f} "
          f"${trend.min():,.0f}-${trend.max():,.0f}")
    print(f"{'Seasonality':<15} ${seasonality.mean():>10,.0f} ${np.std(month_effects):>10,.0f} "
          f"${month_effects.min():+,.0f} to ${month_effects.max():+,.0f}")
    print(f"{'Noise':<15} ${noise.mean():>10,.0f} ${noise.std():>10,.0f} "
          f"${noise.min():+,.0f} to ${noise.max():+,.0f}")

    # Variance decomposition
    total_var = np.var(sales)
    trend_var = np.var(trend)
    seasonal_var = np.var(seasonality)
    noise_var = np.var(noise)

    print("\nVARIANCE DECOMPOSITION:")
    print("-" * 40)
    print(f"Trend explains:       {trend_var / total_var:>6.1%} of total variance")
    print(f"Seasonality explains: {seasonal_var / total_var:>6.1%} of total variance")
    print(f"Noise explains:       {noise_var / total_var:>6.1%} of total variance")

    print("\nKEY INSIGHTS:")
    print("-" * 40)
    print("1. Trend: Sales grow ~$1,000/month (strong upward momentum)")
    print("2. Seasonality: December is $18K above February (plan inventory!)")
    print("3. Noise: ~$2K random variation (irreducible uncertainty)")
    print()
    print("Forecasting strategy:")
    print("  - Extrapolate trend forward")
    print("  - Add expected seasonal effect for target month")
    print("  - Report uncertainty based on noise variance")


def demo_arima_parameter_intuition():
    """
    Show how AR parameter affects time series behavior.
    Demonstrates memory and persistence in AR(1) models.
    """
    print("\n" + "=" * 60)
    print("ARIMA PARAMETER INTUITION")
    print("=" * 60)

    np.random.seed(42)

    n = 50
    shocks = np.random.normal(loc=0, scale=1, size=n)

    print("\nThe AR (AutoRegressive) component: y_t = phi * y_{t-1} + noise")
    print("phi controls how much 'memory' the series has")
    print()

    ar_coefs = [0.0, 0.5, 0.9, 0.99]
    series_dict = {}

    for phi in ar_coefs:
        series = np.zeros(n)
        series[0] = shocks[0]
        for t in range(1, n):
            series[t] = phi * series[t-1] + shocks[t]
        series_dict[phi] = series

    print("EFFECT OF AR COEFFICIENT (phi):")
    print("-" * 60)
    print(f"{'phi':<8} {'Behavior':<25} {'Variance':>10} {'Autocorr(1)':>12}")
    print("-" * 60)

    behaviors = {
        0.0: "White noise (no memory)",
        0.5: "Moderate persistence",
        0.9: "Strong persistence",
        0.99: "Near random walk"
    }

    for phi in ar_coefs:
        series = series_dict[phi]
        var = np.var(series)
        # Lag-1 autocorrelation
        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
        print(f"{phi:<8} {behaviors[phi]:<25} {var:>10.2f} {autocorr:>12.2f}")

    # Show first 10 values for high vs low persistence
    print("\nSAMPLE VALUES (first 10 time points):")
    print("-" * 60)
    print(f"{'t':<4}", end="")
    for phi in [0.0, 0.9]:
        print(f"{'phi=' + str(phi):>12}", end="")
    print(f"{'Shock':>12}")
    print("-" * 60)

    for t in range(10):
        print(f"{t:<4}", end="")
        for phi in [0.0, 0.9]:
            print(f"{series_dict[phi][t]:>12.2f}", end="")
        print(f"{shocks[t]:>12.2f}")

    print("\nINTERPRETATION:")
    print("-" * 60)
    print("phi = 0.0: Each value is just noise. No connection to previous values.")
    print("phi = 0.9: Today's value strongly influenced by yesterday.")
    print("           A shock persists for many periods (slowly decays).")
    print("phi = 0.99: Near 'random walk' - shocks accumulate, series drifts.")
    print()
    print("Practical implication:")
    print("  - High phi: Good for short-term forecasts (momentum continues)")
    print("  - Low phi: Past values don't help much; forecast toward mean")


def demo_walk_forward_vs_kfold():
    """
    Show why k-fold cross-validation leaks information in time series.
    Compares error estimates from both methods.
    """
    print("\n" + "=" * 60)
    print("WALK-FORWARD VS K-FOLD VALIDATION")
    print("=" * 60)

    np.random.seed(42)

    # Generate time series with autocorrelation
    n = 100
    true_values = np.zeros(n)
    true_values[0] = 100

    # AR(1) process: y_t = 0.9 * y_{t-1} + noise
    ar_coef = 0.9
    for i in range(1, n):
        true_values[i] = ar_coef * true_values[i-1] + np.random.normal(0, 5)

    print("\nScenario: Predicting tomorrow's value from today's value")
    print("Data: 100 time points with strong autocorrelation (AR coefficient = 0.9)")
    print()

    # Simple model: predict next value = current value * estimated_coef
    def train_and_predict(train_idx, test_idx, data):
        """Train AR(1) model and return predictions."""
        train = data[train_idx]
        # Estimate AR coefficient from training data
        if len(train) < 2:
            return np.array([])
        y = train[1:]
        x = train[:-1]
        coef = np.sum(x * y) / np.sum(x * x)
        # Predict for test set
        predictions = []
        for idx in test_idx:
            if idx > 0:
                predictions.append(coef * data[idx - 1])
        return np.array(predictions), coef

    # Method 1: K-Fold (WRONG for time series)
    print("METHOD 1: K-Fold Cross-Validation (INCORRECT)")
    print("-" * 45)
    n_folds = 5
    fold_size = n // n_folds
    kfold_errors = []
    kfold_coefs = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_idx = list(range(test_start, test_end))
        train_idx = [i for i in range(n) if i not in test_idx]

        preds, coef = train_and_predict(train_idx, test_idx, true_values)
        actuals = true_values[test_idx]
        # Skip first prediction if it would need data before series start
        valid_mask = np.array(test_idx) > 0
        if valid_mask.sum() > 0:
            mae = np.mean(np.abs(preds - actuals[valid_mask]))
            kfold_errors.append(mae)
            kfold_coefs.append(coef)

    print(f"Average MAE across folds: {np.mean(kfold_errors):.2f}")
    print(f"Problem: Training data includes FUTURE values!")
    print(f"         Fold 1 trains on points 20-99 to predict 0-19")
    print(f"         This is 'peeking' at the future - information leakage")

    # Method 2: Walk-Forward (CORRECT for time series)
    print("\nMETHOD 2: Walk-Forward Validation (CORRECT)")
    print("-" * 45)
    walk_forward_errors = []
    walk_forward_coefs = []

    # Start with at least 50 training points
    min_train = 50
    for test_start in range(min_train, n - 10, 10):
        train_idx = list(range(test_start))
        test_idx = list(range(test_start, min(test_start + 10, n)))

        preds, coef = train_and_predict(train_idx, test_idx, true_values)
        actuals = true_values[test_idx]
        valid_mask = np.array(test_idx) > 0
        if valid_mask.sum() > 0 and len(preds) > 0:
            mae = np.mean(np.abs(preds - actuals[valid_mask]))
            walk_forward_errors.append(mae)
            walk_forward_coefs.append(coef)

    print(f"Average MAE across windows: {np.mean(walk_forward_errors):.2f}")
    print(f"Training always uses PAST data only:")
    print(f"         Window 1: Train on 0-49, test on 50-59")
    print(f"         Window 2: Train on 0-59, test on 60-69")
    print(f"         No information leakage")

    # Compare
    print("\nCOMPARISON:")
    print("-" * 45)
    print(f"{'Method':<25} {'MAE Estimate':>15}")
    print("-" * 45)
    print(f"{'K-Fold (biased)':<25} {np.mean(kfold_errors):>15.2f}")
    print(f"{'Walk-Forward (correct)':<25} {np.mean(walk_forward_errors):>15.2f}")
    print(f"{'Difference':<25} {np.mean(kfold_errors) - np.mean(walk_forward_errors):>+15.2f}")

    print("\nWHY K-FOLD IS OPTIMISTIC:")
    print("-" * 45)
    print("K-fold trains on future data to predict past data.")
    print("The model 'knows' where the series is going, so it appears")
    print("more accurate than it actually is in real deployment.")
    print()
    print("Walk-forward mimics real deployment: you only have past data")
    print("to predict future values. This gives realistic error estimates.")


if __name__ == "__main__":
    demo_time_series_decomposition()
    demo_arima_parameter_intuition()
    demo_walk_forward_vs_kfold()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
