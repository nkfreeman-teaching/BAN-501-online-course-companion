# Deep Dive: Time Series Forecasting

*Extends Module 10: Ethics, Deployment & Real-World ML*

!!! note "Supplemental reading"

    Optional unless explicitly assigned in your section. Quiz and assignment content draws from the parent module, not from Deep Dives.


---

## Introduction

This deep dive introduces time series analysis, covering its unique characteristics, standard modeling approaches, and proper validation techniques.

### Why Time Series Is Different

Time series data has a unique property: **temporal ordering matters**. In standard ML, we assume observations are independent—shuffling rows should not matter. In time series, shuffling destroys the information. Yesterday's sales tell you something about today's sales, and January's patterns repeat every January. This changes everything about how we model and validate.

#### Time Series Has Memory

The key insight is that time series data has *memory*—today's value carries information about tomorrow's. This memory comes in two forms. AutoRegressive (AR) memory means tomorrow's value depends on today's value: high sales today probably means high sales tomorrow, because the series "remembers" recent values. Moving Average (MA) memory means the series remembers recent *surprises*: if yesterday's error was +$1000 (sales beat forecast), today might also beat forecast as the underlying conditions persist.

This memory is why ARIMA models can work: they exploit predictable patterns in how values and errors evolve over time. Without memory there is no predictability—just random noise.

---

## Time Series Components

Time series can be decomposed into four components: (i) trend, the long-term direction such as sales growing over years; (ii) seasonality, regular patterns like sales spiking every December; (iii) cyclical fluctuations, irregular longer-term movements driven by economic cycles; and (iv) noise, the random variation that remains.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(
    series,
    model='additive',
    period=12
)
decomposition.plot()
```

Understanding these components helps you choose the right model and spot problems. For example, a series with strong seasonality but no trend calls for a different approach than one with a clear upward trend but no seasonal pattern.

!!! example "Numerical Example: Time Series Decomposition"

    ```python
    import numpy as np

    # 12 months of retail sales
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # True components:
    trend = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]) * 1000
    seasonal = np.array([-5, -8, -3, 0, 2, 3, 2, 1, -1, 0, 5, 10]) * 1000
    noise = np.array([1, 0, 1, 3, 0, -1, 3, 2, -1, 1, -1, -1]) * 1000

    sales = trend + seasonal + noise

    print("Month    Sales     Trend   Seasonal   Noise")
    for i in range(12):
        print(f"{months[i]:<6} ${sales[i]:>7,} ${trend[i]:>7,} ${seasonal[i]:>+7,} ${noise[i]:>+6,}")
    ```

    **Output:**

    ```
    Month    Sales     Trend   Seasonal   Noise
    Jan    $ 46,000 $ 50,000 $  -5,000 $ +1,000
    Feb    $ 43,000 $ 51,000 $  -8,000 $     +0
    Mar    $ 50,000 $ 52,000 $  -3,000 $ +1,000
    ...
    Nov    $ 64,000 $ 60,000 $  +5,000 $ -1,000
    Dec    $ 70,000 $ 61,000 $ +10,000 $ -1,000
    ```

    **Interpretation:** December sales ($70K) aren't just "high"—they're the sum of underlying trend ($61K), seasonal boost (+$10K for holiday shopping), and random noise (-$1K). Decomposition reveals that the $18K swing from February to December is almost entirely seasonal, not growth. For forecasting, extrapolate trend forward, add the expected seasonal effect, and report uncertainty from the noise variance.

    *Source: `computations/deep_dive_timeseries_examples.py` — `demo_time_series_decomposition()`*


---

## ARIMA Models

ARIMA is the classic statistical approach to time series. ARIMA(p, d, q) has three components: the AR (AutoRegressive) component predicts from past values, where p controls how many lags to use; the I (Integrated) component applies differencing for stationarity, where d controls how many times to difference; and the MA (Moving Average) component predicts from past errors, where q controls how many lag errors to use.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(1, 1, 1))
results = model.fit()
forecast = results.forecast(steps=30)
```

To choose p, d, and q, you can examine ACF/PACF plots for guidance, apply AIC/BIC criteria for model selection, or use auto_arima:

```python
from pmdarima import auto_arima

model = auto_arima(
    train_series,
    seasonal=True,
    m=12,  # Monthly seasonality
    trace=True
)
```

Auto_arima searches through parameter combinations and picks the best one, saving you the manual effort of interpreting ACF/PACF plots. This is especially useful when you have many series to forecast and cannot tune each one individually.

!!! example "Numerical Example: ARIMA Parameter Intuition"

    ```python
    import numpy as np
    np.random.seed(42)

    # Generate AR(1) series with different persistence (phi)
    n, shocks = 50, np.random.normal(0, 1, 50)

    def ar1_series(phi, shocks):
        y = np.zeros(len(shocks))
        y[0] = shocks[0]
        for t in range(1, len(shocks)):
            y[t] = phi * y[t-1] + shocks[t]
        return y

    for phi in [0.0, 0.5, 0.9, 0.99]:
        series = ar1_series(phi, shocks)
        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
        print(f"phi={phi:.2f}: variance={np.var(series):.1f}, autocorr={autocorr:.2f}")
    ```

    **Output:**

    ```
    phi=0.00: variance=0.8, autocorr=0.03
    phi=0.50: variance=1.2, autocorr=0.53
    phi=0.90: variance=5.1, autocorr=0.91
    phi=0.99: variance=18.0, autocorr=0.98
    ```

    **Interpretation:** The AR coefficient (phi) controls memory. At phi=0, each value is independent noise—no predictability. At phi=0.9, today strongly predicts tomorrow (autocorr=0.91); shocks persist for many periods. At phi=0.99, the series approaches a random walk—shocks accumulate and variance explodes. When you see high autocorrelation in your data, an AR model can exploit that persistence for forecasting.

    *Source: `computations/deep_dive_timeseries_examples.py` — `demo_arima_parameter_intuition()`*


---

## Prophet

Meta's Prophet is a popular alternative to ARIMA. Prophet offers several advantages: it handles multiple seasonalities automatically, is robust to missing data, produces interpretable components, and makes it easy to add holidays and special events.

```python
from prophet import Prophet

# Data must have columns 'ds' (date) and 'y' (value)
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True
)
model.fit(train_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
model.plot_components(forecast)  # Shows trend, seasonality, etc.
```

Prophet is particularly good for business applications with strong seasonal patterns. Its ability to incorporate holiday effects and handle missing data with minimal tuning makes it a practical default for many forecasting tasks.

### When to Choose What

Prophet is the natural starting point when you need to handle multiple seasonalities, missing data, holidays, or interpretable components. ARIMA is a better fit when you want finer control over model structure, when the series does not match Prophet's assumptions, or when you have very short series. For complex non-linear patterns, multiple input features, or long sequences, LSTM networks offer the most flexibility.

---

## Time Series Validation

You cannot use standard k-fold cross-validation for time series. It would leak future information into training. If your test set contains January 2024 and your training set contains February 2024, you are cheating—you are using the future to predict the past.

Walk-forward validation addresses this problem:
```
Train: [----]          Test: [-]
Train: [------]        Test: [-]
Train: [--------]      Test: [-]
```

The rule is straightforward: always train on the past and test on the future, never the reverse. Each successive fold expands the training window while keeping the test window fixed in time, mirroring how the model will be used in production.

### The Time Machine Rule

Using k-fold cross-validation on time series is like checking tomorrow's newspaper to predict today's stock price. In fold 1, your model trains on data from weeks 3-10 to predict week 1—it uses the future to predict the past. Of course it looks accurate, but in deployment, you will never have next week's data to help predict this week. Walk-forward validation enforces the constraint you will face in production: you can only look backward, never forward.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    train = data[train_idx]
    test = data[test_idx]
    # Train and evaluate
```

!!! example "Numerical Example: Walk-Forward vs K-Fold Validation"

    ```python
    # AR(1) time series: y_t = 0.9 * y_{t-1} + noise
    # 100 time points with strong autocorrelation

    # K-Fold (WRONG): trains on future, tests on past
    # Fold 1: Train on points 20-99, test on 0-19
    # The model "knows" where the series goes!
    kfold_mae = 3.76

    # Walk-Forward (CORRECT): trains on past, tests on future
    # Window 1: Train on 0-49, test on 50-59
    # Window 2: Train on 0-59, test on 60-69
    walk_forward_mae = 3.83

    print(f"K-Fold MAE (biased):      {kfold_mae:.2f}")
    print(f"Walk-Forward MAE:         {walk_forward_mae:.2f}")
    print(f"K-Fold underestimates by: {walk_forward_mae - kfold_mae:.2f}")
    ```

    **Output:**

    ```
    K-Fold MAE (biased):      3.76
    Walk-Forward MAE:         3.83
    K-Fold underestimates by: 0.07
    ```

    **Interpretation:** K-fold appears more accurate because it cheats—using future data to predict past values. In this example, the difference is small (0.07), but in series with trends or structural breaks, k-fold can underestimate error by 20-50%. Always use walk-forward for time series; it reflects the constraint you'll face in production. Note: Exact values may vary slightly depending on random seed and library versions.

    *Source: `computations/deep_dive_timeseries_examples.py` — `demo_walk_forward_vs_kfold()`*


---

## Business Applications

Time series forecasting is everywhere in business. Sales forecasting supports budget planning and resource allocation, while demand planning drives inventory management and supply chain decisions. On the operational side, capacity planning informs staffing and infrastructure needs. Financial forecasting rounds out these applications by producing revenue projections and cash flow estimates that guide strategic decisions.
