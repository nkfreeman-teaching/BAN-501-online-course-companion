# Deep Dive: Data Preparation and Feature Engineering

*Extends Module 1: Foundations of Machine Learning*

!!! note "Supplemental reading"

    Optional unless explicitly assigned in your section. Quiz and assignment content draws from the parent module, not from Deep Dives.


---

## Introduction

Data preparation often takes 80% of project time, and it is where projects succeed or fail.

> "The algorithm is not the hard part. Getting the data right is the hard part."

Your job as a business analytics professional is often more about data preparation than algorithm selection. The algorithm is almost a commodity at this point—scikit-learn gives you excellent implementations of everything. What matters is what you feed it. Since algorithms are implemented for us, the higher-leverage skills are problem framing (translating vague business requests into well-defined ML problems), data intuition (recognizing quality issues and predictive features), evaluation rigor (proper validation setup), communication (explaining results to stakeholders), and debugging (diagnosing whether issues are data quality, feature engineering, or methodology). Libraries implement algorithms; they don't tell you which algorithm to use or whether your features make sense.

---

## Common Data Quality Issues

Real-world datasets exhibit several recurring quality problems. Missing values appear as NaN, null, empty strings, or placeholder values like -999 and "N/A." Duplicates—whether exact or near-duplicates—inflate the dataset and distort model training. Outliers present extreme values that may be genuine rare events or data errors. Inconsistent formatting surfaces when the same entity appears differently, such as "USA" versus "United States" versus "US." Data entry errors round out the list, introducing typos, wrong units, and swapped fields.

## Initial Data Exploration

Before you do anything else, explore your data:

```python
import polars as pl

df.shape              # How big is this? (rows, columns)
df.schema             # What are the data types?
df.null_count()       # Where are the missing values?
df.describe()         # Basic statistics
df.is_duplicated().sum()  # Any duplicate rows?
```

Always explore before modeling—do not jump straight into building models without checking basic things like "are there missing values?" A reasonable heuristic is to spend 10-20% of your total project time on EDA before modeling. You've explored "enough" when you can answer: What are the data types and ranges? Where are missing values and what causes them? Are there obvious outliers? What is the target distribution? Which features correlate with the target? The goal is to catch major issues—I've seen projects waste weeks on sophisticated modeling only to discover the target was incorrectly defined. A few hours of EDA would have saved that time.

---

## Handling Missing Data

### Deletion

The simplest strategy is to drop rows with any missing values. This approach is straightforward, but you lose information and might introduce bias if missingness is not random. Deletion works best when values are missing completely at random and only a small percentage of rows are affected.

### Imputation

Imputation fills in missing values rather than discarding rows. The simplest approach—replacing missing values with the mean, median, or mode—is easy to implement but ignores relationships between variables. For time series data, forward or backward filling uses the previous or next value to fill gaps. A more sophisticated option is model-based imputation such as k-NN, which predicts the missing value from other features and thereby captures inter-variable relationships.

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# K-NN imputation (considers relationships)
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

### Why Data is Missing Matters

The reason data is missing determines what you should do about it:

| Type | Description | Implication |
|------|-------------|-------------|
| **MCAR** (Missing Completely at Random) | Missingness has nothing to do with any values | Safe to delete |
| **MAR** (Missing at Random) | Missingness is related to other observed variables | Imputation can work |
| **MNAR** (Missing Not at Random) | Missingness is related to the missing value itself | Problematic—missingness is informative |

A concrete example using the same dataset illustrates the three mechanisms. Imagine an employee survey asking about salary.

In the MCAR case, the survey software randomly crashed for 5% of respondents before they reached the salary question. Missingness has nothing to do with salary, department, or any other variable—purely random technical failure. You can safely delete these rows or impute without bias.

In the MAR case, the survey was optional, and employees in the Sales department (who tend to have higher salaries due to commissions) were more likely to skip the survey entirely because they were busy. Salary is missing, but the missingness is explained by *department* (which you observe). If you control for department, the missing salaries aren't systematically different from observed ones. Imputation works because you can use department to guide it.

In the MNAR case, employees with *very high* salaries (executives) and *very low* salaries (embarrassed about compensation) both skip the salary question. The missingness is directly related to the missing value itself—you can't predict who's missing based on other variables because the salary value itself determines the skip. Imputation will underestimate variance, pulling toward the middle. The missingness is informative—itself a signal.

The key diagnostic question is: "If I knew the missing value, would that help me predict *why* it's missing?" If yes, you likely have MNAR. Diagnosing missingness type requires evidence and domain knowledge. For MCAR, compare distributions of other variables between rows with and without missing values—they should be similar if missingness is random. For MAR vs MNAR, build a model predicting whether a value is missing using other features; if it has predictive power, missingness is at least partially MAR. Domain knowledge is essential: ask why data might be missing and whether that reason relates to the missing value itself.

---

## Feature Scaling

Many algorithms are sensitive to the scale of features. If one feature ranges from 0-1 and another from 0-1,000,000, the larger feature will dominate. The distance problem makes the need for scaling concrete. Imagine finding the nearest neighbor for a customer using age (20-70 years) and income ($30,000-$200,000). Two customers might differ by 5 years in age and $1,000 in income. Without scaling, the distance calculation treats the age difference as 5 units and the income difference as 1,000 units.

The income difference completely dominates—the algorithm essentially ignores age. A 50-year age difference (5 units) matters less than a $50 income difference (50 units). But intuitively, age and income should both influence "similarity."

After standardization (converting to z-scores), both features are measured in "standard deviations from the mean." A 1-standard-deviation difference in age has the same weight as a 1-standard-deviation difference in income. Now the algorithm considers both features fairly.

!!! example "Numerical Example: Feature Scaling Impact on k-NN"

    ```python
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Create synthetic data: Age (20-70) and Income ($30k-$200k)
    np.random.seed(42)
    n_samples = 200
    age = np.random.uniform(low=20, high=70, size=n_samples)
    income = np.random.uniform(low=30000, high=200000, size=n_samples)

    # Target: high income AND middle age (35-55) -> class 1
    target = ((income > 80000) & (age > 35) & (age < 55)).astype(int)
    X = np.column_stack([age, income])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y=target, test_size=0.3, random_state=42
    )

    # Without scaling
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print(f"Without scaling: {knn.score(X_test, y_test):.1%}")

    # With scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    print(f"With scaling: {knn_scaled.score(X_test_scaled, y_test):.1%}")
    ```

    **Output:**

    ```
    Without scaling: 61.7%
    With scaling: 91.7%
    ```

    **Interpretation:** A 30 percentage point improvement from scaling alone. Without scaling, income differences (range: $170,000) completely dominate age differences (range: 50 years). The k-NN algorithm essentially ignores age when finding nearest neighbors, missing the actual pattern in the data.

    *Source: `computations/deep_dive_data_prep_examples.py` — `demo_feature_scaling_impact()`*


Algorithms that need scaling include linear regression, SVM, k-NN, and neural networks, as well as regularized methods (regularization only penalizes fairly when features are scaled).

### Standardization (Z-score)

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

Standardization centers values at 0 with a standard deviation of 1. It preserves outliers and is the preferred choice when you want features centered at zero with comparable scales.

### Min-Max Normalization

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Min-Max normalization scales values to the [0, 1] range. It is sensitive to outliers and works best when you need a bounded range, such as for neural networks.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training parameters!

# Min-Max
minmax = MinMaxScaler()
X_train_scaled = minmax.fit_transform(X_train)
```

Scaling is unnecessary for tree-based methods (Decision Trees, Random Forest, XGBoost), which are scale-invariant. They make splits based on thresholds, not distances, so scaling provides no benefit for tree-based models.

If you're unsure which algorithm you'll use, wait until you've chosen one, or incorporate scaling into your pipeline so it's applied conditionally. Different algorithms prefer different scaling (neural networks work better with MinMax [0,1], while SVMs use standardization). The practical solution is sklearn Pipelines—create a pipeline where scaling is a step before the model, which also prevents data leakage during cross-validation.

---

## Outlier Detection

The appropriate method for detecting outliers depends on the distribution of your data. For normally distributed data, the Z-score method identifies points more than 3 standard deviations from the mean as outliers. For non-normal or unknown distributions, several alternatives are available: the IQR method flags values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR (what box plots use), the Modified Z-score uses median and MAD instead of mean and standard deviation to handle skewed data, and the Isolation Forest is a tree-based method that works well in high dimensions.

```python
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05)  # Expect ~5% outliers
outliers = clf.fit_predict(X)  # Returns -1 for outliers
```

Once outliers are detected, investigate before acting. Are they data errors? Legitimate extreme values? Different populations? Options include removing them, capping or winsorizing, applying a transformation (log), or using robust methods.

A critical distinction is that outliers in features and outliers as targets are completely different problems. If you're detecting fraud, fraudulent transactions are precisely what you're trying to predict—they're your positive class, not outliers to remove. Do not remove outliers blindly based on statistics alone. Investigate: are they errors, rare-but-legitimate events, or the signal you're looking for? For anomaly detection tasks, use algorithms designed to find outliers (Isolation Forest, One-Class SVM), don't remove them.

---

## Encoding Categorical Variables

ML algorithms need numbers, not strings. When you have categorical variables, you need to convert them.

### One-Hot Encoding

One-hot encoding creates a binary column for each category. It is appropriate for nominal categories (those with no inherent order) and a small number of categories. Be cautious with high-cardinality features, because many categories create many columns.

### Label Encoding

Label encoding assigns an integer to each category. It is appropriate for ordinal categories (low/medium/high) where the integer ordering reflects a meaningful sequence. Be cautious when applying label encoding to nominal categories, because it implies an ordering where none exists.

### Target Encoding

Target encoding replaces each category with the mean of the target variable for that category. It is appropriate for high-cardinality categories (ZIP codes, product SKUs) where one-hot encoding would produce too many columns. Be cautious about data leakage—target encoding must use cross-validation to avoid letting each row's outcome influence its own encoded feature value.

Target encoding causes leakage because when you calculate the mean target value for a category, you're using information from rows you'll later predict—each row's outcome influences its own encoded feature value. The severity scales with category size (worse for rare categories). The solution is cross-validation-style encoding: for each row, calculate the category mean using only the other rows. Libraries like `category_encoders` implement this properly.

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['category_column']])

# With polars (simpler for exploration)
X_encoded = df.to_dummies(columns=['category_column'])
```

---

## The Cardinal Rule: Preventing Data Leakage

!!! warning

    Never let information from the test set influence training!

This is called **data leakage**, and it will give you overly optimistic results that don't hold up in production. Leakage is dangerous because its effects are invisible. Imagine you scale your entire dataset before splitting. The scaler computes mean=50 and std=10 from all 1000 rows—including the 200 test rows. Now when you standardize the test data, each test value is positioned relative to statistics that *include itself*. The model indirectly "knows" something about test examples because they influenced preprocessing. In production, new data won't have this privilege—it gets scaled using only training statistics—so your test performance is artificially inflated.

The information flow problem is illustrated in the following diagram.
```
WRONG: Data → Scale ALL → Split → Train → Evaluate
       ↑_______↓
       Test data statistics leak into training

RIGHT: Data → Split → Scale TRAIN → Train
                    → Scale TEST (using train params) → Evaluate
       No leakage: test data never influences anything before evaluation
```

Even small leaks compound. If your pipeline has scaling, then feature selection, then imputation—and each step uses all data—you have three sources of leakage, and the resulting performance estimate can be wildly optimistic. Common leakage examples include (i) scaling before splitting, where the scaler sees test data statistics, (ii) feature selection on all data, where test data influences feature choice, (iii) target encoding without proper cross-validation, where test data target values leak, and (iv) time series ordering violations, where future information is used for past predictions.

!!! example "Numerical Example: Data Leakage Effect"

    ```python
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Small dataset where leakage effect is visible
    wrong_accuracies, right_accuracies = [], []

    for trial in range(20):
        X, y = make_classification(
            n_samples=50, n_features=15, n_informative=5,
            n_redundant=3, random_state=trial
        )

        # WRONG: Scale ALL data, then split
        scaler_wrong = StandardScaler()
        X_scaled = scaler_wrong.fit_transform(X)  # Leakage!
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.3, random_state=trial
        )
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_tr, y_tr)
        wrong_accuracies.append(model.score(X_te, y_te))

        # RIGHT: Split first, then scale
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=trial
        )
        scaler_right = StandardScaler()
        X_tr_scaled = scaler_right.fit_transform(X_tr)
        X_te_scaled = scaler_right.transform(X_te)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_tr_scaled, y_tr)
        right_accuracies.append(model.score(X_te_scaled, y_te))

    print(f"WRONG (scale all, then split): {np.mean(wrong_accuracies):.1%}")
    print(f"RIGHT (split, then scale): {np.mean(right_accuracies):.1%}")
    ```

    **Output:**

    ```
    WRONG (scale all, then split): 75.7%
    RIGHT (split, then scale): 74.7%
    ```

    **Interpretation:** On small datasets (n=50), scaling before splitting inflates accuracy by ~1%. This gap widens with more preprocessing steps, smaller data, or time series. The wrong approach reports better results than you'll see in production—a subtle but dangerous form of overfitting.

    *Source: `computations/deep_dive_data_prep_examples.py` — `demo_data_leakage()`*


The correct workflow follows this sequence.

```
1. Split data FIRST
   └─→ Training set | Test set

2. Fit preprocessing on TRAINING only
   └─→ scaler.fit_transform(X_train)

3. Transform test using training parameters
   └─→ scaler.transform(X_test)
```

Extrapolation—when test data contains values outside the training range—requires attention for each preprocessing method. For standardization, values get z-scores beyond the training range, and most models handle this gracefully. For MinMax scaling, values might exceed [0,1], so consider clipping. For one-hot encoding, new categories are problematic, and setting `handle_unknown='ignore'` assigns zeros. In all cases, check whether training data covers the expected production range and monitor for out-of-distribution inputs.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: Fit on training ONLY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 3: Transform test using training parameters
X_test_scaled = scaler.transform(X_test)  # NOT fit_transform!
```

### Understanding fit, transform, and fit_transform

Think of a scaler like packing a suitcase. `fit()` is figuring out your packing strategy—measuring your clothes, deciding how to fold them. `transform()` is actually packing using that strategy. `fit_transform()` does both at once.

For training data, you figure out the strategy *and* pack (`fit_transform`). For test data, you use the *same* strategy you already figured out—you don't re-measure (`transform` only). If you called `fit_transform` on test data, you'd be creating a new packing strategy based on test clothes, which defeats the purpose of consistent preprocessing.

| Method | What it does | When to use |
|--------|--------------|-------------|
| `fit()` | Learns parameters from data (e.g., mean, std) | When you only need to learn, not transform |
| `transform()` | Applies learned parameters to transform data | On test data (using parameters learned from train) |
| `fit_transform()` | Does both in one step | On training data (learn + transform together) |

---

## Common Misconceptions

Several common beliefs about data preparation do not hold up in practice.

| Misconception | Reality |
|--------------|---------|
| "More features always improve models" | Too many features can cause overfitting. Feature selection is often necessary. |
| "Just drop all rows with missing values" | This can introduce bias and waste data. Imputation is often better. |
| "Always standardize your features" | Tree-based models don't need scaling. Know your algorithm. |
| "One-hot encoding is always best" | High-cardinality features may need target encoding or embeddings. |

Embeddings, covered in detail in Module 6, offer an alternative for high-cardinality categories. Instead of one-hot encoding (millions of columns for product IDs) or target encoding (which loses information), each category gets a small vector of continuous values learned during training. The model figures out which categories are "similar" based on their relationship to the target.
