# 📘 Machine Learning Quick Reference Guide (Regression & Classification)

This is a concise, decision-oriented guide for traditional ML projects using scikit-learn, XGBoost, etc.

---

## 📂 1. Data Understanding

### ⬇️ Load & Inspect
```python
import pandas as pd

# Load
df = pd.read_csv("file.csv")

# Inspect
print(df.info())
print(df.describe())
print(df.head())
```

### 📊 Check Distributions, Nulls, Duplicates
```python
import seaborn as sns
sns.histplot(df['feature'])
df.isnull().sum()
df.duplicated().sum()
```

---

## 🧹 2. Preprocessing

### 🧱 Handling Missing Values
| Scenario | Method |
|----------|--------|
| Numerical + small # of nulls | Median Imputation |
| Numerical + relationship to other vars | Predictive Imputation |
| Categorical | Mode or "Unknown" Label |

```python
from sklearn.impute import SimpleImputer
SimpleImputer(strategy="median")
```

### 📦 Binning (Discretization)
**Use when:**
- Feature has nonlinear relation to target
- You want interpretability (e.g. income brackets)

```python
from sklearn.preprocessing import KBinsDiscretizer
KBinsDiscretizer(n_bins=5, encode='ordinal')
```

### 🧮 Categorical Encoding
| Data Type | Cardinality | Tree-Based? | Encoder |
|-----------|-------------|-------------|---------|
| Nominal   | Low         | Any         | OneHotEncoder |
| Nominal   | High        | Tree        | LeaveOneOutEncoder |
| Nominal   | High        | Linear      | TargetEncoder |
| Ordinal   | Any         | Any         | OrdinalEncoder |

### 📐 Feature Scaling
| Algorithm Type | Use Scaling? | Scaler |
|----------------|--------------|--------|
| Linear Models  | ✅ Yes        | StandardScaler |
| Distance-Based (SVM, KNN) | ✅ Yes | StandardScaler / MinMaxScaler |
| Tree-Based Models | ❌ No     | None |

---

## 🛠️ 3. Feature Engineering

### 🔧 Log Transform Skewed Features
```python
import numpy as np
np.log1p(df['skewed_column'])
```

### ✅ Polynomial Features (if non-linear relationship)
```python
from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures(degree=2)
```

### 🪓 Feature Selection
| Technique | When to Use |
|----------|-------------|
| `model.coef_` or `feature_importances_` | After training a model |
| `SelectKBest(score_func=...)` | Statistical filter method |
| Recursive Feature Elimination (RFE) | Wrapper method (slower, more accurate) |

```python
from sklearn.feature_selection import SelectKBest, f_regression
SelectKBest(score_func=f_regression, k=10)
```

---

## ⚙️ 4. Preprocessing Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer

categorical = ['city']
numeric = ['sqft_living', 'sqft_lot']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric),
    ('cat', LeaveOneOutEncoder(cols=categorical), categorical)
])
```

**Use pipeline when:**
- You want to cross-validate correctly
- You need consistent processing during train/test
- You’ll tune hyperparameters later

---

## 🧠 5. Model Selection

| Situation | Suggested Models |
|----------|------------------|
| Linear, fast baseline | Ridge, Lasso, ElasticNet |
| Nonlinear relationships | Random Forest, XGBoost |
| Small dataset, robust | Random Forest |
| High interpretability | Ridge/Lasso |
| Need probability output | Logistic Regression, CalibratedClassifierCV |
| Sensitive to scaling | SVR, KNN |

```python
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
```

---

## 🔁 6. Cross-Validation Strategy

| Problem Type | Recommended CV |
|--------------|----------------|
| Standard (IID) | `KFold` |
| Classification with Imbalance | `StratifiedKFold` |
| Time Series | `TimeSeriesSplit` |

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5)
```

---

## 🧪 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
```

### 🎯 Common Param Grids
```python
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
xgb_params = {
    'n_estimators': [500, 1000],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1]
}
```

---

## 🧮 8. Model Evaluation

### 📊 Metrics for Regression
| Metric | When to Use |
|--------|-------------|
| RMSE   | Penalizes large errors (default) |
| MAE    | More robust to outliers |
| R²     | Explained variance |

### 📊 Metrics for Classification
| Metric | When to Use |
|--------|-------------|
| Accuracy | Balanced classes |
| Precision | False positives are costly |
| Recall | False negatives are costly |
| F1 Score | Balance of precision & recall |
| ROC AUC | Good overall binary classifier eval |

```python
from sklearn.metrics import mean_squared_error, f1_score
rmse = mean_squared_error(y_test, y_pred, squared=False)
```

