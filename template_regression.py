# 📁 ML Project Template: Regression Task (e.g., Housing Price Prediction)

# ====================================================
# 📌 1. SETUP
# ====================================================

## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from category_encoders import LeaveOneOutEncoder

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# ====================================================
# 📌 2. LOAD DATA
# ====================================================

# df = pd.read_csv("your_dataset.csv")
# For demo, assume df is already loaded

# ====================================================
# 📌 3. INITIAL EXPLORATION
# ====================================================

## Shape & basic info
df.shape
df.info()
df.describe().T

## Target variable analysis
sns.histplot(df['price'], kde=True)
plt.title("Distribution of Target Variable")
plt.show()

## Missing values
df.isnull().mean().sort_values(ascending=False)

# --- Numerical Feature Distributions ---
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[num_cols].hist(bins=30, figsize=(15, 10), layout=(len(num_cols) // 3 + 1, 3))
plt.suptitle("Numerical Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# --- Categorical Feature Distributions ---
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=X_train[col], order=X_train[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

# --- Correlation Heatmap (Numerical vs Target) ---
correlation_matrix = X_train[num_cols].copy()
correlation_matrix['target'] = y_train
corrs = correlation_matrix.corr()['target'].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=corrs.values, y=corrs.index)
plt.title("Correlation of Features with Target")
plt.show()

# ====================================================
# 📌 4. TRAIN-TEST SPLIT
# ====================================================

X = df.drop("price", axis=1)
y = df["price"]

# Apply log transformation to target
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42)

# ====================================================
# 📌 5. FEATURE CLASSIFICATION
# ====================================================

categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
log_features = [col for col in numeric_features if (X[col] > 0).all() and X[col].skew() > 1]

# ====================================================
# 📌 6. PREPROCESSING FACTORY
# ====================================================

def get_numeric_transformer(scale=True, log_features=None):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if log_features:
        steps.append(('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')))
    if scale:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps)

def get_categorical_transformer():
    return LeaveOneOutEncoder(cols=categorical_features)

def get_preprocessor(model_name, cat_features, num_features, log_features=None):
    scale = model_name not in ['XGBRegressor', 'RandomForestRegressor']
    return ColumnTransformer([
        ('cat', get_categorical_transformer(), cat_features),
        ('num', get_numeric_transformer(scale, log_features), num_features)
    ])

# ====================================================
# 📌 7. MODELING UTILITIES
# ====================================================

models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'XGBRegressor': XGBRegressor()
}

def evaluate_model(name, model, X_train, y_train, X_test, y_test, log_target=False):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if log_target:
        preds = np.expm1(preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"{name} RMSE: {rmse:.2f}")
    return rmse

# ====================================================
# 📌 8. BASELINE MODEL EVALUATION
# ====================================================

results = []

for name, model in models.items():
    log_target = name not in ['XGBRegressor', 'RandomForestRegressor']
    y_train_final = y_train_log if log_target else y_train
    y_test_final = y_test_log if log_target else y_test
    
    preprocessor = get_preprocessor(name, categorical_features, numeric_features, log_features)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    rmse = evaluate_model(name, pipeline, X_train, y_train_final, X_test, y_test_final, log_target)
    results.append((name, rmse))

# ====================================================
# 📌 9. HYPERPARAMETER TUNING (EXAMPLES)
# ====================================================

ridge_params = {
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}

rf_params = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5, 10]
}

# GridSearch Example
ridge_pipeline = Pipeline([
    ('preprocessor', get_preprocessor('Ridge', categorical_features, numeric_features, log_features)),
    ('model', Ridge())
])

gs_ridge = GridSearchCV(ridge_pipeline, param_grid=ridge_params, scoring='neg_root_mean_squared_error', cv=5)
gs_ridge.fit(X_train, y_train_log)
print("Best Ridge RMSE:", -gs_ridge.best_score_)
print("Best Params:", gs_ridge.best_params_)

# Bias-variance example
alphas = np.logspace(-4, 2, 20)
train_errors = []
val_errors = []

for alpha in alphas:
    model = Pipeline([
        ('preprocessor', get_preprocessor('Ridge', cat_features, num_features)),
        ('ridge', Ridge(alpha=alpha))
    ])

    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, train_preds, squared=False)
    val_rmse = mean_squared_error(y_test, val_preds, squared=False)

    train_errors.append(train_rmse)
    val_errors.append(val_rmse)

plt.figure(figsize=(10, 5))
plt.plot(alphas, train_errors, label="Train RMSE")
plt.plot(alphas, val_errors, label="Validation RMSE")
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.tight_layout()
plt.show()


# ====================================================
# 📌 10. FINAL MODEL SELECTION & FEATURE IMPORTANCE
# ====================================================

# Fit the final model (e.g., XGB)
final_model = Pipeline([
    ('preprocessor', get_preprocessor('XGBRegressor', categorical_features, numeric_features)),
    ('model', XGBRegressor(**xgb_best_params))
])

final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_preds, squared=False)
print("Final XGB RMSE:", final_rmse)


# Random Forest Feature Importance

# Extract feature importances
importances = rf_pipeline.named_steps['model'].feature_importances_

# Get feature names after preprocessing
cat_encoded = rf_pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
all_feature_names = np.concatenate([cat_encoded, numeric_features])

# Plot
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
feat_imp.plot(kind='bar')
plt.title('Feature Importance from Random Forest')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Other 

# ====================================================
# 📌 11. NEXT STEPS
# ====================================================

# - Save model with joblib
# - Report generation
# - Deployment (future)

# End of Template
