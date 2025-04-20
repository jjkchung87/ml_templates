# ==================================================
# 🧠 CHURN PREDICTION - MACHINE LEARNING TEMPLATE
# ==================================================

# ✅ Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Modeling & Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# ✅ Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ✅ Display Settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# -----------------------------
# 📥 LOAD DATA
# -----------------------------
df = pd.read_csv("churn.csv")  # Replace with your dataset path
print(df.shape)
df.head()

# -----------------------------
# 🧹 INITIAL CLEANING
# -----------------------------
# Example: Drop ID columns
if 'customer_id' in df.columns:
    df.drop('customer_id', axis=1, inplace=True)

# -----------------------------
# 📊 EXPLORATORY DATA ANALYSIS
# -----------------------------

# ✅ Data types and missing values
print(df.info())
print(df.isna().sum())

# ✅ Distribution of numerical features
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('churn')
df[num_cols].hist(bins=30, figsize=(15, 8))
plt.tight_layout()
plt.show()

# ✅ Distribution of categorical features
cat_cols = df.select_dtypes(include='object').columns
def plot_categorical_distributions(df, cat_cols):
    for col in cat_cols:
        plt.figure(figsize=(6,3))
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

plot_categorical_distributions(df, cat_cols)

# ✅ Correlation with target (numeric only)
correlations = df[num_cols.tolist() + ['churn']].corr()['churn'].drop('churn')
correlations.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Correlation with Churn")
plt.show()

# -----------------------------
# ✨ FEATURE ENGINEERING
# -----------------------------
# Example: Convert target to binary
if df['churn'].dtype == 'object':
    df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# -----------------------------
# ✂️ FEATURE SELECTION (optional)
# -----------------------------
# Can use correlation, tree-based importance, domain knowledge, etc.

# -----------------------------
# ✂️ TRAIN-TEST SPLIT
# -----------------------------
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# ⚙️ PREPROCESSING PIPELINE
# -----------------------------
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include='object').columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# -----------------------------
# 🧪 MODEL TRAINING & BASELINE
# -----------------------------
def evaluate_model(name, model):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n📌 {name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {name}")
    plt.show()

# Evaluate baseline models
evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000))
evaluate_model("Random Forest", RandomForestClassifier(random_state=42))
evaluate_model("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))

# -----------------------------
# 🔍 HYPERPARAMETER TUNING
# -----------------------------
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=5,
                           scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("\n✅ Best Parameters:", grid_search.best_params_)
print("✅ Best ROC AUC:", grid_search.best_score_)

# -----------------------------
# 📉 BIAS-VARIANCE VISUALIZATION
# -----------------------------
results = pd.DataFrame(grid_search.cv_results_)
plt.plot(results['param_classifier__n_estimators'], -results['mean_test_score'])
plt.xlabel("n_estimators")
plt.ylabel("Negative ROC AUC")
plt.title("Bias-Variance Tradeoff (ROC AUC)")
plt.show()

# -----------------------------
# 📉 FINAL MODEL EVALUATION
# -----------------------------

# Evaluate final model
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Plot ROC curve
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# -----------------------------
# 🧠 FEATURE IMPORTANCE (Tree Models)
# -----------------------------
final_model = grid_search.best_estimator_.named_steps['classifier']
importances = final_model.feature_importances_
feature_names = grid_search.best_estimator_.named_steps['preprocessor'].transformers_[0][2].tolist() + \
                list(grid_search.best_estimator_.named_steps['preprocessor'].named_transformers_['cat']
                     .named_steps['onehot'].get_feature_names_out(cat_features))
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
feat_imp.plot(kind='barh', figsize=(10,6))
plt.title("Top 20 Feature Importances")
plt.show()
