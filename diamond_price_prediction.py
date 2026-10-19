"""
Diamond Price Prediction
------------------------
End-to-end regression project on a large real-world dataset (~54,000 diamonds):
data cleaning, exploratory analysis, feature engineering, model comparison with
cross-validation, hyperparameter tuning, and evaluation on a held-out test set.

Dataset: the classic "diamonds" dataset (53,940 rows, 10 columns). It is downloaded
once from the public seaborn-data GitHub repo and cached in data/diamonds.csv.

Run:  python diamond_price_prediction.py
"""
import json
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "diamonds.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
DATA.parent.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"

# 1. Load ----------------------------------------------------------------------
if not DATA.exists():
    urllib.request.urlretrieve(URL, DATA)
df = pd.read_csv(DATA)
print(f"Raw data: {df.shape[0]:,} rows x {df.shape[1]} columns | missing values: {int(df.isna().sum().sum())}")

# 2. Clean ---------------------------------------------------------------------
n0 = len(df)
dupes = int(df.duplicated().sum())
df = df.drop_duplicates()
bad_dims = int(((df[["x", "y", "z"]] == 0).any(axis=1)).sum())   # physically impossible sizes
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)]
outliers = int(((df["y"] > 30) | (df["z"] > 30) | (df["depth"] < 40)).sum())  # data-entry errors
df = df[(df["y"] <= 30) & (df["z"] <= 30) & (df["depth"] >= 40)]
print(f"Cleaning: removed {dupes} duplicates, {bad_dims} zero-dimension rows, {outliers} outliers "
      f"-> {len(df):,} rows kept ({n0 - len(df):,} removed)")

# 3. EDA -----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.histplot(df["price"], bins=60, ax=ax[0], color="#2F5C8F"); ax[0].set_title("Price (right-skewed)")
sns.histplot(np.log1p(df["price"]), bins=60, ax=ax[1], color="#2F5C8F"); ax[1].set_title("log(1 + price)")
plt.tight_layout(); plt.savefig(OUT / "price_distribution.png", dpi=150); plt.close()

cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]              # worst -> best
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]   # worst -> best

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, col, order in zip(ax, ["cut", "color", "clarity"], [cut_order, color_order, clarity_order]):
    sns.boxplot(data=df, x=col, y="price", order=order, ax=a, color="#9DB7D5", fliersize=1)
    a.set_title(f"Price by {col}")
plt.tight_layout(); plt.savefig(OUT / "price_by_category.png", dpi=150); plt.close()

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(df[["carat", "depth", "table", "x", "y", "z", "price"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numeric correlations"); plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=150); plt.close()

# 4. Feature engineering --------------------------------------------------------
df["volume"] = df["x"] * df["y"] * df["z"]
df["depth_pct_check"] = 2 * df["z"] / (df["x"] + df["y"]) * 100     # recomputed depth %
features_num = ["carat", "depth", "table", "x", "y", "z", "volume", "depth_pct_check"]
features_cat = ["cut", "color", "clarity"]
X, y = df[features_num + features_cat], df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

pre = ColumnTransformer([
    ("cat", OrdinalEncoder(categories=[cut_order, color_order, clarity_order]), features_cat),
], remainder="passthrough")


def make(model):
    """Model on log(price): fixes the skew, errors are then evaluated back in dollars."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", model)]),
        func=np.log1p, inverse_func=np.expm1)


# 5. Compare models with 5-fold CV ---------------------------------------------
models = {
    "Linear Regression": make(LinearRegression()),
    "Random Forest": make(RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=SEED)),
    "Gradient Boosting": make(HistGradientBoostingRegressor(random_state=SEED)),
}
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}
for name, est in models.items():
    s = cross_validate(est, X_train, y_train, cv=cv,
                       scoring=("r2", "neg_mean_absolute_error"), n_jobs=1)
    cv_results[name] = {"cv_r2": round(s["test_r2"].mean(), 4),
                        "cv_mae_usd": round(-s["test_neg_mean_absolute_error"].mean(), 1)}
    print(f"{name:20s} CV R2 = {cv_results[name]['cv_r2']:.4f} | CV MAE = ${cv_results[name]['cv_mae_usd']:,.0f}")

# 6. Tune the best family (gradient boosting) ----------------------------------
search = RandomizedSearchCV(
    make(HistGradientBoostingRegressor(random_state=SEED)),
    {"regressor__model__learning_rate": [0.03, 0.05, 0.1, 0.2],
     "regressor__model__max_iter": [200, 400, 600],
     "regressor__model__max_leaf_nodes": [15, 31, 63],
     "regressor__model__min_samples_leaf": [10, 20, 40],
     "regressor__model__l2_regularization": [0.0, 0.1, 1.0]},
    n_iter=15, cv=3, scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
best_params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
print("Tuned Gradient Boosting:", best_params)

# 7. Final evaluation on the held-out test set ---------------------------------
pred = best.predict(X_test)
test_metrics = {
    "r2": round(r2_score(y_test, pred), 4),
    "mae_usd": round(mean_absolute_error(y_test, pred), 1),
    "rmse_usd": round(float(np.sqrt(mean_squared_error(y_test, pred))), 1),
    "median_abs_pct_error": round(float(np.median(np.abs(pred - y_test) / y_test) * 100), 2),
}
lin_pred = models["Linear Regression"].fit(X_train, y_train).predict(X_test)
baseline_mae = round(mean_absolute_error(y_test, lin_pred), 1)
print("\nTest metrics (tuned Gradient Boosting):", test_metrics)
print(f"Baseline Linear Regression test MAE: ${baseline_mae:,.0f}")

plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, s=3, alpha=0.3, color="#2F5C8F")
lim = [0, max(y_test.max(), pred.max())]
plt.plot(lim, lim, "r--", lw=1); plt.xlabel("Actual price ($)"); plt.ylabel("Predicted price ($)")
plt.title("Predicted vs actual (test set)"); plt.tight_layout()
plt.savefig(OUT / "predicted_vs_actual.png", dpi=150); plt.close()

# 8. Feature importance (permutation, model-agnostic) ---------------------------
sample = X_test.sample(3000, random_state=SEED)
pi = permutation_importance(best, sample, y_test.loc[sample.index], n_repeats=5,
                            random_state=SEED, scoring="r2", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
imp.plot.barh(figsize=(6, 4.2), color="#2F5C8F")
plt.title("Permutation importance (drop in R2)"); plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()
top = list(imp.index[::-1][:4])
print("Most important features:", top)

# 9. Save results ---------------------------------------------------------------
with open(OUT / "results.json", "w") as f:
    json.dump({"rows_raw": n0, "rows_clean": len(df), "removed": {"duplicates": dupes,
               "zero_dimension": bad_dims, "outliers": outliers},
               "train_rows": len(X_train), "test_rows": len(X_test),
               "cross_validation": cv_results, "best_params": {k: str(v) for k, v in best_params.items()},
               "test_metrics": test_metrics, "baseline_linear_mae_usd": baseline_mae,
               "top_features": top}, f, indent=2)
