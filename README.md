# Diamond Price Prediction

An end-to-end machine learning project that predicts the **price of a diamond** from its physical characteristics and quality grades, trained on a large real-world dataset of about 54,000 diamonds.

- **Dataset:** the classic "diamonds" dataset, 53,940 rows and 10 columns: carat, cut, color, clarity, depth, table, x/y/z dimensions and price. The script downloads it once from the public seaborn-data repo on GitHub and caches it in `data/diamonds.csv`.
- **Task:** Regression (predict price in USD).
- **Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

## What the script does

1. **Cleans the data.** It removes 146 duplicate rows, 19 rows with impossible zero dimensions and 3 data-entry outliers, leaving 53,772 rows.
2. **Explores it.** Price is heavily right-skewed, so it plots the distribution and price by cut, colour and clarity, and a correlation heatmap.
3. **Engineers features.** It adds a `volume` feature (x * y * z) and a recomputed depth percentage, and encodes cut, colour and clarity in their natural quality order.
4. **Models log(price).** This handles the skew, and errors are reported back in dollars.
5. **Splits the data** 80/20 into 43,017 training rows and 10,755 test rows.
6. **Compares models** with 5-fold cross-validation: Linear Regression, Random Forest and Gradient Boosting.
7. **Tunes the best family.** It runs `RandomizedSearchCV` on Gradient Boosting.
8. **Evaluates once** on the held-out test set and plots predicted vs actual prices.
9. **Explains the model** with permutation importance.

## Results

| Model | CV R² | CV MAE (USD) |
|---|---|---|
| Linear Regression | 0.9556 | 444 |
| Random Forest | 0.9807 | 271 |
| Gradient Boosting | 0.9810 | 278 |

After tuning, Gradient Boosting (learning rate 0.03, 600 iterations, 63 leaf nodes) gave these scores on the held-out test set of 10,755 diamonds:

| Metric | Score |
|---|---|
| R² | 0.9822 |
| Mean absolute error | $265.8 |
| RMSE | $526.6 |
| Median absolute % error | 4.54% |

The linear baseline had a test MAE of $440, so the tuned model cuts the average error by about 40%.

The most influential features were `y`, `carat`, `volume` and `clarity`. Several of these are strongly correlated with each other (size measurements), so their importance is shared between them.

Plots are saved in `outputs/`: price distribution, price by category, correlation heatmap, predicted vs actual, and feature importance.

## Run it

```bash
pip install scikit-learn pandas matplotlib seaborn
python diamond_price_prediction.py
```

## Limitations

- The data is a single historical snapshot with no dates, so the model can't capture price changes over time.
- Prices of the most expensive diamonds are harder to predict, so errors in dollars are larger at the high end.
- This is a learning project, not a pricing tool for real trading.

## Ideas to extend it

- Add XGBoost or LightGBM and compare them.
- Add SHAP values to explain individual predictions.
- Wrap the model in a FastAPI endpoint with a small web form and deploy it.
