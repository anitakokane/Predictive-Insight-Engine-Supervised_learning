# House Price Prediction using Regression Models

A machine learning project that predicts house prices from property features. It compares Simple Linear, Multiple Linear and Polynomial Regression, and studies Gradient Descent optimization, bias-variance and overfitting.

## Dataset

`RealEstate_HousePrice_Dataset_4200.csv` - 4,200 houses, no missing values.

| Column | Description |
|---|---|
| `house_id` | Unique ID (dropped before modelling) |
| `area_sqft` | Built-up area in square feet |
| `bedrooms`, `bathrooms` | Number of bedrooms / bathrooms |
| `location_score` | Quality of the location |
| `age_years` | Age of the house |
| `distance_city_km` | Distance from the city centre |
| `lot_size_sqft` | Plot size |
| `has_garage`, `has_pool` | 1 if present, else 0 |
| `renovation_years_ago` | Years since last renovation |
| `house_price_inr` | **Target**: house price in INR |

## Project Structure

| Part | Content |
|---|---|
| A | Theory: supervised learning, regression vs classification, assumptions, bias-variance |
| B | Data loading, scaling (`StandardScaler`), train/test split (80/20) |
| C | Simple Linear Regression (`area_sqft` -> price), assumption checks |
| D | Evaluation metrics: MSE, MAE, RMSE, R², Adjusted R² |
| E | Multiple Linear Regression using all features |
| F | Polynomial Regression (degree 2) |
| G | Gradient Descent: SGD and Mini-Batch using `SGDRegressor` |
| H | Bias-variance analysis and model comparison |
| Summary | Findings and business interpretation |

## Results

All values are on scaled data (errors are in standard deviations of price).

| Model | Test RMSE | Test R² |
|---|---|---|
| Simple Linear (area only) | 0.661 | 0.563 |
| Multiple Linear (all features) | 0.286 | 0.918 |
| Polynomial (degree 2) | 0.180 | 0.967 |
| SGD (all features) | - | 0.915 |
| Mini-Batch GD (all features) | - | 0.910 |

## Key Findings

- **Best model:** Polynomial Regression (degree 2). It uses all features and captures curves and interactions.
- **Underfitting:** Simple regression has high error on both train and test (0.654 / 0.661).
- **No overfitting:** Train and test errors are close for multiple (0.274 / 0.286) and polynomial (0.178 / 0.180) models.
- **Gradient Descent:** SGD and Mini-Batch reach almost the same line as normal linear regression.
- **Main price drivers:** `area_sqft` and `location_score` raise the price; `age_years` and `distance_city_km` lower it.

## Business Interpretation

Area and location are the most important factors when pricing a house. The best model's typical error is about 0.18 standard deviations of price, which is accurate enough for pricing guidance and for spotting over- or under-priced listings. It should support human judgment, not replace it.

## Known Limitation

In Part F, the polynomial model is fitted on all rows (`pr.fit(x_poly, y)`), including the test rows, so its test score is slightly optimistic. Fitting on `x_train_poly` and `y_train` gives RMSE of about 0.183 and R² of about 0.966, and the model ranking stays the same.

## Tech Stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn
