# Indian Flight Price Prediction

A machine learning project to predict flight ticket prices in India using regression models with hyperparameter tuning.

## Project Overview

This project builds an end-to-end machine learning pipeline to predict Indian flight ticket prices. The workflow includes comprehensive exploratory data analysis (EDA), feature engineering, data preprocessing, baseline model training, hyperparameter optimization using RandomizedSearchCV, and model selection.

## Dataset

**Source:** [Indian Flight Dataset - Kaggle](https://www.kaggle.com/datasets/neurocipher/indian-flight-dataser)

**Data Characteristics:**

- Airline information (6 major airlines + others)
- Flight routes (source and destination cities)
- Journey dates with temporal features
- Number of stops
- Additional information about flights
- Target: Flight ticket price (in ₹)

## Exploratory Data Analysis (EDA)

The project includes comprehensive analysis across multiple dimensions:

### Temporal Analysis

- **Hourly demand patterns** - Peak booking hours
- **Monthly trends** - Seasonal price variations
- **Weekly patterns** - Day-of-week effects on prices
- **Seasonal analysis** - Winter, Spring, Summer, Autumn patterns

### Categorical Analysis

- **Airline performance metrics**
  - Average flight duration by airline
  - Average ticket price by airline
- **Route analysis** - Top 12 routes, origin-destination pairs
- **Destination distribution** - Popular and unpopular routes

### Combined Analysis

- Season × Weekday patterns
- Month × Weekday demand heatmaps
- Season × Month relationships

## Feature Engineering

**Temporal Features (Cyclical Encoding):**

- `day_sin`, `day_cos` - Day of month encoded cyclically
- `month_sin`, `month_cos` - Month encoded cyclically
- `weekday_sin`, `weekday_cos` - Day of week encoded cyclically
- `season` - Derived season (Winter, Spring, Summer, Autumn)

**Categorical Features (Top-N Grouping):**

- `airline` - Top 6 airlines + others
- `source` - Origin city
- `destination` - Destination city (Top 12 + others)
- `additional_info` - Top 3 info types + others
- `season` - Seasonal category

**Numerical Features:**

- `duration` - Flight duration in minutes (converted from "2h 50m" format)
- `total_stops` - Number of stops

## Data Preprocessing

**Preprocessing Pipeline:**

1. **Numerical Scaling** - StandardScaler for standardization
2. **Categorical Encoding** - OneHotEncoder for one-hot encoding
3. **Train-Test Split** - 80-20 split with random_state=42

**Total Features After Preprocessing:** 30+ features (8 numerical + encoded categorical)

## Models Trained

### Default Models (Baseline)

1. **Linear Regression** - Linear baseline model
2. **Ridge** - L2 regularization
3. **Lasso** - L1 regularization  
4. **ElasticNet** - Combined L1/L2 regularization
5. **Decision Tree** - Tree-based model
6. **Random Forest** - Ensemble of decision trees

### Hyperparameter Tuning

Used **RandomizedSearchCV** with:

- **Iterations:** 20 random parameter combinations per model
- **Cross-Validation:** 5-fold CV for robust evaluation
- **Scoring Metric:** R² score

**Parameter Grids:**

- **Ridge:** alpha [0.001-100], solver variations
- **Lasso:** alpha [0.0001-1], max_iter [1000, 5000, 10000]
- **ElasticNet:** alpha, l1_ratio, max_iter combinations
- **Decision Tree:** max_depth, min_samples_split, min_samples_leaf
- **Random Forest:** n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

## Evaluation Metrics

**Metrics Used:**

- **MAE (Mean Absolute Error)** - Average absolute prediction error
- **RMSE (Root Mean Squared Error)** - Square root of mean squared error
- **R² Score** - Coefficient of determination (0-1, higher is better)

**Best Model:** [Best performing tuned model based on R² score]

## Project Structure

```
IndianFlightPricePrediction/
├── IndianFlightPricePrediction.ipynb  # Main Jupyter notebook
├── README.md                           # This file
└── models/                             # Saved model artifacts
    ├── best_model.pkl                 # Tuned best model
    ├── preprocessor.pkl               # Data preprocessor
    ├── feature_names.pkl              # Feature names list
    └── best_params.pkl                # Best hyperparameters
```

## Quick Start

### Installation

```bash
# Required libraries
pip install pandas numpy scikit-learn matplotlib seaborn joblib kagglehub
```

### Loading and Using the Saved Model

```python
import joblib
import pandas as pd

# Load saved artifacts
model = joblib.load('models/best_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
feature_names = joblib.load('models/feature_names.pkl')
best_params = joblib.load('models/best_params.pkl')

# Prepare new data (must have these columns)
# airline, source, destination, duration, total_stops, 
# additional_info, day_sin, day_cos, month_sin, month_cos, 
# weekday_sin, weekday_cos, season

X_new = pd.DataFrame({...})  # Your new flight data

# Preprocess
X_new_preprocessed = preprocessor.transform(X_new)

# Make predictions
predicted_prices = model.predict(X_new_preprocessed)
```

## Model Performance

### Comparison: Before vs After Hyperparameter Tuning

The hyperparameter tuning process improved model performance across all models:

**Metrics Compared:**

- R² Score improvement
- MAE (Mean Absolute Error) reduction
- RMSE (Root Mean Squared Error) reduction

**Visualizations Generated:**

- Before/After R² comparison bar chart
- R² improvement chart (color-coded: green for gains, red for losses)
- Sorted MAE and RMSE visualizations for tuned models
- Actual vs Predicted scatter plots for best models
- Line plot showing all 6 models' performance improvement

## Hyperparameter Tuning Results

**Process:**

1. Defined parameter grids for each model
2. Ran RandomizedSearchCV with 20 iterations and 5-fold CV
3. Evaluated on test set
4. Created comprehensive comparison visualizations
5. Selected best tuned model

**Output:**

- Best hyperparameters for each model
- Cross-validation scores
- Test set metrics (MAE, RMSE, R²)
- Detailed comparison dataframes

## Model Persistence

The best tuned model and all necessary artifacts are saved for production use:

- **best_model.pkl** - Trained model ready for predictions
- **preprocessor.pkl** - Fitted preprocessor for data transformation
- **feature_names.pkl** - Feature names for reference
- **best_params.pkl** - Best hyperparameters found during tuning

## 📝 Key Findings

1. **Temporal Patterns:** Prices vary significantly by season, month, and day of week
2. **Airline Effects:** Different airlines have different average prices and flight durations
3. **Route Impact:** Popular routes (top 12) show distinct price patterns
4. **Cyclical Features:** Cyclical encoding of temporal features captures seasonal patterns effectively
5. **Model Selection:** Ensemble methods (Random Forest) generally outperform linear models

## Next Steps

Potential improvements and extensions:

1. **Feature Importance Analysis** - Identify most influential features
2. **Cross-Validation Analysis** - Test consistency across different data splits
3. **Residual Analysis** - Analyze prediction errors for patterns
4. **Ensemble Methods** - Combine multiple tuned models
5. **Time Series Analysis** - Capture temporal dependencies
6. **Deep Learning** - Neural networks for non-linear patterns

## Contact & Notes

- **Dataset Source:** Kaggle Indian Flight Dataset
- **Tools Used:** Python, scikit-learn, pandas, matplotlib, seaborn
- **Jupyter Notebook:** IndianFlightPricePrediction.ipynb

## License

Dataset sourced from Kaggle. Check original dataset for license information.

---

**Created:** December 2025  
**Project Type:** Machine Learning Regression  
**Status:** Complete with hyperparameter tuning and model persistence
