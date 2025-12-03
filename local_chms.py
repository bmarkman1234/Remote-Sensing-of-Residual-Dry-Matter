import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

# Load the dataset
data_file_path = r'/data/Dangermond Field Data.xlsx'
df = pd.read_excel(data_file_path)

# Define predictors (CHM metrics) and response variable
chm_cols = ['chm_min', 'chm_max', 'chm_range', 'chm_mean', 'chm_std', 'chm_sum', 'chm_median', 'chm_pct90']
y_col = 'oven_weight'

# Loop through each study area
for study_area in df['study_area'].unique():
    study_data = df[df['study_area'] == study_area]

    y = study_data[y_col]

    # Create scatterplots for each CHM metric
    plt.figure(figsize=(20, 15))
    for idx, metric in enumerate(chm_cols, start=1):
        x = study_data[[metric]].values

        # Fit Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(x, y)
        lr_predictions = lr_model.predict(x)
        lr_r2 = r2_score(y, lr_predictions)
        lr_rmse = mean_squared_error(y, lr_predictions, squared=False)

        # Fit Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(x, y)
        rf_predictions = rf_model.predict(x)
        rf_r2 = r2_score(y, rf_predictions)
        rf_rmse = mean_squared_error(y, rf_predictions, squared=False)

        # Plot LR and RF predicted values against true values
        plt.subplot(3, 3, idx)
        plt.scatter(lr_predictions, y, label=f'LR (R²={lr_r2:.2f}, RMSE={lr_rmse:.2f})', alpha=0.6, color='orange')
        plt.scatter(rf_predictions, y, label=f'RF (R²={rf_r2:.2f}, RMSE={rf_rmse:.2f})', alpha=0.6, color='blue')

        # Add 1:1 line
        min_val = min(min(lr_predictions), min(rf_predictions), min(y))
        max_val = max(max(lr_predictions), max(rf_predictions), max(y))
        plt.plot([min_val, max_val], [min_val, max_val], color = 'black')

        # Add plot labels and legend
        plt.title(f'{metric}', fontsize=14)
        plt.xlabel('Predicted RDM (grams)')
        plt.ylabel('True RDM (grams)')
        plt.legend()

    plt.tight_layout()
    plt.suptitle(f'Predicted vs True RDM in Study Area: {study_area}', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()
