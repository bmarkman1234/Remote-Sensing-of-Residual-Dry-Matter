import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tabulate import tabulate

# Load dataset
file_path = r'C:\Users\bmark\PycharmProjects\MS_Thesis\data\Dangermond Field Data.xlsx'
df = pd.read_excel(file_path)
df['study_area'] = df['study_area'].str.lower()

# Define site groups
standing_sites = ['cmt_ungrazed', 'jalama_horse']
mixed_sites = ['jalachichi', 'steves_flat', 'jalama_bull', 'east_tinta']
laying_sites = ['cojo_cow', 'jalama_mare']

structure_map = {site: 'Standing' for site in standing_sites}
structure_map.update({site: 'Mixed' for site in mixed_sites})
structure_map.update({site: 'Laying' for site in laying_sites})
df['structure'] = df['study_area'].map(structure_map)

# Keep only relevant rows
df = df[df['structure'].isin(['Standing', 'Mixed', 'Laying'])].copy()

# LiDAR predictors
lidar_cols = ['chm_max', 'chm_range', 'chm_mean', 'chm_std', 'chm_median', 'chm_pct90']
df = df.dropna(subset=lidar_cols + ['oven_weight'])

# RF parameters
rf_params = {
    'n_estimators': 1000,
    'max_depth': 15,
    'min_samples_split': 5,
    'random_state': 42
}

results = []
palette = {'Standing': 'red', 'Mixed': 'green', 'Laying': 'blue'}

# Prepare figure
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("LiDAR LOOCV Predictions by Vegetation Structure", fontsize=18)

for i, structure in enumerate(['Standing', 'Mixed', 'Laying']):
    df_struct = df[df['structure'] == structure].copy()
    X = df_struct[lidar_cols]
    y = df_struct['oven_weight']

    if len(df_struct) < 2:
        print(f"Skipping {structure}: not enough data.")
        continue

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    rf = RandomForestRegressor(**rf_params)
    lr = LinearRegression()
    cv = LeaveOneOut()

    # Predict
    y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=cv)
    y_pred_lr = cross_val_predict(lr, X_scaled, y, cv=cv)

    # Metrics
    rf_r2 = r2_score(y, y_pred_rf)
    rf_mae = mean_absolute_error(y, y_pred_rf)
    lr_r2 = r2_score(y, y_pred_lr)
    lr_mae = mean_absolute_error(y, y_pred_lr)

    results.append([
        structure,
        round(rf_r2, 2), round(rf_mae, 2),
        round(lr_r2, 2), round(lr_mae, 2),
        ", ".join(sorted(df_struct['study_area'].unique())),
        len(df_struct)
    ])

    # Plot
    ax = axes[i]
    # RF: filled circles
    ax.scatter(y, y_pred_rf,
               c=palette[structure], edgecolor='k', marker='o',
               s=80, alpha=0.8, label="RF (filled)")

    # LR: hollow circles
    ax.scatter(y, y_pred_lr,
               facecolors='none', edgecolors=palette[structure], marker='o',
               s=80, linewidth=1.5, alpha=0.9, label="LR (hollow)")

    # Line of best fit (from LR)
    z = np.polyfit(y, y_pred_lr, 1)
    x_vals = np.linspace(y.min(), y.max(), 100)
    y_fit = np.polyval(z, x_vals)
    ax.plot(x_vals, y_fit, color='black', linewidth=2)

    # Annotate metrics
    perf_text = (
        f"RF  R² = {rf_r2:.2f}, MAE = {rf_mae:.2f}\n"
        f"LR  R² = {lr_r2:.2f}, MAE = {lr_mae:.2f}"
    )
    ax.text(0.05, 0.95, perf_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    ax.set_title(f"{structure} ({len(df_struct)} samples)", fontsize=16)
    ax.set_xlabel("Actual Oven Weight (g)", fontsize=14)
    if i == 0:
        ax.set_ylabel("Predicted Oven Weight (g)", fontsize=14)
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Print results table
headers = [
    "Vegetation Structure",
    "RF LOOCV R²", "RF LOOCV MAE",
    "LR LOOCV R²", "LR LOOCV MAE",
    "Sites", "Sample Size (N)"
]
print("\nModel Performance by Vegetation Structure (LiDAR Only):")
print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=".2f"))
