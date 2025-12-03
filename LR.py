import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# === Load dataset ===
file_path = r'C:/Users/bmark/PycharmProjects/MS_Thesis/data/Dangermond Field Data Copy.xlsx'
df = pd.read_excel(file_path)
df['study_area'] = df['study_area'].str.lower()

# === Define predictor sets and target ===
spectral_predictors = ['cai', 'lcai', 'ndli']
lidar_predictors = ['chm_max', 'chm_range', 'chm_mean', 'chm_std', 'chm_median', 'chm_pct90']
combined_predictors = spectral_predictors + lidar_predictors
target = 'oven_weight'
selected_areas = ['steves_flat', 'jalama_bull', 'east_tinta', 'cojo_cow']

# === Define model specs ===
model_specs = [
    {"name": "Spectral", "predictors": spectral_predictors, "use_selected": True},
    {"name": "LiDAR", "predictors": lidar_predictors, "use_selected": False},
    {"name": "Combined", "predictors": combined_predictors, "use_selected": True}
]

results = []
plot_data = []

# === Loop over each predictor set ===
for spec in model_specs:
    predictors = spec["predictors"]
    set_name = spec["name"]
    use_selected = spec["use_selected"]

    df_filtered = df[df['study_area'].isin(selected_areas)] if use_selected else df
    df_subset = df_filtered[predictors + [target]].dropna()

    X = df_subset[predictors].copy()
    y = df_subset[target].copy()
    n = len(X)
    k = len(predictors)

    # Scale predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === LOOCV setup ===
    loo = LeaveOneOut()

    # === Linear Regression ===
    lr = LinearRegression()
    y_pred_lr = cross_val_predict(lr, X_scaled, y, cv=loo)

    r2_lr = r2_score(y, y_pred_lr)
    mae_lr = mean_absolute_error(y, y_pred_lr)
    sd_lr = np.std(y - y_pred_lr)

    results.append({
        "Model": f"{set_name} (LR)",
        "R2": round(r2_lr, 2),
        "MAE": round(mae_lr, 2),
        "SD_Error": round(sd_lr, 2),
        "Predictors": k,
        "Sample_Size": n
    })

    # === Random Forest ===
    rf = RandomForestRegressor(n_estimators=1000, max_depth=15, min_samples_split=5, random_state=42)
    y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=loo)

    r2_rf = r2_score(y, y_pred_rf)
    mae_rf = mean_absolute_error(y, y_pred_rf)
    sd_rf = np.std(y - y_pred_rf)

    results.append({
        "Model": f"{set_name} (RF)",
        "R2": round(r2_rf, 2),
        "MAE": round(mae_rf, 2),
        "SD_Error": round(sd_rf, 2),
        "Predictors": k,
        "Sample_Size": n
    })

    # === Store data for plotting ===
    df_plot = pd.DataFrame({
        "Actual": y,
        "Pred_LR": y_pred_lr,
        "Pred_RF": y_pred_rf,
        "Set": set_name,
        "R2_LR": r2_lr,
        "MAE_LR": mae_lr,
        "SD_LR": sd_lr,
        "R2_RF": r2_rf,
        "MAE_RF": mae_rf
    })
    plot_data.append(df_plot)

# === Combine and print results ===
results_df = pd.DataFrame(results)
print("Model Performance Summary (LOOCV):")
print(results_df)

# === Plot all three side-by-side with annotation ===
fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
fig.suptitle("Actual vs Predicted Oven Weight (LOOCV)", fontsize=18)

for ax, set_name in zip(axes, ["Spectral", "LiDAR", "Combined"]):
    df_all = pd.concat(plot_data)
    df_set = df_all[df_all["Set"] == set_name]

    ax.scatter(df_set["Actual"], df_set["Pred_RF"], color="darkgreen", s=50, label="RF")
    ax.scatter(df_set["Actual"], df_set["Pred_LR"], facecolors="none", edgecolors="blue", s=50, label="LR")

    # Line of best fit for LR
    coef = np.polyfit(df_set["Actual"], df_set["Pred_LR"], 1)
    x_vals = np.linspace(df_set["Actual"].min(), df_set["Actual"].max(), 100)
    y_vals = coef[0] * x_vals + coef[1]
    ax.plot(x_vals, y_vals, color="black")

    # Annotate performance
    stats = df_set.iloc[0]
    textstr = (
        f"RF  R² = {stats['R2_RF']:.2f}, MAE = {stats['MAE_RF']:.2f}\n"
        f"LR  R² = {stats['R2_LR']:.2f}, MAE = {stats['MAE_LR']:.2f}, SD = {stats['SD_LR']:.2f}"
    )
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    ax.set_title(f"{set_name} Predictors")
    ax.set_xlabel("Actual oven_weight")
    if ax is axes[0]:
        ax.set_ylabel("Predicted oven_weight")
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
