import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from merf.merf import MERF
from libpysal.weights import lat2W
from esda import Moran
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ----------------------- Data Loading -----------------------
file_path = r'/data/Dangermond Field Data.xlsx'
df = pd.read_excel(file_path)

selected_sites = ['steves_flat', 'jalama_bull', 'east_tinta', 'cojo_cow']

spectral_indices = ['cai', 'lcai', 'ndli']
lidar_metrics = ['chm_max', 'chm_range', 'chm_mean', 'chm_std', 'chm_median', 'chm_pct90']

plot_random = [col for col in ['slope', 'aspect', 'elevation',
                               'avg_NDVI_growing_season', 'max_NDVI_growing_season', 'sum_NDVI_growing_season'] if col in df.columns]

study_random = [col for col in ['grazing_days_total', 'aws0150wta', 'rootznaws', 'hydgrpdcd', 'kffact', 'slopegradw', 'brockdepmi'] if col in df.columns]

model_types = {
    "LiDAR Only": {"features": lidar_metrics, "data": df.copy()},
    "Spectral Only": {"features": spectral_indices, "data": df[df['study_area'].isin(selected_sites)].copy()},
    "Spectral + LiDAR": {"features": spectral_indices + lidar_metrics, "data": df[df['study_area'].isin(selected_sites)].copy()}
}

performance_results = []
merf_importance_results = {}
moran_boxplot_data = []

for model_name, model_info in model_types.items():
    features = model_info["features"]
    data_subset = model_info["data"].dropna(subset=["oven_weight", "study_area"] + features + plot_random + study_random)

    x_data = data_subset[features].astype(float)
    y_data = data_subset["oven_weight"].astype(float)

    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x_data), columns=features, index=x_data.index)

    exog_re = data_subset[plot_random]
    vc_formula = {var: f"0 + {var}" for var in study_random}
    mlmm_formula = "oven_weight ~ " + " + ".join(features)

    sites = data_subset["study_area"].unique()
    mlmm_preds, merf_preds, actuals = [], [], []

    for site in sites:
        train_idx = data_subset["study_area"] != site
        test_idx = data_subset["study_area"] == site

        # MLMM
        mlmm_model = mixedlm(mlmm_formula, data_subset.loc[train_idx], groups=data_subset.loc[train_idx, "study_area"], exog_re=exog_re.loc[train_idx], vc_formula=vc_formula).fit(reml=False)
        mlmm_site_preds = mlmm_model.predict(data_subset.loc[test_idx])
        mlmm_preds.extend(mlmm_site_preds)

        # MERF
        merf = MERF()
        merf.fit(X=x_scaled.loc[train_idx], Z=x_scaled.loc[train_idx], clusters=data_subset.loc[train_idx, "study_area"], y=y_data.loc[train_idx])
        merf_site_preds = merf.predict(X=x_scaled.loc[test_idx], Z=x_scaled.loc[test_idx], clusters=data_subset.loc[test_idx, "study_area"])
        merf_preds.extend(merf_site_preds)

        # Actual values
        actuals.extend(y_data.loc[test_idx])

        # Moran's I computation
        if len(data_subset.loc[test_idx]) == 9:
            w = lat2W(3, 3, rook=False)
            mlmm_resid = y_data.loc[test_idx] - mlmm_site_preds
            merf_resid = y_data.loc[test_idx] - merf_site_preds
            moran_boxplot_data.extend([
                {'Model': f'{model_name} (MLMM)', "Moran's I": Moran(mlmm_resid, w).I},
                {'Model': f'{model_name} (MERF)', "Moran's I": Moran(merf_resid, w).I}
            ])

    performance_results.extend([
        {'Model': f'{model_name} (MLMM)',
         'LosoCV R²': r2_score(actuals, mlmm_preds),
         'LosoCV RMSE': mean_squared_error(actuals, mlmm_preds, squared=False),
         'AIC': mlmm_model.aic},
        {'Model': f'{model_name} (MERF)',
         'LosoCV R²': r2_score(actuals, merf_preds),
         'LosoCV RMSE': mean_squared_error(actuals, merf_preds, squared=False),
         'AIC': 'N/A'}
    ])

    merf_importance_results[model_name] = pd.DataFrame({
        'Feature': features,
        'Importance': merf.trained_fe_model.feature_importances_
    }).sort_values('Importance', ascending=False)

# Performance Summary Table
perf_df = pd.DataFrame(performance_results)
print(tabulate(perf_df, headers='keys', tablefmt='grid'))

# Variable Importance Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (model_name, imp_df) in zip(axes, merf_importance_results.items()):
    sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax, color='black')
    ax.set_title(f"Feature Importance ({model_name})")
plt.tight_layout()
plt.show()

# Moran's I Boxplot
moran_df = pd.DataFrame(moran_boxplot_data).dropna()
if not moran_df.empty:
    moran_df['Method'] = moran_df['Model'].apply(lambda x: x.split('(')[-1].replace(')', '').strip())
    moran_df['Category'] = moran_df['Model'].apply(lambda x: x.split(' (')[0])
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=moran_df, x='Category', y="Moran's I", hue='Method', palette={"MLMM": "black", "MERF": "gray"}, medianprops=dict(color='white', linewidth=2))
    plt.xlabel("Model Category")
    plt.ylabel("Moran's I")
    plt.xticks(rotation=45)
    plt.title("Moran's I Distribution by Model")
    plt.tight_layout()
    plt.show()
else:
    print("No valid Moran's I values for plotting.")
