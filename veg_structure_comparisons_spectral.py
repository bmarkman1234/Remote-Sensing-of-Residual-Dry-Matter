import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.lines import Line2D
from tabulate import tabulate  # for printing tables

# Load data
file_path = r'/data/Dangermond Field Data.xlsx'
df = pd.read_excel(file_path)
df['study_area'] = df['study_area'].str.lower()

# Define selected sites where spectral data is available
selected_sites = ['steves_flat', 'jalama_bull', 'east_tinta', 'cojo_cow']
df_selected = df[df['study_area'].isin(selected_sites)].copy()

# Define spectral indices
indices = ['cai', 'lcai', 'ndli']
X = df_selected[indices]
y = df_selected['oven_weight']

# Scale spectral indices
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Set Random Forest parameters
rf_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'oob_score': True
}

# Fit models and compute LOOCV predictions
rf = RandomForestRegressor(**rf_params)
rf.fit(X_scaled, y)
loo = LeaveOneOut()
rf_preds = cross_val_predict(rf, X_scaled, y, cv=loo)

lr = LinearRegression()
lr.fit(X_scaled, y)
lr_preds = cross_val_predict(lr, X_scaled, y, cv=loo)

# Store predictions in dataframe
df_selected['rf_loocv'] = rf_preds
df_selected['lr_loocv'] = lr_preds

# Assign vegetation type (if relevant)
veg_mapping = {
    'steves_flat': 'Mixed',
    'jalama_bull': 'Mixed',
    'east_tinta': 'Mixed',
    'cojo_cow': 'Laying'
}
df_selected['veg_type_3'] = df_selected['study_area'].map(veg_mapping)

# Define color palette
palette = {'Mixed': 'green', 'Laying': 'blue'}

# Prepare scatterplots for RF and LR LOOCV predictions
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# RF Scatterplot
sns.scatterplot(data=df_selected, x='oven_weight', y='rf_loocv', hue='veg_type_3',
                palette=palette, s=100, edgecolor='k', alpha=0.8, ax=axes[0])
axes[0].plot([df_selected['oven_weight'].min(), df_selected['oven_weight'].max()],
             [df_selected['oven_weight'].min(), df_selected['oven_weight'].max()],
             linewidth=2, color='black')
axes[0].set_xlabel("RDM (g)", fontsize=23, labelpad=15)
axes[0].set_ylabel("RF LOOCV Predicted Oven Weight (g)", fontsize=25, labelpad=15)
axes[0].tick_params(axis='both', labelsize=26)

# Compute metrics for RF predictions
rf_metrics = {}
for veg in df_selected['veg_type_3'].unique():
    sub = df_selected[df_selected['veg_type_3'] == veg]
    rf_metrics[veg] = (r2_score(sub['oven_weight'], sub['rf_loocv']),
                       np.sqrt(mean_squared_error(sub['oven_weight'], sub['rf_loocv'])))

# Create legend handles for RF
rf_handles, rf_labels = [], []
for veg in df_selected['veg_type_3'].unique():
    rf_handles.append(Line2D([], [], marker='o', color=palette[veg], linestyle='', markersize=12))
    rf_labels.append(veg)
axes[0].legend(rf_handles, rf_labels, title="RF", fontsize=24, title_fontsize=28)

# LR Scatterplot
sns.scatterplot(data=df_selected, x='oven_weight', y='lr_loocv', hue='veg_type_3',
                palette=palette, s=100, edgecolor='k', alpha=0.8, ax=axes[1])
axes[1].plot([df_selected['oven_weight'].min(), df_selected['oven_weight'].max()],
             [df_selected['oven_weight'].min(), df_selected['oven_weight'].max()],
             linewidth=2, color='black')
axes[1].set_xlabel("RDM (g)", fontsize=23, labelpad=15)
axes[1].set_ylabel("LR LOOCV Predicted Oven Weight (g)", fontsize=25, labelpad=15)
axes[1].tick_params(axis='both', labelsize=26)

# Compute metrics for LR predictions
lr_metrics = {}
for veg in df_selected['veg_type_3'].unique():
    sub = df_selected[df_selected['veg_type_3'] == veg]
    lr_metrics[veg] = (r2_score(sub['oven_weight'], sub['lr_loocv']),
                       np.sqrt(mean_squared_error(sub['oven_weight'], sub['lr_loocv'])))

# Create legend handles for LR
lr_handles, lr_labels = [], []
for veg in df_selected['veg_type_3'].unique():
    lr_handles.append(Line2D([], [], marker='o', color=palette[veg], linestyle='', markersize=12))
    lr_labels.append(veg)
axes[1].legend(lr_handles, lr_labels, title="LR", fontsize=24, title_fontsize=28)

plt.tight_layout()
plt.show()

# Create and print metrics table using tabulate
table_data = []
for veg in df_selected['veg_type_3'].unique():
    rf_r2, rf_rmse = rf_metrics[veg]
    lr_r2, lr_rmse = lr_metrics[veg]
    table_data.append([
        veg,
        f"{rf_r2:.2f}" if rf_r2 is not None else "N/A",
        f"{rf_rmse:.2f}" if rf_rmse is not None else "N/A",
        f"{lr_r2:.2f}" if lr_r2 is not None else "N/A",
        f"{lr_rmse:.2f}" if lr_rmse is not None else "N/A"
    ])

print(tabulate(table_data,
               headers=["Vegetation", "RF R²", "RF RMSE", "LR R²", "LR RMSE"],
               tablefmt="pretty"))
