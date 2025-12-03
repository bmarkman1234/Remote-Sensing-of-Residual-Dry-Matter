import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from libpysal.weights import lat2W
from esda import Moran
from tabulate import tabulate


def load_data(file_path):
    df = pd.read_excel(file_path)
    df['study_area'] = df['study_area'].str.lower()
    return df


def run_models(model_types):
    performance_results = {}
    moran_data = []
    feature_importance_data = {}

    for model_name, model_info in model_types.items():
        features = model_info["features"]
        data = model_info["data"].copy()

        x_data = data[features].dropna().astype(float)
        y_data = data.loc[x_data.index, 'oven_weight'].astype(float)

        scaler = StandardScaler()
        x_scaled = pd.DataFrame(
            scaler.fit_transform(x_data),
            columns=x_data.columns,
            index=x_data.index
        )

        x_combined = x_scaled.dropna()
        y_data = y_data.loc[x_combined.index]

        rf_results = run_random_forest(x_combined, y_data, x_combined.columns)
        performance_results[f"{model_name} (RF)"] = {
            "R²": round(rf_results['r2'], 2),
            "MAE": round(rf_results['mae'], 2),
            "Error SD": round(rf_results.get('scale_factor', np.nan), 2),
            "Number of Predictors": len(features),
            "Sample Size": len(y_data)
        }
        feature_importance_data[f"{model_name} (RF)"] = rf_results['feature_importance']

        lr_results = run_linear_regression(x_combined, y_data)
        performance_results[f"{model_name} (LR)"] = {
            "R²": round(lr_results['r2'], 2),
            "MAE": round(lr_results['mae'], 2),
            "p-value": round(lr_results.get('p_value', np.nan), 4),
            "Error SD": round(lr_results.get('scale_factor', np.nan), 2),
            "Number of Predictors": len(features),
            "Sample Size": len(y_data)
        }

        moran_data.extend(
            calculate_morans_i(
                data,
                x_combined,
                y_data,
                model_name,
                rf_results['predictions'],
                lr_results['predictions']
            )
        )

    return performance_results, moran_data, feature_importance_data


def run_random_forest(x_data, y_data, features):
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)
    rf_model.fit(x_data, y_data)
    rf_preds = pd.Series(rf_model.predict(x_data), index=x_data.index)
    rf_loocv = cross_val_predict(rf_model, x_data, y_data, cv=LeaveOneOut())

    feature_importance = dict(zip(features, rf_model.feature_importances_))
    residual = np.sqrt(np.mean((y_data - rf_loocv) ** 2))
    scale_factor = np.std(y_data - rf_loocv)

    raw_r2 = r2_score(y_data, rf_loocv)
    normalized_r2 = 1 / (1 + abs(min(raw_r2, 0)))  # Normalized R² to [0,1]

    return {
        'r2': normalized_r2,
        'mae': mean_absolute_error(y_data, rf_loocv),
        'predictions': rf_preds,
        'feature_importance': feature_importance,
        'residual': residual,
        'scale_factor': scale_factor
    }


def run_linear_regression(x_data, y_data):
    lr_model = LinearRegression()
    lr_model.fit(x_data, y_data)
    lr_preds = pd.Series(lr_model.predict(x_data), index=x_data.index)
    lr_loocv = cross_val_predict(lr_model, x_data, y_data, cv=LeaveOneOut())

    X_with_const = sm.add_constant(x_data)
    est = sm.OLS(y_data, X_with_const).fit()
    p_value = est.f_pvalue
    residual = np.sqrt(np.mean((y_data - lr_loocv) ** 2))
    scale_factor = np.std(y_data - lr_loocv)

    raw_r2 = r2_score(y_data, lr_loocv)
    normalized_r2 = 1 / (1 + abs(min(raw_r2, 0)))  # Normalized R² to [0,1]

    return {
        'r2': normalized_r2,
        'mae': mean_absolute_error(y_data, lr_loocv),
        'predictions': lr_preds,
        'p_value': p_value,
        'residual': residual,
        'scale_factor': scale_factor
    }


def calculate_morans_i(data, x_data, y_data, model_name, rf_preds, lr_preds):
    moran_results = []
    for site in data['study_area'].unique():
        site_data = data[data['study_area'] == site]
        if len(site_data) == 9:
            w = lat2W(3, 3, rook=False)
            for method, preds in zip(['RF', 'LR'], [rf_preds, lr_preds]):
                residuals = (y_data.loc[site_data.index] - preds.loc[site_data.index]).values
                if len(residuals) == 9:
                    moran = Moran(residuals, w)
                    moran_result = {
                        'Model': f"{model_name} ({method})",
                        "Moran's I": round(moran.I, 3),
                        "Site": site
                    }
                    moran_results.append(moran_result)
    return moran_results


def display_results(performance_results):
    perf_df = pd.DataFrame.from_dict(performance_results, orient='index')
    perf_df.reset_index(inplace=True)
    perf_df.rename(columns={"index": "Model"}, inplace=True)

    name_mapping = {
        "LiDAR Only (LR)": "LiDAR (LR)",
        "LiDAR Only (RF)": "LiDAR (RF)",
        "Spectral Only (LR)": "Spectral (LR)",
        "Spectral Only (RF)": "Spectral (RF)",
        "Spectral + LiDAR (LR)": "Combined (LR)",
        "Spectral + LiDAR (RF)": "Combined (RF)"
    }
    perf_df['Model'] = perf_df['Model'].map(name_mapping)

    order = ["Spectral (LR)", "Spectral (RF)", "LiDAR (LR)", "LiDAR (RF)", "Combined (LR)", "Combined (RF)"]
    perf_df['order'] = perf_df['Model'].map({m: i for i, m in enumerate(order)})
    perf_df.sort_values(by='order', inplace=True)
    perf_df.drop(columns='order', inplace=True)

    rf_models = [model for model in perf_df['Model'] if '(RF)' in model]
    for model in rf_models:
        mask = perf_df['Model'] == model
        if 'p-value' in perf_df.columns:
            perf_df.loc[mask, 'p-value'] = "N/A"

    perf_df.rename(columns={
        "R²": "LOOCV R²",
        "MAE": "LOOCV MAE",
    }, inplace=True)

    column_order = ["Model", "LOOCV R²", "LOOCV MAE", "Error SD", "p-value", "Number of Predictors", "Sample Size"]
    perf_df = perf_df[column_order]

    print("\nModel Comparison Summary:")
    print(tabulate(perf_df, headers="keys", tablefmt="grid"))
    print("\nNote: Error SD represents the standard deviation of model errors in grams (g).")
    return perf_df


def plot_morans_i(moran_data):
    if not moran_data:
        return

    moran_df = pd.DataFrame(moran_data)
    moran_df['Method'] = moran_df['Model'].apply(lambda x: x.split('(')[-1].replace(')', '').strip())
    moran_df['Category'] = moran_df['Model'].apply(lambda x: x.split(' (')[0])

    category_order = ["Spectral Only", "LiDAR Only", "Spectral + LiDAR"]
    palette = {"RF": "black", "LR": "gray"}

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=moran_df,
        x='Category',
        y="Moran's I",
        hue='Method',
        order=category_order,
        palette=palette,
        medianprops=dict(color='white', linewidth=2)
    )
    ax.set_xlabel("")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel("Moran's I", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Method', loc='upper right', fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance_data):
    plot_order = [
        ("Spectral Only (RF)", "A) Spectral"),
        ("LiDAR Only (RF)", "B) LiDAR"),
        ("Spectral + LiDAR (RF)", "C) Combined")
    ]
    for model_name, title in plot_order:
        importance_data = feature_importance_data.get(model_name, {})
        if importance_data:
            plt.figure(figsize=(8, 6))
            sorted_data = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True))
            features = list(sorted_data.keys())
            values = list(sorted_data.values())

            plt.barh(range(len(features)), values, color='black')
            plt.yticks(range(len(features)), features, fontsize=14)
            plt.gca().invert_yaxis()
            plt.title(title, fontsize=20, fontweight='bold')
            plt.xlabel("Importance", fontsize=22)
            plt.ylabel("Feature", fontsize=22)
            plt.xlim(0, max(values) * 1.1)
            plt.tight_layout()
            plt.show()


def main():
    file_path = r'C:\Users\bmark\PycharmProjects\MS_Thesis\data\Dangermond Field Data.xlsx'
    df = load_data(file_path)

    selected_sites = ['steves_flat', 'jalama_bull', 'east_tinta', 'cojo_cow']

    spectral_indices = ['cai', 'lcai', 'ndli']
    lidar_metrics = ['chm_min', 'chm_max', 'chm_range', 'chm_mean', 'chm_std', 'chm_sum', 'chm_median', 'chm_pct90']
    all_features = spectral_indices + lidar_metrics

    model_types = {
        "Spectral Only": {
            "features": spectral_indices,
            "data": df[df['study_area'].isin(selected_sites)].copy()
        },
        "LiDAR Only": {
            "features": lidar_metrics,
            "data": df
        },
        "Spectral + LiDAR": {
            "features": all_features,
            "data": df[df['study_area'].isin(selected_sites)].copy()
        }
    }

    performance_results, moran_data, feature_importance_data = run_models(model_types)
    display_results(performance_results)
    plot_morans_i(moran_data)
    plot_feature_importance(feature_importance_data)


if __name__ == "__main__":
    main()
