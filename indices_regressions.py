import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import os


def load_and_preprocess_data(file_path, features_to_use, target_column, sites=None):
    """Load and preprocess the dataset."""
    # Load data
    df = pd.read_excel(file_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Print column names for debugging
    print("Columns in dataframe:")
    print(df.columns.tolist())

    # Drop rows with missing values
    df = df.dropna(subset=features_to_use + [target_column, 'spectra_file'])

    # Filter by site if specified and if site column exists
    if sites is not None and 'site' in df.columns:
        df = df[df['site'].str.lower().isin(sites)]

    return df


def read_spectrum(file_path):
    """Read spectral data from .sig file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()[27:]  # Skip header
        reflectance = [float(line.split()[3]) for line in lines if len(line.split()) >= 4]
    return reflectance


def evaluate_model(actual, predicted):
    """Calculate and return evaluation metrics."""
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return r2, rmse


def plot_results(actual_values, prediction_dict, title, diagonal_line=True):
    """Plot actual vs predicted values with metrics."""
    plt.figure(figsize=(9, 9))

    colors = {
        'Linear Regression': 'dodgerblue',
        'Random Forest': 'forestgreen',
        'PLSR (1 comp)': 'darkorange',
        'PLSR (2 comp)': 'firebrick',
        'PLSR (3 comp)': 'purple'
    }

    for model_name, predictions in prediction_dict.items():
        r2, rmse = evaluate_model(actual_values, predictions)
        plt.scatter(actual_values, predictions, alpha=0.6, color=colors.get(model_name, 'gray'),
                    label=f'{model_name}\nR²={r2:.2f}, RMSE={rmse:.2f}')

    if diagonal_line:
        plt.plot([min(actual_values), max(actual_values)],
                 [min(actual_values), max(actual_values)], 'k--', linewidth=1)

    plt.xlabel("Actual RDM (g)")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_loocv(X, y, models):
    """Run Leave-One-Out Cross-Validation for multiple models."""
    loo = LeaveOneOut()
    actual = []
    predictions = {name: [] for name in models.keys()}

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            predictions[name].append(pred)

        actual.append(y_test[0])

    return actual, predictions


def analyze_indices_only(df, features, title, target='oven_weight'):
    """Run LOOCV for index-based models."""
    X = df[features].values
    y = df[target].values

    # Scale the indices data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    actual, predictions = run_loocv(X_scaled, y, models)
    plot_results(actual, predictions, f"Predictions with {title}")


def analyze_full_spectra(df, spectra_folder, target='oven_weight', spectra_col='spectra_file'):
    """Run LOOCV for full-spectrum PLSR models."""
    y = df[target].values
    file_paths = [os.path.join(spectra_folder, f"{name}.sig") for name in df[spectra_col]]

    # Read all spectra
    spectra_list = [read_spectrum(path) for path in file_paths]

    # Find most common length
    lengths = [len(s) for s in spectra_list]
    from collections import Counter
    most_common_length = Counter(lengths).most_common(1)[0][0]

    # Filter to spectra with common length
    valid_idx = [i for i, length in enumerate(lengths) if length == most_common_length]
    X = np.array([spectra_list[i] for i in valid_idx])
    y = y[valid_idx]

    # LOOCV for PLSR models
    loo = LeaveOneOut()
    actual = []
    pls_preds = {f'PLSR ({n} comp)': [] for n in [1, 2, 3]}

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for n in [1, 2, 3]:
            pls = PLSRegression(n_components=min(n, X.shape[1]))
            pls.fit(X_train, y_train)
            pls_preds[f'PLSR ({n} comp)'].append(pls.predict(X_test)[0][0])

        actual.append(y_test[0])

    plot_results(actual, pls_preds, "PLSR (Reflectance)")


def analyze_pls_components(df, spectra_folder, max_components=50, target='oven_weight', spectra_col='spectra_file'):
    """Analyze how R² changes with increasing number of PLS components."""
    y = df[target].values
    file_paths = [os.path.join(spectra_folder, f"{name}.sig") for name in df[spectra_col]]

    # Read spectral data
    spectra_list = [read_spectrum(path) for path in file_paths]

    # Find most common length
    lengths = [len(s) for s in spectra_list]
    from collections import Counter
    most_common_length = Counter(lengths).most_common(1)[0][0]

    # Filter to spectra with common length
    valid_idx = [i for i, length in enumerate(lengths) if length == most_common_length]
    X = np.array([spectra_list[i] for i in valid_idx])
    y = y[valid_idx]

    # Determine maximum possible components
    max_components = min(max_components, X.shape[1], X.shape[0] - 1)
    components_range = range(1, max_components + 1)

    # Lists to store results
    r2_scores = []

    # For each number of components, perform LOOCV
    for n_comp in components_range:
        loo = LeaveOneOut()
        y_true = []
        y_pred = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit PLS model with current number of components
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_train, y_train)

            # Predict and store results
            pred = pls.predict(X_test)[0][0]
            y_true.append(y_test[0])
            y_pred.append(pred)

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        r2_scores.append(r2)

    # Plot R² vs number of components
    plt.figure(figsize=(10, 6))
    plt.plot(components_range, r2_scores, 'o-', color='blue', linewidth=2)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('R²')
    plt.title('R² vs Number of Dimensions')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return components_range, r2_scores


def main():
    # Define paths and parameters
    file_path = r'/data/Dangermond Field Data.xlsx'
    spectra_folder = r"C:\Users\bmark\PycharmProjects\MS_Thesis\data\Dangermond_Spectra"

    # Define feature sets
    features_basic = ['cai', 'lcai', 'ndli']
    features_extended = ['cai', 'lcai', 'ndli', 'ndvi', 'gcc', 'rcc']
    target = 'oven_weight'

    # Load and preprocess data
    df = load_and_preprocess_data(file_path, features_extended, target)
    print("Data loaded successfully")

    # Run analyses
    analyze_indices_only(df, features_basic, "NPV Indices (CAI, LCAI, NDLI)")
    analyze_indices_only(df, features_extended, "All Indices (CAI, LCAI, NDLI, NDVI, GCC, RCC)")
    analyze_full_spectra(df, spectra_folder)

    # Run the new PLS component analysis with 50 components
    analyze_pls_components(df, spectra_folder, max_components=50)


if __name__ == "__main__":
    main()