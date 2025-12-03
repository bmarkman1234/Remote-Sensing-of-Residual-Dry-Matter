import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths
spectra_dir = r'/data/Dangermond_Spectra'
data_file_path = r'/data/Dangermond Field Data.xlsx'
output_directory = r'C:\Users\bmark\PycharmProjects\MS_Thesis\output\spectral_ranges_lcai'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Load metadata (oven_weight) from the Excel file
metadata_df = pd.read_excel(data_file_path, sheet_name=0)  # Update sheet name if necessary

# Ensure filenames are consistent for matching
metadata_df["spectra_file"] = metadata_df["spectra_file"].astype(str).str.strip()  # Remove extra spaces

# Function to get reflectance at the closest available wavelength
def get_reflectance_at_wavelength(data, target_wavelength):
    """
    Finds the closest available reflectance value for a given target wavelength.
    Assumes wavelength in column 0 and reflectance in column 3.
    """
    closest_index = (data.iloc[:, 0] - target_wavelength).abs().idxmin()
    return data.iloc[closest_index, 3]

# Dictionary of index wavelength centers, colors, and labels
index_details = {
    "CAI": {"wavelengths": [2000, 2100, 2200], "color": "#1E90FF"},  # Blue
    "LCAI": {"ranges": [(2185, 2225), (2200, 2240), (2250, 2290), (2295, 2335), (2340, 2380)], "color": "#FFD700"},  # Yellow
    "NDLI": {"wavelengths": [1754, 1680], "color": "#32CD32"}  # Green
}

# Get the list of spectral files
file_list = [f for f in os.listdir(spectra_dir) if f.endswith(".sig")]

# Store LCAI values across all files to determine the global y-axis range
all_lcai_values = []

# First pass: Compute global LCAI y-axis limits
for filename in file_list:
    file_path = os.path.join(spectra_dir, filename)
    try:
        data = pd.read_csv(file_path, delim_whitespace=True, skiprows=28, header=None)
    except Exception:
        continue  # Skip problematic files

    lcai_values = [
        (get_reflectance_at_wavelength(data, upper) - get_reflectance_at_wavelength(data, lower))
        for lower, upper in index_details["LCAI"]["ranges"]
    ]
    all_lcai_values.extend(lcai_values)

# Set the y-axis limits for LCAI histograms based on min/max LCAI values across all files
if all_lcai_values:
    lcai_y_min, lcai_y_max = min(all_lcai_values), max(all_lcai_values)
else:
    lcai_y_min, lcai_y_max = -0.1, 0.1  # Default if no data is found

# ---------------------------------------
# Function to plot spectrum and LCAI histogram
# ---------------------------------------
def plot_spectrum_with_indices(data, filename, biomass_value, lcai_values, lcai_labels):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # Spectrum Plot (Fixed Axes)
    axes[0].plot(data.iloc[:, 0], data.iloc[:, 3], color='black', label='Reflectance')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Reflectance')
    axes[0].set_title(f'Spectrum for {filename} | RDM: {biomass_value:.2f} g')

    # Set fixed y-axis range for reflectance
    axes[0].set_ylim(-5, 100)  # Adjust this range if needed

    # Highlight index regions
    for index, details in index_details.items():
        if "wavelengths" in details:
            for wl in details["wavelengths"]:
                axes[0].axvline(wl, color=details["color"], linestyle='--', alpha=0.8)
        elif "ranges" in details:
            for lower, upper in details["ranges"]:
                axes[0].axvspan(lower, upper, color=details["color"], alpha=0.3)

    # Add color-coded legend
    legend_patches = [plt.Line2D([0], [0], color=details["color"], lw=4, label=index) for index, details in index_details.items()]
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=10)

    # LCAI Histogram (Fixed Axes)
    axes[1].bar(lcai_labels, lcai_values, color=index_details["LCAI"]["color"])
    axes[1].set_xlabel("LCAI Wavelength Range (nm)")
    axes[1].set_ylabel("LCAI Index Value")
    axes[1].set_title("LCAI Index Across Wavelength Ranges")

    # Set fixed y-axis limits for LCAI
    axes[1].set_ylim(lcai_y_min, lcai_y_max)

    # Annotate each bar with its value
    for bar, value in zip(axes[1].patches, lcai_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    # Save the figure
    output_filepath = os.path.join(output_directory, f"{filename}.png")
    plt.savefig(output_filepath, dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"Saved figure: {output_filepath}")

# ---------------------------------------
# Process each .sig file
# ---------------------------------------
def process_file(filename):
    file_path = os.path.join(spectra_dir, filename)
    print(f"Processing file: {filename}")

    try:
        data = pd.read_csv(file_path, delim_whitespace=True, skiprows=28, header=None)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return

    # Ensure correct filename matching (strip extension if needed)
    filename_no_ext = os.path.splitext(filename)[0]  # Remove .sig extension
    biomass_value = metadata_df.loc[metadata_df["spectra_file"].str.contains(filename_no_ext, na=False, case=False), "oven_weight"].values

    # Handle cases where no biomass match is found
    biomass_value = biomass_value[0] if len(biomass_value) > 0 else np.nan

    # Compute LCAI values for each wavelength range
    lcai_labels = [f"{lower}-{upper}" for lower, upper in index_details["LCAI"]["ranges"]]  # Label each LCAI interval
    lcai_values = [
        (get_reflectance_at_wavelength(data, upper) - get_reflectance_at_wavelength(data, lower))
        for lower, upper in index_details["LCAI"]["ranges"]
    ]

    # Plot results and save the figure
    plot_spectrum_with_indices(data, filename, biomass_value, lcai_values, lcai_labels)

# Process all files sequentially
for filename in file_list:
    process_file(filename)
