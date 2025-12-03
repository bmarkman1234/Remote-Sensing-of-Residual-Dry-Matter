import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Define paths
spectra_dir = r'C:\Users\bmark\PycharmProjects\Dangermond\data\Dangermond_Spectra'  # Path to spectra files
excel_file_path = r'/data/Dangermond Field Data.xlsx'  # Path to Excel file

# Load the Excel file and filter to only include rows with relevant data
field_data = pd.read_excel(excel_file_path)
field_data_filtered = field_data.dropna(subset=['spectra_file', 'oven_weight'])

# Create a color map based on oven weight
oven_weights = field_data_filtered['oven_weight'].values
norm = Normalize(vmin=min(oven_weights), vmax=max(oven_weights))
colormap = plt.colormaps["coolwarm"]

# Exclude specific file
excluded_file = "gr092324_0011.sig"

# Plot all spectra on the same plot
plt.figure(figsize=(12, 8))

for _, row in field_data_filtered.iterrows():
    spectra_filename = row['spectra_file']
    if not spectra_filename.endswith('.sig'):
        spectra_filename += '.sig'  # Add .sig extension if missing

    oven_weight = row['oven_weight']
    file_path = os.path.join(spectra_dir, spectra_filename)

    # Exclude specific file
    if spectra_filename == excluded_file:
        continue

    # Check if the file exists, then load and plot
    if os.path.isfile(file_path):
        try:
            # Load the spectrum data
            data = pd.read_csv(file_path, delim_whitespace=True, skiprows=27, header=None)
            wavelength = data.iloc[:, 0].values  # Wavelength in nm
            reflectance = data.iloc[:, 3].values  # Reflectance values (in %)

            # Mask noisy regions
            mask = (wavelength < 1850) | ((wavelength > 1950) & (wavelength < 2450))
            wavelength = wavelength[mask]
            reflectance = reflectance[mask]

            # Plot with color based on oven weight
            color = colormap(norm(oven_weight))
            plt.plot(wavelength, reflectance, color=color)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

# Add white rectangles to mask excluded regions
plt.gca().add_patch(plt.Rectangle((1850, 0), 100, 100, color='white', zorder=10))
plt.gca().add_patch(plt.Rectangle((2450, 0), 100, 100, color='white', zorder=10))

# Add shaded regions for indices with abbreviated labels
plt.axvspan(2100, 2300, color='orange', alpha=0.3, label="LCAI")
plt.axvspan(1680, 1754, color='green', alpha=0.3, label="NDLI")
plt.axvspan(2000, 2200, color='blue', alpha=0.3, label="CAI")

# Add color bar to show oven weight with increased font sizes for label and ticks
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('RDM (g)', fontsize=30)
cbar.ax.tick_params(labelsize=18)

# Set axis limits and labels with same label spacing for both axes using labelpad
plt.xlim(0, 2500)  # Wavelength range
plt.ylim(0, 100)   # Reflectance range in %
plt.xlabel("Wavelength (nm)", fontsize=40, labelpad=20)
plt.ylabel("Reflectance (%)", fontsize=40, labelpad=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=30, title_fontsize=40)
plt.tight_layout()
plt.show()
