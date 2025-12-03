# =============================================================================
# Spectral Reflectance Visualization for Residual Dry Matter (RDM) Field Data
#
# This script reads field metadata from an Excel spreadsheet and matches each
# sampled vegetation plot to its corresponding spectral reflectance data file.
# For each plot with both a spectra file and oven-dried biomass measurement:
#   - Loads the reflectance data, masks out noisy wavelength regions,
#   - Plots the reflectance spectrum, colored by oven-dried biomass (RDM, g),
#   - Highlights key spectral indices (LCAI, NDLI, CAI) for interpretation,
#   - Customizes the plot with labels, legends, and colorbar.
#
# Usage:
#   - Update 'spectra_dir' and 'excel_file_path' as needed for data locations.
#   - Run to generate a figure visualizing spectral signatures by biomass.
#
# Dependencies: pandas, matplotlib

# Author: Bruce Markman
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

spectra_dir = r'C:\Users\bmark\PycharmProjects\Dangermond\data\Dangermond_Spectra'  
excel_file_path = r'/data/Dangermond Field Data.xlsx'  


field_data = pd.read_excel(excel_file_path)
field_data_filtered = field_data.dropna(subset=['spectra_file', 'oven_weight'])

oven_weights = field_data_filtered['oven_weight'].values
norm = Normalize(vmin=min(oven_weights), vmax=max(oven_weights))
colormap = plt.colormaps["coolwarm"]

plt.figure(figsize=(12, 8))

for _, row in field_data_filtered.iterrows():
    spectra_filename = row['spectra_file']
    if not spectra_filename.endswith('.sig'):
        spectra_filename += '.sig' 
        
    oven_weight = row['oven_weight']
    file_path = os.path.join(spectra_dir, spectra_filename)

    if spectra_filename == excluded_file:
        continue

    if os.path.isfile(file_path):
        try:
            data = pd.read_csv(file_path, delim_whitespace=True, skiprows=27, header=None)
            wavelength = data.iloc[:, 0].values  
            reflectance = data.iloc[:, 3].values  

            mask = (wavelength < 1850) | ((wavelength > 1950) & (wavelength < 2450))
            wavelength = wavelength[mask]
            reflectance = reflectance[mask]

            color = colormap(norm(oven_weight))
            plt.plot(wavelength, reflectance, color=color)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

plt.gca().add_patch(plt.Rectangle((1850, 0), 100, 100, color='white', zorder=10))
plt.gca().add_patch(plt.Rectangle((2450, 0), 100, 100, color='white', zorder=10))

plt.axvspan(2100, 2300, color='orange', alpha=0.3, label="LCAI")
plt.axvspan(1680, 1754, color='green', alpha=0.3, label="NDLI")
plt.axvspan(2000, 2200, color='blue', alpha=0.3, label="CAI")

sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('RDM (g)', fontsize=30)
cbar.ax.tick_params(labelsize=18)

plt.xlim(0, 2500) 
plt.ylim(0, 100)   
plt.xlabel("Wavelength (nm)", fontsize=40, labelpad=20)
plt.ylabel("Reflectance (%)", fontsize=40, labelpad=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=30, title_fontsize=40)
plt.tight_layout()
plt.show()



