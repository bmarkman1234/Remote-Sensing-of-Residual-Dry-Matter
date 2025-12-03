import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the directory containing all spectra files
spectra_dir = r'/data/Dangermond_Spectra'

# List to store results
results = []

# Define a helper function to find the nearest wavelength index
def find_index(df, target_wavelength):
    return (df[0] - target_wavelength).abs().idxmin()

# Define index calculation functions
def calculate_indices(file_path):
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=28, header=None)

        # Find reflectance values
        refl_2000 = df.iloc[find_index(df, 2000), 3]
        refl_2100 = df.iloc[find_index(df, 2100), 3]
        refl_2200 = df.iloc[find_index(df, 2200), 3]
        refl_2165 = df.iloc[find_index(df, 2165), 3]
        refl_2205 = df.iloc[find_index(df, 2205), 3]
        refl_2330 = df.iloc[find_index(df, 2330), 3]
        refl_1754 = df.iloc[find_index(df, 1754), 3]
        refl_1680 = df.iloc[find_index(df, 1680), 3]
        refl_nir = df.iloc[find_index(df, 800), 3]
        refl_red = df.iloc[find_index(df, 680), 3]
        refl_green = df.iloc[find_index(df, 550), 3]
        refl_blue = df.iloc[find_index(df, 470), 3]

        # Calculate indices
        cai = 0.5 * (refl_2000 + refl_2200) - refl_2100
        lcai = (refl_2205 - refl_2165) + (refl_2205 - refl_2330)
        ndli = (np.log(1 / refl_1754) - np.log(1 / refl_1680)) / (np.log(1 / refl_1754) + np.log(1 / refl_1680)) * 1000
        ndvi = (refl_nir - refl_red) / (refl_nir + refl_red)
        gcc = refl_green / (refl_red + refl_green + refl_blue)
        rcc = refl_red / (refl_red + refl_green + refl_blue)

        return cai, lcai, ndli, ndvi, gcc, rcc

    except Exception as e:
        print(f"Error calculating indices for {file_path}: {e}")
        return (np.nan,) * 6

# Loop through each .sig file in the directory
for filename in os.listdir(spectra_dir):
    if filename.endswith(".sig"):
        file_path = os.path.join(spectra_dir, filename)
        print(f"Processing file: {filename}")

        # Calculate indices
        cai, lcai, ndli, ndvi, gcc, rcc = calculate_indices(file_path)

        # Append results to the list
        results.append({
            'File Name': filename,
            'CAI': cai,
            'LCAI': lcai,
            'NDLI': ndli,
            'NDVI': ndvi,
            'GCC': gcc,
            'RCC': rcc
        })

        # Plot the spectra
        try:
            spectra_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=28, header=None)
            wavelength = spectra_data.iloc[:, 0]
            reflectance = spectra_data.iloc[:, 3]

            plt.figure(figsize=(10, 6))
            plt.plot(wavelength, reflectance, label=filename)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.title(f'Spectra for {filename}')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error plotting spectra for {file_path}: {e}")

# Convert results to a DataFrame and save to Excel
results_df = pd.DataFrame(results)
output_file_path = r'/output/indices.xlsx'
results_df.to_excel(output_file_path, index=False)
print(f"Results saved to {output_file_path}")
