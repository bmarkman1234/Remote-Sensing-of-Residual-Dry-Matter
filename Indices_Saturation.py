import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load dataset
file_path = r'/data/Dangermond Field Data.xlsx'
df = pd.read_excel(file_path)

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# Filter dataset for specified study areas
selected_sites = ['steves_flat', 'jalama_bull', 'east_tinta', 'cojo_cow']
filtered_df = df[df['study_area'].isin(selected_sites)].copy()

# Define spectral indices and biomass variable
indices = ['cai', 'lcai', 'ndli']
biomass = 'oven_weight'

# Convert columns to numeric
filtered_df[indices] = filtered_df[indices].apply(pd.to_numeric, errors='coerce')
filtered_df[biomass] = pd.to_numeric(filtered_df[biomass], errors='coerce')

# Define colors for each index
colors = {"ndli": "green", "cai": "blue", "lcai": "gold"}

sns.set(style="whitegrid")
fig, axs = plt.subplots(1, len(indices), figsize=(18, 6), sharey=True)

for ax, index in zip(axs, indices):
    # Scatter plot of the raw data
    sns.scatterplot(data=filtered_df, x=biomass, y=index,
                    ax=ax, color=colors.get(index, 'black'))

    # Add subplot title with the index name
    ax.set_title(index.upper(), fontsize=24)

    ax.set_xlabel("RDM (g)", fontsize=20)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, linestyle='--', alpha=0.5)

# Set the shared y-axis label with increased font size
axs[0].set_ylabel("Spectral Index Value", fontsize=20)

plt.tight_layout()
plt.show()
