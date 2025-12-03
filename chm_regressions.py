import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r'/data/Dangermond Field Data.xlsx'
df = pd.read_excel(file_path)

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# Exclude the outlier with plot_number 1 and study_area "jalama mare"
if 'plot_number' in df.columns:
    df = df[~((df['plot_number'] == 1) & (df['study_area'].str.lower() == 'jalama_mare'))]

# Define the CHM columns and target variable
chm_cols = ['chm_max', 'chm_range', 'chm_mean', 'chm_std', 'chm_median', 'chm_pct90']
y_var = 'oven_weight'

# Convert columns to numeric in case of issues
df[chm_cols] = df[chm_cols].apply(pd.to_numeric, errors='coerce')
df[y_var] = pd.to_numeric(df[y_var], errors='coerce')

sns.set(style="whitegrid")

# Create a 2x3 grid of subplots with a large figure size for readability
fig, axs = plt.subplots(2, 3, figsize=(24, 12))
axs = axs.flatten()  # Flatten the array for easy iteration

# Loop over each CHM column and create a scatter plot
for ax, col in zip(axs, chm_cols):
    sns.scatterplot(data=df, x=col, y=y_var, ax=ax, color='black')
    ax.set_xlabel(col, fontsize=30)                     # Increase x-axis label size
    ax.set_ylabel("RDM (g)", fontsize=30)               # Increase y-axis label size
    ax.tick_params(axis='both', labelsize=18)           # Increase tick label size
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
