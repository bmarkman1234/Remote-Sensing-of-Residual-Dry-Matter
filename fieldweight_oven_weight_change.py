import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import Your Data ---
file_path = r'C:\Users\bmark\PycharmProjects\MS_Thesis\data\Dangermond Field Data.xlsx'
data = pd.read_excel(file_path)

# Check required columns
required_cols = {'study_area', 'field_weight', 'oven_weight'}
if not required_cols.issubset(data.columns):
    raise ValueError(f"The Excel file must contain the columns: {required_cols}")

# Transform study_area names: remove underscores and capitalize words
data['study_area'] = data['study_area'].str.replace('_', ' ').str.title()

# Fix specific name
data['study_area'] = data['study_area'].replace({'Steves Flat': "Steve's Flat"})

# --- Compute Changes in Mass ---
# Absolute change: Field Weight - Oven Weight (grams)
data['difference'] = data['field_weight'] - data['oven_weight']
# Percent change relative to Field Weight
data['percent_change'] = (data['difference'] / data['field_weight']) * 100

# --- Create the Boxplot ---
# Reshape data for boxplots using melt
data_melted = pd.melt(data,
                      id_vars=['study_area'],
                      value_vars=['field_weight', 'oven_weight'],
                      var_name='Measurement',
                      value_name='Mass')

# Rename for aesthetics
data_melted['Measurement'] = data_melted['Measurement'].replace({
    'field_weight': 'Field Weight',
    'oven_weight': 'Oven Weight'
})

# Define palette: Oven Weight in black, Field Weight in gray
palette = {"Field Weight": "gray", "Oven Weight": "black"}

plt.figure(figsize=(12, 8))
sns.boxplot(x='study_area', y='Mass', hue='Measurement', data=data_melted,
            palette=palette, medianprops={'color': 'white', 'linewidth': 2})

# Increase label sizes and add consistent label spacing for both axes
plt.xlabel("Study Area", fontsize=20, labelpad=20)
plt.ylabel("RDM (g)", fontsize=20, labelpad=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Remove title (optional)
# plt.title("Field Weight vs Oven Weight by Study Area", fontsize=22)

# Customize legend with larger fonts
plt.legend(title="Measurement", loc='upper right', fontsize=18, title_fontsize=20)

plt.tight_layout()
plt.show()

# --- Print Summary Statistics for Changes ---
print("Summary Statistics for Change in Mass (Field Weight - Oven Weight) [grams]:")
print(data['difference'].describe(), "\n")

print("Summary Statistics for Percent Change (% of Field Weight):")
print(data['percent_change'].describe(), "\n")
