import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = r'/data/Dangermond Field Data.xlsx'
data = pd.read_excel(file_path)

# Rename columns for easier access
data = data.rename(columns={
    'study_area': 'Study Area',
    'field_weight': 'RDM (Field Weight in grams)',
})

# Convert field weight to lbs/acre by multiplying by 10
data['RDM (lbs/acre)'] = data['RDM (Field Weight in grams)'] * 10

# Set up the plot canvas
plt.figure(figsize=(14, 8))

# Create the box and whisker plot with hollow boxes
sns.boxplot(
    data=data, 
    x='Study Area', 
    y='RDM (lbs/acre)', 
    showcaps=True,  # Show the top and bottom caps
    boxprops={'facecolor': 'none', 'edgecolor': 'black'},  # Make box hollow
    whiskerprops={'color': 'black'},  # Whisker color
    medianprops={'color': 'black', 'linewidth': 2},  # Median line style
    capprops={'color': 'black'},  # Cap line color
)

# Formatting the plot
plt.title('Jack & Laura Dangermond Preserve RDM Distribution', fontsize=15)
plt.xlabel('Study Area', fontsize=20)
plt.ylabel('RDM (Field Weight in lbs/acre)', fontsize=20)
plt.xticks(rotation=45, fontsize=15)
plt.tight_layout()

# Display the plot
plt.show()
