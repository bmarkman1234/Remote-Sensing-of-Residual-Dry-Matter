import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'/data/Dangermond_Preserve_Monthly_Precipitation_Averages.csv'
df = pd.read_csv(file_path)

# Define the correct month order
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Make sure 'month_name' is ordered correctly
df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)
df = df.sort_values('month_name')

# Create the plot with styling
plt.figure(figsize=(5, 3))
plt.bar(df['month_name'], df['precipitation_mm'], color='skyblue')

# Make axis labels bold and bigger
plt.xlabel('Month', fontweight='bold', fontsize=14)
plt.ylabel('Precipitation (mm)', fontweight='bold', fontsize=14)

# Make tick labels slightly larger
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()