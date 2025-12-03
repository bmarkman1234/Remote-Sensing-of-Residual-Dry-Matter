import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------- Data Loading -----------------------
# Update the file path to the location of your Excel file
file_path = r'/data/Dangermond Field Data.xlsx'
df = pd.read_excel(file_path)

# ----------------------- Define Predictor Groups -----------------------
# Spectral predictors
indices = ['cai', 'lcai', 'ndli']

# LiDAR predictors
lidar_metrics = ['chm_min', 'chm_max', 'chm_range', 'chm_mean', 'chm_std', 'chm_sum', 'chm_median', 'chm_pct90']

# ----------------------- Create Heatmap for LiDAR Predictors -----------------------
lidar_corr = df[lidar_metrics].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(lidar_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap: LiDAR CHM Predictors")
plt.tight_layout()
plt.show()

# ----------------------- Create Heatmap for Spectral Predictors -----------------------
spectral_corr = df[indices].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(spectral_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap: Spectral Index Predictors")
plt.tight_layout()
plt.show()
