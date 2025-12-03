import pandas as pd

# File path to the SSURGO data
ssurgo_data = r'C:\Users\bmark\PycharmProjects\MS_Thesis\data\ssurgo_plots.csv'

# Load the CSV data into a DataFrame
df = pd.read_csv(ssurgo_data)

# List of relevant columns to keep, ensuring they match the SSURGO variables
relevant_columns = [
    "plot_number", "study_area", "Long", "Lat", "muname",  # Include soil unit name
    "aws0150wta",  # Available Water Storage (0-150 cm)
    "rootznaws",  # Root Zone Available Water Storage
    "hydgrpdcd",  # Hydrologic Soil Group
    "kffact",  # Soil Erodibility Factor
    "slopegradw",  # Slope Gradient
    "brockdepmi",  # Bedrock Depth (Minimum)
    "nccpi3sg"  # National Commodity Crop Productivity Index (Small Grains)
]

# Filter the DataFrame to include only the relevant columns
filtered_data = df[relevant_columns]

# Export the filtered data to an Excel file
output_path_excel = r"/output/ssurgo_cleaned.xlsx"
filtered_data.to_excel(output_path_excel, index=False)

# Display the first few rows of the filtered data
print(filtered_data.head())
