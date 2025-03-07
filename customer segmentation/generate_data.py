import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)  # For reproducibility

# Create dataset with more than 10 values
data = {
    "CustomerID": range(1, 16),  # 15 customers
    "TotalSpend": np.random.randint(100, 1000, 15),  # Random spending amounts
    "PurchaseFrequency": np.random.randint(1, 50, 15),  # Random purchase frequency
    "AverageOrderValue": np.random.uniform(20, 200, 15).round(2),  # Random AOV
    "CategoryPreference": np.random.choice(
        ["Electronics", "Clothing", "Groceries", "Beauty"], 15
    ),  # Random category
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Introduce null values randomly in "TotalSpend" and "PurchaseFrequency"
df.loc[np.random.choice(df.index, 3, replace=False), "TotalSpend"] = np.nan
df.loc[np.random.choice(df.index, 2, replace=False), "PurchaseFrequency"] = np.nan

# Save dataset as a CSV file
df.to_csv("customer_data.csv", index=False)

# Print the dataset
print("Generated Dataset with Null Values:")
print(df)
