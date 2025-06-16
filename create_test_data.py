import pandas as pd

# Load your original dataset (the one with no missing values)
df = pd.read_csv("imputed_relationship_fir.csv")  # or whatever your original file is

# Randomly mask 10% of the 'relationship' column to simulate missing values
df.loc[df.sample(frac=0.1, random_state=42).index, 'relationship'] = None

# Save the new version
df.to_csv("test_missing_relationship.csv", index=False)

print("Test file with 10% missing 'relationship' values created as 'test_missing_relationship.csv'")
