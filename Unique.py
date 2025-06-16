import pandas as pd

df = pd.read_csv("crime_type_prediction_dataset_rajasthan.csv")

# List of columns to get unique values for
categorical_columns = ['district', 'crime_location_type', 'season', 'reported_by',
                       'victim_gender', 'accused_gender', 'weapon_used',
                       'vehicle_involved', 'relationship', 'day_of_week', 'crime_category']

for col in categorical_columns:
    unique_vals = df[col].dropna().unique()
    print(f"Unique values for {col}:")
    print(list(unique_vals))
    print("\n")
