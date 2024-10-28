import pandas as pd

# Load the CSV file
df = pd.read_csv('combined_data2.csv')

# Shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame to a new CSV file
df.to_csv('combined_data2.csv', index=False)