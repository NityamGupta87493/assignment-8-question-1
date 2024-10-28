import os
import pandas as pd

# Define the directory where your text files are stored
directory_path = r"C:\Users\Subhas Kr Gupta\OneDrive\Desktop\assignment-8\train-mails"

# Initialize a list to store file data
data = []

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        
        # Read the entire file content as a single string
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Determine if the file is spam based on the filename
        is_spam = 1 if filename.startswith("sp") else 0
        
        # Append the filename, content, and spam indicator as a single row to the list
        data.append({'content': content, 'spam': is_spam})

# Convert list to a DataFrame
df = pd.DataFrame(data)

# Save the combined data to a single CSV file
output_file = 'combined_data2.csv'
df.to_csv(output_file, index=False)

print(f"Combined data saved to {output_file}")
