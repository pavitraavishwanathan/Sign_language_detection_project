import os
import pandas as pd

# -----------------------------
# Folder containing all CSV files
# Each CSV should represent one class (letter/word)
# -----------------------------
csv_folder = '/Users/pavitraa/Desktop/SIGNLANG_V2/csv_data/'  # change if needed
output_file = os.path.join(csv_folder, 'ALL_DATA.csv')

# -----------------------------
# Initialize empty list to store dataframes
# -----------------------------
all_dfs = []

# -----------------------------
# Iterate through CSV files in the folder
# -----------------------------
for file_name in os.listdir(csv_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(csv_folder, file_name)
        df = pd.read_csv(file_path)
        
        # Optional: automatically assign label from filename if not present
        if 'label' not in df.columns:
            # Take filename without extension as label
            label = os.path.splitext(file_name)[0]
            df['label'] = label
        
        all_dfs.append(df)

# -----------------------------
# Combine all dataframes
# -----------------------------
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"✅ Combined dataset saved as {output_file}")
    print("Class counts:")
    print(combined_df['label'].value_counts())
else:
    print("❌ No CSV files found in the folder!")
