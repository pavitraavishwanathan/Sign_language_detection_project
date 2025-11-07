import pandas as pd

# Correct path to your dataset
data_path = '/Users/pavitraa/Desktop/SIGNLANG_V2/csv_data/ALL_DATA.csv'

# Load dataset
data = pd.read_csv(data_path)

# Detect label column
if 'label' not in data.columns:
    label_col = data.columns[-1]  # assume last column
else:
    label_col = 'label'

# Count samples per class
class_counts = data[label_col].value_counts().sort_index()
print("Class counts in the dataset:")
print(class_counts)

threshold = class_counts.mean() * 0.5
underrepresented = class_counts[class_counts < threshold]

if not underrepresented.empty:
    print("\n⚠️ Underrepresented classes:")
    print(underrepresented)
else:
    print("\n✅ All classes have reasonable representation.")
