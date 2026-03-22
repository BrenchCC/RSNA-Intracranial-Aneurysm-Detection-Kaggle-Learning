import pandas as pd
import os

# Read the patch metadata
patch_metadata = pd.read_csv('data/processed_data/patch_metadata.csv')

# Read the train labels
train_labels = pd.read_csv('data/train.csv')

# Merge the dataframes on series_uid
merged_df = pd.merge(
    patch_metadata,
    train_labels,
    left_on='series_uid',
    right_on='SeriesInstanceUID',
    how='inner'
)

# Create nifti_path by combining the directory with nifti_file
merged_df['nifti_path'] = merged_df['nifti_file'].apply(lambda x: os.path.join('data/processed_data', x))

# Define the columns to keep
columns_to_keep = [
    'series_uid',
    'nifti_path',
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

# Select only the required columns
result_df = merged_df[columns_to_keep]

# Write the result to the output file
output_path = 'data/processed_data/train_labels_14class.csv'
result_df.to_csv(output_path, index=False)

print(f"Generated {output_path} with {len(result_df)} rows")
print("Columns:")
print(result_df.dtypes)