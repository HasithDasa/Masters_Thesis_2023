import pandas as pd

# Load the Excel files into pandas DataFrames
df1 = pd.read_excel('D:/Academic/MSc/Thesis/Project files/New folder/ES_1.xlsx')  # Assuming 'ES_1.xlsx' is the file name of the first Excel file
df2 = pd.read_excel('D:/Academic/MSc/Thesis/Project files/New folder/ES_2.xlsx')  # Assuming 'ES_2.xlsx' is the file name of the second Excel file

# Ensure the 'image name' column is of string type for accurate matching
df1['image_name'] = df1['image_name'].astype(str)
df2['image_name'] = df2['image_name'].astype(str)

# Create a new column in df2 for the '261_264' values from df1
df2['260_270_copy'] = ''

# Loop through each row in df2 to find matching 'image name' in df1 and copy '261_264' value
for index, row in df2.iterrows():
    image_name = row['image_name']
    match = df1[df1['image_name'] == image_name]['260_270']
    if not match.empty:
        df2.at[index, '260_270_copy'] = match.values[0]

# Save the modified df2 back to an Excel file
df2.to_excel('D:/Academic/MSc/Thesis/Project files/New folder/ES_2.xlsx', index=False)

