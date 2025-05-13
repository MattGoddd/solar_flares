import pandas as pd
import os

def data_parser(csv_name: str):
    """
    Parses the CSV file to extract relevant data and save it to a new CSV file.
    
    Args:
        csv_path (str): Path to the input CSV file.
    """

    
    # Load the data from the CSV file
    output_dir = "data/sharp_csv"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_name)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # Add new column for classification
    df_filtered = df.copy()

    if 'y_value' not in df_filtered.columns:
        df_filtered = df_filtered.assign(y_value = 0)

    for index, row in df_filtered.iterrows():
        if type(row.flare_label) != str:
            row.flare_label = 'None'
        
        if row.flare_label == 'None':
            letter = ''
            magnitude = 0
        else:
            letter = row.flare_label[0]
            magnitude = float(row.flare_label[1:])
            
        #Ignore A class flares
        if letter == '' or letter == 'A':
            df_filtered.at[index, 'y_value'] = 0
            continue
        
        # If flare class is B, assign y_value as 1
        if letter == 'B':
            df_filtered.at[index, 'y_value'] = 1

        # If flare class is C and magnitude < 5.0, assign y_value as 2
        elif letter == 'C' and magnitude < 5.0:
            df_filtered.at[index, 'y_value'] = 2
        # If flare class is C and magnitude >= 5.0, assign y_value as 3
        elif letter == 'C' and magnitude >= 5.0:
            df_filtered.at[index, 'y_value'] = 3
        # If flare class is M and magnitude < 5.0, assign y_value as 4
        elif letter == 'M' and magnitude < 5.0:
            df_filtered.at[index, 'y_value'] = 4
        # If flare class is M and magnitude >= 5.0, assign y_value as 5
        elif letter == 'M' and magnitude >= 5.0:
            df_filtered.at[index, 'y_value'] = 5
        # If flare class is X, assign y_value as 6
        elif letter == 'X':
            df_filtered.at[index, 'y_value'] = 6
    
    valid_rows = ~df_filtered.isnull().any(axis=1)
    df_filtered = df_filtered[valid_rows]
    print(df_filtered.isnull().sum())  # Shows count of NaN or None per column
    print(df_filtered.isnull().any())  # Shows True/False per column
    
    df_filtered.to_csv(csv_path, index=False)

data_parser('testing_data.csv')
