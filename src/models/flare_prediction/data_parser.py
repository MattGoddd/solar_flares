import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
import os

def data_classifier(csv_name: str):
    """
        Classifies the data based on the flare intensity (if any) and save it to the existing CSV file.
    
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
    else: 
        df_filtered['y_value'] = 0

    for index, row in df_filtered.iterrows():
        if type(row.flare_label) != str or row.flare_label == '':
            row.flare_label = 'None'
            df_filtered.at[index, 'flare_label'] = 'None'
            df_filtered.at[index, 'y_value'] = 0
            continue
        
        letter = row.flare_label[0]
        magnitude = float(row.flare_label[1:])
            
        #Ignore A class flares
        if letter == '' or letter == 'A' or letter == 'B':
            df_filtered.at[index, 'y_value'] = 0
            continue
    
        # If flare class is C and magnitude < 5.0, assign y_value as 1
        elif letter == 'C' and magnitude < 5.0:
            df_filtered.at[index, 'y_value'] = 1
        # If flare class is C and magnitude >= 5.0, assign y_value as 2
        elif letter == 'C' and magnitude >= 5.0:
            df_filtered.at[index, 'y_value'] = 2
        # If flare class is M and magnitude < 5.0, assign y_value as 3
        elif letter == 'M' and magnitude < 5.0:
            df_filtered.at[index, 'y_value'] = 3
        # If flare class is M and magnitude >= 5.0, assign y_value as 4
        elif letter == 'M' and magnitude >= 5.0:
            df_filtered.at[index, 'y_value'] = 4
        # If flare class is X, assign y_value as 5
        elif letter == 'X':
            df_filtered.at[index, 'y_value'] = 5
    
    valid_rows = ~df_filtered.isnull().any(axis=1)
    df_filtered = df_filtered[valid_rows]
    label_counts = df_filtered['y_value'].value_counts().sort_index()
    print(label_counts)
    
    df_filtered.to_csv(csv_path, index=False)

def data_cleaner(csv_name: str):
    """
    Cleans the data by removing rows with NaN values and saves it to the existing CSV file.
    
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

    has_nan = df.isnull().values.any()
    print(f"Data contains NaN values: {has_nan}")
    
    if not has_nan:
        return

    nan_columns = df.columns[df.isnull().any()].tolist()
    nan_columns.remove('flare_label') if 'flare_label' in nan_columns else None
    print(f"Columns with NaN values: {nan_columns}")

    df_filtered = df.copy()

    df_filtered = df_filtered.dropna(subset=nan_columns)

    df_filtered = df_filtered.drop(columns = ['T_REC', 'NOAA_AR'])

    # Save the cleaned DataFrame to a new CSV file
    csv_path = os.path.join(output_dir, csv_name)
    df_filtered.to_csv(csv_path, index=False)

def data_balancer(csv_name: str):
    """
        Balances the data by implementing SMOTE for the minority class(es) and saves it to a new CSV file.
    
    Args:
        csv_path (str): Path to the input CSV file.
    """

    if "test" in csv_name.lower():
        data_type = "testing"
    else:
        data_type = "training"
    print(f"Type of file: {data_type}")
    
    # Load the data from the CSV file
    output_dir = "data/sharp_csv"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_name)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    df_filtered = df.copy()
    print(df_filtered.columns)

    Y_train = df_filtered['y_value'].copy()
    unique, count = np.unique(Y_train, return_counts=True)
    print(f"Unique classes and their counts before sampling: {dict(zip(unique, count))}")
    
    # Change your sampling strategy here

    adasyn = ADASYN(sampling_strategy='not majority')
    x = df_filtered.drop(columns=['y_value', 'flare_label'])
    y = df_filtered['y_value']
    x_resampled, y_resampled = adasyn.fit_resample(x, y)

    unique, count = np.unique(y_resampled, return_counts=True)
    print(f"Unique classes and their counts after resampling: {dict(zip(unique, count))}")

    df_resampled = pd.DataFrame(x_resampled, columns=x.columns)
    df_resampled['y_value'] = y_resampled

    print(df_resampled.columns)


    # Save the balanced DataFrame to a new CSV file
    csv_path = os.path.join(output_dir, f"resampled_{data_type}_data.csv")
    df_resampled.to_csv(csv_path, index=False)




    
data_classifier("training_data_raw.csv")