import pandas as pd

def load_flight_data(path):
    """
    Load the flight dataset from a CSV file.
    
    Parameters:
        path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(path)
    return df
