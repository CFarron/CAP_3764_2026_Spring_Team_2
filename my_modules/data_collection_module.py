import pandas as pd
import numpy as pd

def read_data_pd(path):
    df = pd.read_csv(path)

    return df