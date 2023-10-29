import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, data_path, condition_path):
        self.data_path = data_path
        self.condition_path = condition_path

    def load_data(self):
        # Load data
        data = pd.read_csv(self.data_path)
        conditions = pd.read_csv(self.condition_path)

        # Normalize data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        conditions = scaler.fit_transform(conditions)

        return data, conditions