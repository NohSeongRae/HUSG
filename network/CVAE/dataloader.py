import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, data_path, condition_path):
        self.data_path = data_path
        self.condition_path = condition_path

    def load_data(self):
        # Load data
        with np.load(self.data_path) as data:
            data = data['arr_0']
        with np.load(self.condition_path) as conditions:
            conditions = conditions['arr_0']

        # Normalize data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        conditions = scaler.fit_transform(conditions)

        return data, conditions