import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import glob

# Load the entire dataset
npy_folder_path = 'Z:/iiixr-drive/Projects/2023_City_Team/features'

data_files = glob.glob(f"{npy_folder_path}/*.npy")
data = [np.load(file) for file in data_files]
data = np.concatenate(data)
# data = data.reshape(-1, 1)

non_zero_data = data[data != 0]

# Fit the scaler
scaler = MinMaxScaler()
scaler.fit(non_zero_data.reshape(-1, 1))

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
