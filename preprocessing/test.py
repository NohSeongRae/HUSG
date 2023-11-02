import os


pkl_file_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "6_64_polygon",
                                  "atlanta", "64_point_polygon_exteriors.pkl")

import pickle
import numpy as np
import pandas as pd

# Given file path
file_path = pkl_file_path

# Open the pickle file and load the data
with open(file_path, 'rb') as file:
   data = pickle.load(file)

# Print the overall data structure
print("Overall data structure:", type(data))

# Print the first item in the data if possible
if hasattr(data, '__getitem__'):  # Check if data supports indexing
   first_item = data[0]
   print("First item in data:", first_item)
   try:
      data = np.array(data)
      print("Data has been converted to a numpy array.")
   except Exception as e:
      print("Data could not be converted to a numpy array. Error:", e)
   print("Shape of overall data:", data.shape)

   # Print the shape of the first item if it's a numpy array or pandas DataFrame
   if isinstance(first_item, np.ndarray):
      print("Shape of the first item:", first_item.shape)
   elif isinstance(first_item, pd.DataFrame):
      print("Shape of the first item:", first_item.shape)
   else:
      print("First item is not a numpy array or pandas DataFrame, so it has no shape attribute.")
else:
   print("Data does not support indexing, so the first item cannot be retrieved.")