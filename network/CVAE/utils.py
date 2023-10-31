import pickle
import numpy as np

def summarize_structure(obj, level=0, sample=False):
    indent = ' ' * level
    if isinstance(obj, dict):
        print(f"{indent}{type(obj)} containing {len(obj)} key-value pairs:")
        for i, (key, value) in enumerate(obj.items()):
            print(f"{indent}  Key: {key} ->", end=" ")
            summarize_structure(value, level+4, sample=(i == 0))  # Sample the first item
    elif isinstance(obj, list):
        print(f"{indent}{type(obj)} with {len(obj)} elements:")
        if len(obj) > 0 and sample:
            print(f"{indent}  Sample element:")
            summarize_structure(obj[0], level+4, sample=True)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{type(obj)} with shape {obj.shape} and data type {obj.dtype}")
        if sample:
            # Take a slice from each dimension; adjust the slice sizes as necessary
            sample_slice = tuple(slice(0, min(10, dim)) for dim in obj.shape)
            sample_data = obj[sample_slice]
            print(f"{indent}  Sample data (slice of the array): {sample_data}")
    elif hasattr(obj, '__dict__'):
        print(f"{indent}{type(obj)} with attributes:")
        for attr in obj.__dict__:
            print(f"{indent}  {attr}")
            if sample:
                attr_value = getattr(obj, attr)
                if isinstance(attr_value, (np.ndarray, list)) and len(attr_value) > 0:
                    print(f"{indent}    Sample data: {attr_value[0]}")
                else:
                    print(f"{indent}    Value: {attr_value}")
                sample = False  # Only sample the first attribute
    else:
        print(f"{indent}{type(obj)}")
        if sample:
            print(f"{indent}  Value: {obj}")



with open(r'Z:\iiixr-drive\Projects\2023_City_Team\2_transformer\train_dataset\littlerock\node_features.pkl',
          'rb') as file:
    data = pickle.load(file)
    summarize_structure(data, sample=True)
