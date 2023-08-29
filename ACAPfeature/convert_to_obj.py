import numpy as np
import pandas as pd
import os
from scipy.spatial import Delaunay

pointnum = 596

city_list = ['lasvegas', 'littlerock', 'manchester', 'milan', 'minneapolis', 'nottingham', 'paris', 'philadelphia', 'phoenix', 'portland', 'richmond', 'saintpaul', 'sanfrancisco']

def convert_csv_to_obj_using_pattern(csv_filename, output_filename, triangulation_pattern=None):
    # Load data from CSV
    data = pd.read_csv(csv_filename, header=None, names=["x", "y"])

    # Compute Delaunay triangulation if no pattern is given
    if triangulation_pattern is None:
        tri = Delaunay(data.values)
        simplices = tri.simplices
    else:
        simplices = triangulation_pattern

    with open(output_filename, 'w') as f:
        # Write vertices
        for _, row in data.iterrows():
            f.write(f"v {row['x']} {row['y']} 0\n")

        # Write faces using the given triangulation pattern
        for simplex in simplices:
            f.write(f"f {simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}\n")

    return simplices


if __name__ == "__main__":
    # Convert normalized_points.csv to square.obj and get its triangulation pattern
    for idx in range(len(city_list)):
        city_name = city_list[idx]

        square_csv = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'square_{pointnum}.csv')
        square_obj = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'square_{pointnum}.obj')

        triangulation_pattern = convert_csv_to_obj_using_pattern(square_csv, square_obj)

        # Directory containing the csv files you want to convert
        csv_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'Boundarycsv_{pointnum}')

        # Directory where you want to save the .obj files
        obj_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaryobj')

        # Make the obj_directory if it doesn't exist
        if not os.path.exists(obj_directory):
            os.makedirs(obj_directory)

        # Get list of all CSV files in the register directory
        all_csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

        for csv_file in all_csv_files:
            input_path = os.path.join(csv_directory, csv_file)
            # Construct the name for the output .obj file based on the csv file name
            output_filename = csv_file.replace('.csv', '.obj')
            output_path = os.path.join(obj_directory, output_filename)
            # Convert current CSV to .obj using the triangulation pattern
            convert_csv_to_obj_using_pattern(input_path, output_path, triangulation_pattern)

        print(f"{city_name} done")