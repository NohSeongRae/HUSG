import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def convert_csv_to_obj(csv_filename, output_filename):
    # Load data from CSV
    data = pd.read_csv(csv_filename, header=None, names=["x", "y"])

    # Compute Delaunay triangulation
    tri = Delaunay(data.values)

    with open(output_filename, 'w') as f:
        # Write vertices
        for _, row in data.iterrows():
            f.write(f"v {row['x']} {row['y']} 0\n")

        # Write faces
        for simplex in tri.simplices:
            f.write(f"f {simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}\n")


if __name__ == "__main__":
    # Convert normalized_points.csv to square.obj
    convert_csv_to_obj("C:/Users/rlaqhdrb/Desktop/misong/normalized_points.csv",
                       "C:/Users/rlaqhdrb/Desktop/misong/square.obj")

    # Convert final_points.csv to boundary.obj
    convert_csv_to_obj("C:/Users/rlaqhdrb/Desktop/misong/final_points.csv",
                       "C:/Users/rlaqhdrb/Desktop/misong/boundary.obj")
