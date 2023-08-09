import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def convert_csv_to_obj(csv_filename, output_filename, face_limit=None):
    # Load data from CSV
    data = pd.read_csv(csv_filename, header=None, names=["x", "y"])

    # Compute Delaunay triangulation
    tri = Delaunay(data.values)

    with open(output_filename, 'w') as f:
        # Write vertices
        for _, row in data.iterrows():
            f.write(f"v {row['x']} {row['y']} 0\n")

        # Write faces with limit
        face_count = 0
        for simplex in tri.simplices:
            if face_limit and face_count >= face_limit:
                break
            f.write(f"f {simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}\n")
            face_count += 1

    return face_count  # Return the number of faces written


if __name__ == "__main__":
    # Create square.obj without any face limit
    square_face_count = convert_csv_to_obj("C:/Users/rlaqhdrb/Desktop/misong/normalized_points.csv",
                                           "C:/Users/rlaqhdrb/Desktop/misong/square.obj")

    # Create boundary.obj using the square's face count as the limit
    convert_csv_to_obj("C:/Users/rlaqhdrb/Desktop/misong/final_points.csv",
                       "C:/Users/rlaqhdrb/Desktop/misong/boundary.obj",
                       face_limit=square_face_count)
