import os
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, LineString, Point
import copy

matplotlib.use('TkAgg')
# city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
# "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
# "sanfrancisco", "miami", "seattle", "boston", "providence",
# "neworleans", "denver", "pittsburgh", "tampa", "washington"]

def plot_boundary_building(building_polygons, boundary_polygon):
    plt.figure(figsize=(8, 8))
    for poly in building_polygons:
        x, y = poly.exterior.xy

        plt.plot(x, y)

    boundary_x, boundary_y = boundary_polygon.exterior.xy
    plt.plot(boundary_x, boundary_y)

    plt.show()


def sort_key(filename):
    # 파일 이름에서 숫자만 추출
    num = int(''.join(filter(str.isdigit, filename)))
    return num


def get_obb_rotation_angle(polygon):
    """
    Calculate the oriented bounding box (OBB) for the given polygon
    and return the angle (in degrees) to align the longest side of the OBB with x-axis.
    """
    # Get the oriented bounding box (OBB) of the polygon
    obb = polygon.minimum_rotated_rectangle
    # Get the coordinates of the OBB
    obb_coords = list(obb.exterior.coords)
    # Find two consecutive vertices with the maximum distance (longest side)
    max_dist = 0
    p1, p2 = Point(obb_coords[0]), Point(obb_coords[1])
    max_dist = p1.distance(p2)
    for i in range(len(obb_coords) - 1):
        temp_p1, temp_p2 = Point(obb_coords[i]), Point(obb_coords[i + 1])
        dist = temp_p1.distance(temp_p2)
        if dist > max_dist:
            max_dist = dist
            p1, p2 = Point(obb_coords[i]), Point(obb_coords[i + 1])

    # Calculate the angle to rotate the longest side to align with x-axis
    angle_rad = np.arctan2(p2.y - p1.y, p2.x - p1.x)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180
    if angle_deg > 90 and angle_deg < 180:
        angle_deg -= 90
    # if angle_deg <90 and angle_deg >45:
    #     angle_deg-=45
    return angle_deg


def compute_mean_without_outliers(data):
    """
    Compute the mean of the data without considering outliers using the IQR method.
    """
    # Step 1: Sort the data
    sorted_data = sorted(data)

    # Step 2: Compute Q1 and Q3
    Q1 = np.percentile(sorted_data, 25)
    Q3 = np.percentile(sorted_data, 75)
    # print(f'Q1: {Q1}, Q2: {Q3}')
    # Step 3: Calculate IQR
    IQR = Q3 - Q1
    # print("IQR", IQR)

    # Step 4: Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Step 5: Filter the data
    filtered_data = [x for x in sorted_data if lower_bound <= x <= upper_bound]
    mean1 = np.mean(sorted_data)
    mean2 = np.mean(filtered_data)
    print(f"mean1: {mean1}, mean2:{mean2}")
    # Step 6: Compute the mean of the filtered data
    return np.mean(filtered_data)


def align_block_to_axis(block, buildings):
    """
    Align the buildings of the block to the axis and return the rotated block and buildings.
    """
    # Calculate the average rotation angle of all buildings
    # counter=0
    # for building in buildings.geometry:
    #     print(counter, building)
    #     counter+=1
    angles = [get_obb_rotation_angle(building) for building in buildings.geometry]
    avg_angle = np.mean(angles)
    print(f"avg_angle{avg_angle}")
    avg_angle_no_outlier = compute_mean_without_outliers(angles)
    print(f"avg_angle no outlier{avg_angle_no_outlier}")
    # Rotate the entire block and buildings by the negative average angle
    rotated_block = block.rotate(-avg_angle_no_outlier)
    rotated_buildings = buildings.rotate(-avg_angle_no_outlier, origin=(0.5, 0.5))

    return rotated_block, rotated_buildings


counter = 0
city_names = ["neworleans"]

for city_name in city_names:
    print("city : ", city_name)
    building_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_Normalized', 'Buildings')
    boundary_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_Normalized', 'Boundaries')

    # Iterate over all .geojson files in the directory
    for building_filepath in tqdm(
            sorted([f for f in os.listdir(building_dir_path) if f.endswith('.geojson')], key=sort_key)):
        num = sort_key(building_filepath)
        boundary_filepath = building_filepath.replace('buildings', 'boundaries')

        # Construct the full paths
        building_filename = os.path.join(building_dir_path, building_filepath)
        boundary_filename = os.path.join(boundary_dir_path, boundary_filepath)

        if os.path.exists(building_filename):

            boundary_gdf = gpd.read_file(boundary_filename)
            building_gdf = gpd.read_file(building_filename)
            boundary_gdf2 = copy.deepcopy(boundary_gdf)
            building_gdf2 = copy.deepcopy(building_gdf)
            boundary_polygon = boundary_gdf.iloc[0]['geometry']
            building_polygon = [row['geometry'] for idx, row in building_gdf.iterrows()]
            building_polygons = []
            building_polygons.extend(building_polygon)
            plot_boundary_building(building_polygons, boundary_polygon)

            building_polygons.clear()

            rotated_block_gdf, rotated_buildings_gdf = align_block_to_axis(boundary_gdf2, building_gdf2)

            fig, ax = plt.subplots(figsize=(8, 8))
            rotated_block_gdf.boundary.plot(ax=ax, color='blue', label='Rotated Block Boundary')
            rotated_buildings_gdf.plot(ax=ax, color='red', label='Rotated Buildings')
            ax.set_title("Aligned Building Block with Buildings")
            ax.legend()
            plt.show()
            counter += 1
            if counter > 30:
                break
