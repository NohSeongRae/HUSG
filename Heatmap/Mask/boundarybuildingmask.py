import os
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from skimage.morphology import dilation, square
from tqdm import tqdm
import numpy as np
import imageio



def get_square_bounds(geojson_path):
    gdf = gpd.read_file(geojson_path)
    bounds = gdf.total_bounds
    square_size = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    center_x, center_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

    square_coords = [
        (center_x - square_size / 2, center_y - square_size / 2),
        (center_x + square_size / 2, center_y + square_size / 2)
    ]

    return square_coords[0][0], square_coords[0][1], square_coords[1][0], square_coords[1][1]

def boundarybuildingmask(city_name):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    for i in tqdm(range(1, filenum + 1)):
        boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'filtered_data', 'Boundaries', f'{city_name}_boundaries{i}.geojson')
        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'filtered_data', 'Buildings', f'{city_name}_buildings{i}.geojson')

        if os.path.exists(building_filename):
            buildings_gdf = gpd.read_file(building_filename)
            boundary_gdf = gpd.read_file(boundary_filename)
            boundaries = boundary_gdf.geometry.boundary

            boundaries_list = list(boundaries)


            width, height = 224, 224
            left, bottom, right, top = get_square_bounds(boundary_filename)
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

            # Set everything to white initially
            final_mask = np.ones((height, width), dtype=np.uint8) * 255


            # Make block boundary as black
            block_boundary_mask = geometry_mask(boundaries_list, transform=transform, invert=True, out_shape=(height, width))
            final_mask[block_boundary_mask] = 0


            # 경계선 굵기 조절
            thick_boundary_mask = dilation(block_boundary_mask, square(1))

            final_mask = np.where(thick_boundary_mask, 0, 255).astype(np.uint8)

            # Mask buildings as black
            inside_building_mask = geometry_mask(buildings_gdf.geometry, transform=transform, invert=True, out_shape=(height, width))
            final_mask[inside_building_mask] = 0


            # Draw building boundaries as white
            for geometry in buildings_gdf['geometry']:
                boundary = geometry.boundary
                boundary_mask = geometry_mask([boundary], transform=transform, invert=True, out_shape=(height, width))
                final_mask[boundary_mask] = 255

            insidemask_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarybuildingmask', f'{city_name}_boundarybuildingmask{i}.png')
            imageio.imsave(insidemask_filename, final_mask)

