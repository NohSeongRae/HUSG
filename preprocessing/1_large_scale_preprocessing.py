import folium
import osmnx as ox
import random
import pandas as pd
from shapely.ops import unary_union, polygonize, transform
from shapely.geometry import mapping, shape
import pickle
import pyproj

# Define the coordinates for the bounding box
north, south, east, west = 41.8075265791, 41.7963455875, -71.404870643, -71.425158872
north, south, east, west = 40.8772895974, 40.8686900945, -74.0427050295, -74.0543780031
north, south, east, west = 41.2125143307, 41.2015207154, -73.206770667, -73.2269837941

# Create the bounding box
bbox = (north, south, east, west)

# Download the street network for the bounding box
G = ox.graph_from_bbox(north, south, east, west, network_type='all')

# Get the buildings within the bounding box
buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})

# Define a function to transform geometries to a common coordinate system (UTM zone 19N for Rhode Island)
def transform_to_utm(geom):
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True).transform
    return transform(project, geom)

def transform_to_wgs84(geom):
    project = pyproj.Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True).transform
    return transform(project, geom)

# Transform the geometries to UTM
G_utm = ox.project_graph(G, to_crs="EPSG:32619")
buildings_utm = buildings.copy()
buildings_utm['geometry'] = buildings_utm['geometry'].apply(transform_to_utm)

# Extract edges from the graph and use them to form polygons
edges = ox.graph_to_gdfs(G_utm, nodes=False, edges=True)
lines = [line for line in edges.geometry]
merged_lines = unary_union(lines)
polygons = list(polygonize(merged_lines))

# Create a folium map centered around the midpoint of the bounding box
midpoint = [(north + south) / 2, (east + west) / 2]
m = folium.Map(location=midpoint, zoom_start=16)

# Function to add a polygon to the map
def add_polygon(map_obj, polygon, color):
    folium.GeoJson(polygon, style_function=lambda x: {'color': color, 'fillOpacity': 0.5}).add_to(map_obj)

# Function to generate a random color
def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Lists to store block and building information
block_data = []

# Check if polygons list is empty
if not polygons:
    print("No blocks found within the bounding box.")
else:
    # Add each block with a random color to the map if it contains buildings
    for polygon in polygons:
        buildings_in_block = buildings_utm[buildings_utm.intersects(polygon)]
        if not buildings_in_block.empty:
            color = random_color()
            block_polygon_wgs84 = transform_to_wgs84(polygon)
            add_polygon(m, block_polygon_wgs84.__geo_interface__, color)
            block_info = {
                "block_polygon": mapping(polygon),
                "buildings_bbox": []
            }
            for building in buildings_in_block.geometry:
                # Calculate the minimum rotated bounding box
                building_min_rot_bbox = building.minimum_rotated_rectangle
                building_bbox_wgs84 = transform_to_wgs84(building_min_rot_bbox)
                block_info["buildings_bbox"].append(mapping(building_min_rot_bbox))
                add_polygon(m, building_bbox_wgs84.__geo_interface__, 'blue')
            block_data.append(block_info)

    # Add the street network to the map
    edges_wgs84 = edges.copy()
    edges_wgs84['geometry'] = edges_wgs84['geometry'].apply(transform_to_wgs84)
    folium.GeoJson(edges_wgs84.to_json(), name='streets', style_function=lambda x: {'color': 'black'}).add_to(m)

    # Add the buildings to the map
    buildings_wgs84 = buildings.copy()
    buildings_wgs84['geometry'] = buildings_wgs84['geometry'].apply(transform_to_wgs84)
    folium.GeoJson(buildings_wgs84.to_json(), name='buildings', style_function=lambda x: {'color': 'gray'}).add_to(m)

    # Add layer control to toggle streets and buildings
    folium.LayerControl().add_to(m)

    # Save the map as an HTML file
    m.save('blocks_with_buildings_map.html')

    # Save block and building information to a pickle file
    with open('block_building_info.pkl', 'wb') as f:
        pickle.dump(block_data, f)

# To load the pickle file and visualize the data again
with open('block_building_info.pkl', 'rb') as f:
    loaded_block_data = pickle.load(f)

# Create a new folium map for visualization
m = folium.Map(location=midpoint, zoom_start=16)

# Iterate through the loaded block data and add polygons to the map
for block_info in loaded_block_data:
    block_polygon = shape(block_info["block_polygon"])
    block_polygon_wgs84 = transform_to_wgs84(block_polygon)
    add_polygon(m, block_polygon_wgs84.__geo_interface__, 'red')

    for bbox in block_info["buildings_bbox"]:
        building_bbox = shape(bbox)
        building_bbox_wgs84 = transform_to_wgs84(building_bbox)
        add_polygon(m, building_bbox_wgs84.__geo_interface__, 'blue')

# Save the map with loaded data as an HTML file
m.save('blocks_with_buildings_loaded_map.html')
