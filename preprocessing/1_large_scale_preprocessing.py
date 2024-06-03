import folium
import osmnx as ox
import random
import pandas as pd
from shapely.ops import unary_union, polygonize
from shapely.geometry import mapping, shape, box
import pickle

# Define the coordinates for the bounding box
north, south, east, west = 27.9510075145, 27.9458517384, -82.4931926234, -82.5015503867
north, south, east, west = 28.8072097729, 28.8026094475, -81.2638223259, -81.2680548517
north, south, east, west = 28.8080997878, 28.8026094475, -81.2607834513, -81.2731967147
north, south, east, west = 28.8108448495, 28.8034245995, -81.2573493648, -81.2732709575
north, south, east, west = 28.8118765659, 28.8024754626, -81.2546626686, -81.2732450126
north, south, east, west = 28.8118765659, 28.7938256985, -81.2546626686, -81.2732450126
north, south, east, west = 27.9227874812, 27.9114301093, -82.4986335891, -82.518138613
north, south, east, west = 33.771452, 33.73163, -84.364965, -84.416463
north, south, east, west = 42.2894327485, 42.2757483703, -71.2119247425, -71.2310113419
north, south, east, west = 41.8075265791, 41.7963455875, -71.404870643, -71.425158872

# Create the bounding box
bbox = (north, south, east, west)

# Download the street network for the bounding box
G = ox.graph_from_bbox(north, south, east, west, network_type='all')

# Get the buildings within the bounding box
buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})

# Create a folium map centered around the midpoint of the bounding box
midpoint = [(north + south) / 2, (east + west) / 2]
m = folium.Map(location=midpoint, zoom_start=16)

# Extract edges from the graph and use them to form polygons
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
lines = [line for line in edges.geometry]
merged_lines = unary_union(lines)
polygons = list(polygonize(merged_lines))

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
        buildings_in_block = buildings[buildings.intersects(polygon)]
        if not buildings_in_block.empty:
            color = random_color()
            add_polygon(m, polygon.__geo_interface__, color)
            block_info = {
                "block_polygon": mapping(polygon),
                "buildings_bbox": []
            }
            for building in buildings_in_block.geometry:
                building_bbox = box(*building.bounds)
                block_info["buildings_bbox"].append(mapping(building_bbox))
            block_data.append(block_info)

    # Add the street network to the map
    folium.GeoJson(edges.to_json(), name='streets', style_function=lambda x: {'color': 'black'}).add_to(m)

    # Add the buildings to the map
    folium.GeoJson(buildings.to_json(), name='buildings', style_function=lambda x: {'color': 'gray'}).add_to(m)

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
    add_polygon(m, block_polygon.__geo_interface__, 'red')

    for bbox in block_info["buildings_bbox"]:
        building_bbox = shape(bbox)
        add_polygon(m, building_bbox.__geo_interface__, 'blue')

# Save the map with loaded data as an HTML file
m.save('blocks_with_buildings_loaded_map.html')
