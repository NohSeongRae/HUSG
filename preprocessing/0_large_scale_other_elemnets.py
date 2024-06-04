import folium
import osmnx as ox
import pandas as pd
from shapely.ops import transform
import pyproj
import pickle
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon

# Define the coordinates for the bounding box
north, south, east, west = 41.8075265791, 41.7963455875, -71.404870643, -71.425158872

# Create the bounding box
bbox = (north, south, east, west)

# Define a function to transform geometries to a common coordinate system (UTM zone 19N for Rhode Island)
def transform_to_utm(geom):
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True).transform
    return transform(project, geom)

def transform_to_wgs84(geom):
    project = pyproj.Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True).transform
    return transform(project, geom)

# Extract natural elements (e.g., parks, forests, grasslands, water bodies)
natural_tags = {
    'leisure': True,
    'landuse': ['forest', 'grass', 'meadow', 'park'],
    'natural': ['wood', 'scrub', 'heath', 'grassland', 'wetland', 'water', 'bay', 'cape', 'beach']
}
natural_elements = ox.features_from_bbox(bbox=bbox, tags=natural_tags)

# Transform the natural elements to UTM
natural_elements_utm = natural_elements.copy()
natural_elements_utm['geometry'] = natural_elements_utm['geometry'].apply(transform_to_utm)

# Create a folium map centered around the midpoint of the bounding box
midpoint = [(north + south) / 2, (east + west) / 2]
m = folium.Map(location=midpoint, zoom_start=16)

# Define a function to assign colors based on tags
def get_color_by_tag(properties):
    if properties.get('landuse') == 'forest':
        return 'darkgreen'
    elif properties.get('landuse') == 'grass':
        return 'limegreen'
    elif properties.get('landuse') == 'meadow':
        return 'yellowgreen'
    elif properties.get('landuse') == 'park':
        return 'blue'  # Use blue for parks
    elif properties.get('natural') in ['wood', 'scrub', 'heath']:
        return 'forestgreen'
    elif properties.get('natural') in ['grassland']:
        return 'greenyellow'
    elif properties.get('natural') in ['wetland']:
        return 'blue'
    elif properties.get('natural') in ['water', 'bay']:
        return 'aqua'
    elif properties.get('natural') in ['cape', 'beach']:
        return 'sandybrown'
    elif properties.get('leisure') == 'park':
        return 'blue'
    elif properties.get('leisure') == 'playground':
        return 'pink'
    elif properties.get('leisure') == 'garden':
        return 'violet'
    elif properties.get('leisure') == 'pitch':
        return 'orange'
    return 'gray'  # Default color

# Add natural elements to the map
if not natural_elements.empty:
    natural_elements_wgs84 = natural_elements.copy()
    natural_elements_wgs84['geometry'] = natural_elements_wgs84['geometry'].apply(transform_to_wgs84)
    folium.GeoJson(natural_elements.to_json(), name='natural_elements', style_function=lambda x: {
        'color': get_color_by_tag(x['properties'])
    }).add_to(m)

    # Save natural elements to a pickle file
    with open('natural_elements_utm.pkl', 'wb') as f:
        pickle.dump(natural_elements_utm, f)
else:
    print("No natural elements found within the bounding box.")

# Add layer control to toggle streets and natural elements
folium.LayerControl().add_to(m)

# Save the map as an HTML file
m.save('natural_elements_map.html')

# Load the pickle file for verification
with open('natural_elements_utm.pkl', 'rb') as f:
    loaded_natural_elements_utm = pickle.load(f)

# Verify that the loaded data matches the original data
print(loaded_natural_elements_utm.head())

# Plotting the natural elements with Matplotlib

# Load the pickle file
with open('natural_elements_utm.pkl', 'rb') as f:
    natural_elements_utm = pickle.load(f)

# Ensure the loaded data is a GeoDataFrame
if isinstance(natural_elements_utm, gpd.GeoDataFrame):
    # Print the columns to understand the structure of the DataFrame
    print(natural_elements_utm.columns)

    # Define a function to get category label
    def get_category_label(row):
        if row.get('landuse') == 'forest':
            return 'Forest'
        elif row.get('landuse') == 'grass':
            return 'Grass'
        elif row.get('landuse') == 'meadow':
            return 'Meadow'
        elif row.get('landuse') == 'park':
            return 'Park'
        elif row.get('natural') in ['wood', 'scrub', 'heath']:
            return 'Wood/Scrub/Heath'
        elif row.get('natural') in ['grassland']:
            return 'Grassland'
        elif row.get('natural') in ['wetland']:
            return 'Wetland'
        elif row.get('natural') in ['water', 'bay']:
            return 'Water/Bay'
        elif row.get('natural') in ['cape', 'beach']:
            return 'Cape/Beach'
        elif row.get('leisure') == 'park':
            return 'Park'
        elif row.get('leisure') == 'playground':
            return 'Playground'
        elif row.get('leisure') == 'garden':
            return 'Garden'
        elif row.get('leisure') == 'pitch':
            return 'Pitch'
        return 'Other'  # Default label

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each feature with its corresponding color and label
    for idx, row in natural_elements_utm.iterrows():
        gpd.GeoSeries([row['geometry']]).plot(ax=ax, color=get_color_by_tag(row))
        centroid = row['geometry'].centroid
        ax.text(centroid.x, centroid.y, get_category_label(row), fontsize=8, ha='center')

    # Add a legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='darkgreen', lw=4, label='Forest'),
        Line2D([0], [0], color='limegreen', lw=4, label='Grass'),
        Line2D([0], [0], color='yellowgreen', lw=4, label='Meadow'),
        Line2D([0], [0], color='blue', lw=4, label='Park'),
        Line2D([0], [0], color='forestgreen', lw=4, label='Wood/Scrub/Heath'),
        Line2D([0], [0], color='greenyellow', lw=4, label='Grassland'),
        Line2D([0], [0], color='blue', lw=4, label='Wetland'),
        Line2D([0], [0], color='aqua', lw=4, label='Water/Bay'),
        Line2D([0], [0], color='sandybrown', lw=4, label='Cape/Beach'),
        Line2D([0], [0], color='pink', lw=4, label='Playground'),
        Line2D([0], [0], color='violet', lw=4, label='Garden'),
        Line2D([0], [0], color='orange', lw=4, label='Pitch'),
        Line2D([0], [0], color='gray', lw=4, label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.title('Natural Elements from Pickle File')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
else:
    print("Loaded data is not a GeoDataFrame")
