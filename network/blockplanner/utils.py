import pandas as pd
import io
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from tqdm import tqdm

matplotlib.use('TkAgg')
def plot_data(building_gdf, boundary_gdf):
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    boundary_gdf.boundary.plot(ax=ax2, color='blue', label='Rotated Block Boundary')
    building_gdf.plot(ax=ax2, color='red', label='Rotated Buildings')
    ax2.set_title("Aligned Building Block with Buildings")
    ax2.legend()
    plt.show()

building_filename=r'Z:\iiixr-drive\Projects\2023_City_Team\neworleans_dataset\density20_building120_rotate_normalized\Buildings\neworleans_buildings6.geojson'
boundary_filename=r'Z:\iiixr-drive\Projects\2023_City_Team\neworleans_dataset\density20_building120_rotate_normalized\Boundaries\neworleans_boundaries6.geojson'
boundary_gdf = gpd.read_file(boundary_filename)
building_gdf = gpd.read_file(building_filename)
plot_data(building_gdf, boundary_gdf)