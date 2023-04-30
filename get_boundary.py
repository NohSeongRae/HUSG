import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon

def get_boundary(city_name, location):
    custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"]'
    graph = ox.graph_from_place(location, network_type="all", simplify=False, custom_filter=custom_filter)

    simple_graph = nx.Graph(graph)
    simple_graph = simple_graph.to_undirected()

    cycles = nx.cycle_basis(simple_graph)

    polygons = []
    for cycle in cycles:
        coords = []
        for node in cycle:
            node_data = simple_graph.nodes[node]
            if 'x' in node_data and 'y' in node_data:
                coords.append((node_data['x'], node_data['y']))

        if len(coords) >= 3:
            polygon = Polygon(coords)
            polygons.append(polygon)

    for j in range(len(polygons)):
        gdf = gpd.GeoDataFrame(geometry=[polygons[j]])
        polygon_filename = city_name + '_dataset/Boundaries/' + city_name + f'_boundaries{j+1}.geojson'
        gdf.to_file(polygon_filename, driver='GeoJSON')

    print("Boundary 추출 완료")

    filenum = len(polygon)

    return filenum