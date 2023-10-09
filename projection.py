import numpy as np
from shapely.geometry import Polygon, LineString


def segment_linestring(linestring, segment_length):
    assert isinstance(linestring, LineString)
    total_length = linestring.length
    segments = []
    for i in np.arange(0, total_length, segment_length):
        point1 = linestring.interpolate(i)
        point2 = linestring.interpolate(min(i + segment_length, total_length))
        segment = LineString([point1, point2])
        segments.append(segment)
    return segments


def project_point_onto_line(P, A, B):
    AP = P - A
    AB = B - A
    AB_squared_norm = np.sum(AB ** 2)
    scalar_proj = np.sum(AP * AB) / AB_squared_norm
    proj = A + scalar_proj * AB
    return proj


def project_polygon_onto_linestring(polygon, linestring, segment_length):
    assert isinstance(polygon, Polygon)
    assert isinstance(linestring, LineString)

    poly_coords = np.array(polygon.exterior.coords)
    segments = segment_linestring(linestring, segment_length)
    overlaps = []

    for seg in segments:
        # Extract coordinates
        line_coords = np.array(seg.coords)

        # Project vertices of polygon onto line segment
        proj_coords = np.array([project_point_onto_line(p, line_coords[0], line_coords[1]) for p in poly_coords])

        # Check if the projected polygon overlaps with the line segment
        projected_polygon = Polygon(proj_coords)
        overlaps.append(projected_polygon.intersects(seg))

    return overlaps

polygon = Polygon([(0,0), (1,2), (2,0)])
linestring = LineString([(0,1), (3,1)])
segment_length = 1.0

overlaps = project_polygon_onto_linestring(polygon, linestring, segment_length)