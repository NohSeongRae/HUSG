from shapely.geometry import Polygon

def check_intersection(polygons):
    # Calculate area for each polygon
    areas = {i: polygon.area for i, polygon in enumerate(polygons)}

    # Sort polygons by area -> 오름차순
    sorted_polygons = sorted(areas.items(), key=lambda item: item[1])

    intersection_polygons = []

    for i in range(len(sorted_polygons)):
        for j in range(i + 1, len(sorted_polygons)):
            # If a smaller polygon intersects with a larger one
            if polygons[sorted_polygons[i][0]].intersects(polygons[sorted_polygons[j][0]]):
                # 작은 거 : 큰 거
                print(f"Polygon {sorted_polygons[i][0]} intersects with Polygon {sorted_polygons[j][0]}")
                intersection_polygons.append({sorted_polygons[i][0]: sorted_polygons[j][0]})

    return intersection_polygons

