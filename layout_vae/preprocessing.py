import json
import networkx as nx
import os
from tqdm import tqdm

def convert_gpickle_to_json(gpickle_dir, output_path):
    image_data = []
    annotations = []
    categories = {}
    annotation_id = 0
    image_id = 0

    gpickle_files = [os.path.join(gpickle_dir, file) for file in os.listdir(gpickle_dir) if file.endswith('.gpickle')]

    for gpickle_file in tqdm(gpickle_files):
        graph = nx.read_gpickle(gpickle_file)

        # Add image data
        image_data.append({
            "id": image_id,
            "width": 1,  # Assuming a fixed size, modify as needed
            "height": 1,  # Assuming a fixed size, modify as needed
            "file_name": os.path.basename(gpickle_file)
        })

        for node_id, node_attrs in graph.nodes(data=True):
            bbox = node_attrs['node_features'][:4]  # x, y, w, h 값
            category_id = 0

            if category_id not in categories:
                categories[category_id] = {"id": category_id, "name": f"category{category_id}"}

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox.tolist()
            })
            annotation_id += 1

        image_id += 1

    categories = list(categories.values())

    dataset = {
        "images": image_data,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)


# Example usage
gpickle_dir = "C:\\Users\\Dobby\\Downloads\\datasets\\ours_graph_datasets\\val"
output_path = "C:\\Users\\Dobby\\Downloads\\datasets\\ours_graph_datasets\\instances_val.json"
convert_gpickle_to_json(gpickle_dir, output_path)