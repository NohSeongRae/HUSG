import json
import numpy as np

annotations_path = "../annotations/stuff_train2017.json"
max_length = 128

with open(annotations_path, "r") as f:
    data = json.load(f)


categories = {c["id"]: c for c in data["categories"]}
number_labels = len(categories)


json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate([c["id"] for c in categories.values()])
        }

contiguous_category_id_to_json_id = {
            v: k for k, v in json_category_id_to_contiguous_id.items()
        }

image_to_annotations = {}

for annotation in data["annotations"]:
    image_id = annotation["image_id"]

    if not (image_id in image_to_annotations):
        image_to_annotations[image_id] = []

    image_to_annotations[image_id].append(annotation)


label_sets = []
counts = []
boxes = []
labels = []
annotation_ids = []
widths = []
heights = []
image_ids = []
permutations = []

images = []
annotations = []

for image in data["images"]:
    image_id = image["id"]
    height, width = float(image["height"]), float(image["width"])

    if image_id not in image_to_annotations:
        continue

    annotations = image_to_annotations[image_id]

    if (max_length is not None) and (len(annotations) > max_length):
        annotations = annotations[:max_length]

    # hack.
    for i, annotation in enumerate(annotations):
        annotation["index"] = i

    # sort the annotations left to right with labels (smallest first).
    sorted_annotations = []
    for label_index in range(number_labels):
        category_id = contiguous_category_id_to_json_id[label_index + 1]
        annotations_of_label = [a for a in annotations if a["category_id"] == category_id]
        annotations_of_label = list(sorted(annotations_of_label, key=lambda a: a["bbox"][0]))
        sorted_annotations += annotations_of_label

    annotations.append(sorted_annotations)

    label_set = np.zeros((number_labels,)).astype(np.uint8)
    count = np.zeros((number_labels,)).astype(np.uint8)
    box = np.zeros((len(sorted_annotations), 4))
    label = np.zeros((len(sorted_annotations),))
    annotation_id = np.zeros((len(sorted_annotations),))

    for annotation_index, annotation in enumerate(sorted_annotations):
        contiguous_id = json_category_id_to_contiguous_id[annotation["category_id"]]
        label_set[contiguous_id - 1] = 1
        count[contiguous_id - 1] += 1
        x, y, w, h = annotation["bbox"]

        # a good question is if we should divide by the long edge only.
        box[annotation_index] = np.array([x / width, y / height, w / width, h / height])
        label[annotation_index] = contiguous_id
        annotation_id[annotation_index] = annotation["id"]

    permutation = np.array([a["index"] for a in sorted_annotations]).astype(int)

    label_sets.append(label_set)
    counts.append(count)
    boxes.append(box)
    labels.append(label)
    widths.append(width)
    heights.append(height)
    annotation_ids.append(annotation_id)
    image_ids.append(image_id)
    permutations.append(permutation)
    images.append(image)

# print(widths)
print(images)
