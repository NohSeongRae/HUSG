import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import seaborn as sns
import random

from box import AutoregressiveBoxEncoder, AutoregressiveBoxDecoder
from layout import BatchCollator, LayoutDataset
from train_layouts import AutoregressiveBoxVariationalAutoencoder, plot_layout

def gen_colors(num_colors):
    """
    Generate `num_colors` distinct colors.
    :param num_colors: The number of distinct colors to generate.
    :return: A list of color triples in RGB format.
    """
    palette = sns.color_palette("hls", num_colors)
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in palette]
    return colors

def sample_layout(model, label_encodings, max_number_boxes, width, height, device, single_label_index):
    model.eval()
    with torch.no_grad():
        label_set = label_encodings[single_label_index].unsqueeze(0).to(device)
        previous_labels = torch.zeros((1, 0, label_encodings.size(1))).to(device)
        previous_boxes = torch.zeros((1, 0, 4)).to(device)
        state = None
        predicted_boxes = torch.zeros((1, max_number_boxes, 4)).to(device)

        for step in range(max_number_boxes):
            current_label = label_set

            dummy_x = torch.zeros((1, 4)).to(device)

            predicted_box, _, _, state = model(
                dummy_x,
                label_set,
                current_label,
                previous_labels,
                previous_boxes,
                state=state
            )

            # predicted_box, _, _, state = model(
            #     None,
            #     label_set,
            #     current_label,
            #     previous_labels,
            #     previous_boxes,
            #     state=state
            # )
            predicted_boxes[:, step, :] = predicted_box
            previous_labels = torch.cat([previous_labels, current_label.unsqueeze(1)], dim=1)
            previous_boxes = torch.cat([previous_boxes, predicted_box.unsqueeze(1)], dim=1)

        predicted_boxes[:, :, [0, 2]] *= width
        predicted_boxes[:, :, [1, 3]] *= height

        return predicted_boxes.squeeze(0)

NUMBER_LABELS = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "C:/Users/rlaqhdrb/Desktop/eval/DeepLayout/layout_vae/logs/110623_014140_box_coco_instances/checkpoints/epoch_20.pth"  # 모델 체크포인트 경로를 설정하세요.
autoencoder = AutoregressiveBoxVariationalAutoencoder(
    NUMBER_LABELS,
    conditioning_size=128,
    representation_size=32).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
autoencoder.load_state_dict(checkpoint['model_state_dict'])


label_encodings = torch.eye(NUMBER_LABELS).to(device)


max_number_boxes = 20  # 최대 박스 수 설정
width, height = 256, 256


number_of_layouts_to_generate = 5  # 생성할 레이아웃의 수

def normalize_coordinates(boxes, image_width, image_height):
    image_width = 25
    image_height = 25

    normalized_boxes = []
    for box in boxes:
        x = box[0] / image_width
        y = box[1] / image_height
        w = box[2] / image_width
        h = box[3] / image_height
        normalized_boxes.append([x, y, w, h])
    return normalized_boxes

for _ in range(number_of_layouts_to_generate):
    single_label_index = random.choice(range(NUMBER_LABELS))
    colors = gen_colors(NUMBER_LABELS)

    predicted_boxes = sample_layout(autoencoder, label_encodings, max_number_boxes, width, height, device, single_label_index)

    real_boxes = np.zeros((max_number_boxes, 4))
    labels = np.zeros(max_number_boxes)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image_width = 256
    image_height = 256

    normalized_boxes = normalize_coordinates(predicted_boxes, 30, 30)
    #
    # for i, box in enumerate(predicted_boxes):
    #     print(f"Box {i}: {box}")

    fig, ax = plt.subplots()

    ax.imshow(np.ones((image_height, image_width, 3)), extent=[0, image_width, 0, image_height], cmap='gray')

    for box in predicted_boxes:

        box_cpu = box.cpu().numpy()
        x, y, w, h = box_cpu  # box 좌표

        x *= image_width/40
        y *= image_height/40
        w *= image_width/40
        h *= image_height/40

        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')

    plt.show()
