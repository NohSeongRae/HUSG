import gc
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageDraw
import random
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from box import AutoregressiveBoxEncoder, AutoregressiveBoxDecoder
from layout import BatchCollator, LayoutDataset

# 기존의 bbox, 회전 관련 함수 추가
def get_bbox_corners(x, y, w, h):
    half_w = w / 2
    half_h = h / 2

    top_left = [x - half_w, y - half_h]
    top_right = [x + half_w, y - half_h]
    bottom_left = [x - half_w, y + half_h]
    bottom_right = [x + half_w, y + half_h]

    return [top_left, top_right, bottom_right, bottom_left]

def rotate_points_around_center(points, center, theta_deg):
    theta_rad = np.radians(theta_deg)

    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    points = np.array(points)
    center = np.array(center)
    translated_points = points - center

    rotated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_points + center

    return rotated_points

# 기존의 레이아웃 플롯 함수 수정
def plot_layout(real_boxes, predicted_boxes, labels, width, height, colors=None, save_path_1=None, save_path_2=None):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))

    rotation_scale = 45  # 각도 스케일 추가

    for i in range(len(real_boxes)):
        real_box = real_boxes[i].tolist()
        predicted_box = predicted_boxes[i].tolist()
        label = int(labels[i])

        # 실제 박스
        x, y, w, h, theta = real_box[0], real_box[1], real_box[2], real_box[3], 0
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)
        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)

        ax1.plot(rotated_box[:, 0], rotated_box[:, 1], color='k')

        # 예측 박스
        x, y, w, h, theta = predicted_box[0], predicted_box[1], predicted_box[2], predicted_box[3], 0
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)
        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)

        ax2.plot(rotated_box[:, 0], rotated_box[:, 1], color='k')

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_axis_off()
    if save_path_1:
        ax1.figure.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close(ax1.figure)

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_axis_off()
    if save_path_2:
        ax2.figure.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close(ax2.figure)

# 색상 생성 함수 추가
def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)] for x in palette]
    return rgb_triples

def evaluate_and_visualize(model, loader, loss, save_dir, prefix='', colors=None):
    errors = []
    model.eval()
    losses = None
    box_losses = []
    divergence_losses = []

    # Save directory for visualization
    os.makedirs(save_dir, exist_ok=True)

    for batch_i, (indexes, target, filename) in tqdm(enumerate(loader)):
        if batch_i > 1000:
            break

        torch.cuda.empty_cache()
        gc.collect()

        label_set = torch.stack([t.label_set for t in target], dim=0).to(device)
        counts = torch.stack([t.count for t in target], dim=0).to(device)
        boxes = [t.bbox.to(device) for t in target]
        labels = [t.label.to(device) for t in target]
        number_boxes = np.stack([len(t) for t in target], axis=0)
        max_number_boxes = np.max(number_boxes)
        batch_size = label_set.size(0)

        predicted_boxes = torch.zeros((batch_size, max_number_boxes, 4)).to(device)
        for step in range(max_number_boxes):
            has_box = number_boxes > step

            current_label_set = label_set[has_box, :]
            current_counts = counts[has_box, :]

            all_boxes = [boxes[i] for i, has in enumerate(has_box) if has]
            all_labels = [labels[i] for i, has in enumerate(has_box) if has]
            current_label = torch.stack([l[step] for l in all_labels], dim=0).to(device)
            current_label = label_encodings[current_label.long() - 1]
            current_box = torch.stack([b[step] for b in all_boxes], dim=0).to(device)

            if step == 0:
                previous_labels = torch.zeros((batch_size, 0, 7)).to(device)
                previous_boxes = torch.zeros((batch_size, 0, 4)).to(device)
            else:
                previous_labels = torch.stack([l[step - 1] for l in all_labels], dim=0).unsqueeze(1)
                previous_labels = label_encodings[previous_labels.long() - 1]
                previous_boxes = torch.stack([b[step - 1] for b in all_boxes], dim=0).unsqueeze(1)

            state = (h[has_box].unsqueeze(0), c[has_box].unsqueeze(0)) if step > 1 else None
            predicted_boxes_step, kl_divergence, z, state = model(current_box, current_label_set, current_label,
                                                                  previous_labels, previous_boxes, state=state)
            predicted_boxes[has_box, step] = predicted_boxes_step

            box_loss_step = loss(predicted_boxes_step, current_box)
            losses = box_loss_step if losses is None else torch.cat([losses, box_loss_step])

            box_losses.append(box_loss_step.reshape(-1))
            divergence_losses.append(kl_divergence.reshape(-1))

            if state is not None:
                h, c = torch.zeros((batch_size, 128)).to(device), torch.zeros((batch_size, 128)).to(device)
                h[has_box, :] = state[0][-1]
                c[has_box, :] = state[1][-1]

        for i in range(batch_size):
            number = filename[0].replace(".gpickle", "")
            save_path_1 = os.path.join(save_dir, f"ground_truth_{number}.png")    # 실제 결과 저장 경로
            save_path_2 = os.path.join(save_dir, f"prediction_{number}.png")      # 예측 결과 저장 경로

            plot_layout(
                boxes[i].detach().cpu().numpy(),
                predicted_boxes[i].detach().cpu().numpy(),
                labels[i].detach().cpu().numpy(),
                500,
                500,
                colors=colors,
                save_path_1=save_path_1,
                save_path_2=save_path_2
            )

            # 저장된 결과를 pickle 파일로 저장
            with open(save_path_1.replace('.png', '.pkl'), 'wb') as file:
                pickle.dump(predicted_boxes[i].detach().cpu().numpy().tolist(), file)

            with open(save_path_2.replace('.png', '.pkl'), 'wb') as file:
                pickle.dump(boxes[i].detach().cpu().numpy().tolist(), file)

    average_loss = torch.mean(losses)
    print(f"validation: average loss: {average_loss}")
    count_losses = torch.cat(box_losses)
    divergence_losses = torch.cat(divergence_losses)
    loss_epoch = torch.mean(count_losses) + torch.mean(divergence_losses)

    return loss_epoch.item()


class GaussianLogLikelihood(nn.Module):
    def __init__(self):
        super(GaussianLogLikelihood, self).__init__()

        self.var = 0.02 ** 2

    def forward(self, predicted, expected):
        error = torch.mean((predicted - expected) ** 2, dim=-1)
        return error


class AutoregressiveBoxVariationalAutoencoder(nn.Module):
    def __init__(self, number_labels, conditioning_size, representation_size):
        super(AutoregressiveBoxVariationalAutoencoder, self).__init__()

        self.representation_size = representation_size

        self.encoder = AutoregressiveBoxEncoder(number_labels, conditioning_size, representation_size)
        self.decoder = AutoregressiveBoxDecoder(conditioning_size, representation_size)

    def sample(self, mu, log_var):
        batch_size = mu.size(0)
        device = mu.device

        standard_normal = torch.randn((batch_size, self.representation_size), device=device)
        z = mu + standard_normal * torch.exp(0.5 * log_var)

        kl_divergence = -0.5 * torch.sum(
            1 + log_var - (mu ** 2) - torch.exp(log_var), dim=1)

        return z, kl_divergence

    def forward(self, x, label_set, current_label, labels_so_far, boxes_so_far, state=None):
        mu, s, condition, state = self.encoder(x, label_set, current_label, labels_so_far, boxes_so_far, state)

        z, kl_divergence = self.sample(mu, s)
        z = torch.normal(mean=0, std=1, size=(1, self.representation_size)).to(device=device)
        boxes = self.decoder(z, condition)

        return boxes, kl_divergence, z, state

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Box VAE Test')
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")
    parser.add_argument("--test_json", default="instances_test.json", help="/path/to/test/json")
    parser.add_argument("--max_length", type=int, default=128, help="max length for dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epoch", required=True, help="checkpoint to evaluate")
    parser.add_argument("--save_dir", default="./visualization", help="directory to save visualizations")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    collator = BatchCollator()
    test_dataset = LayoutDataset(args.test_json, args.max_length)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator)

    NUMBER_LABELS = test_dataset.number_labels
    colors = gen_colors(NUMBER_LABELS)

    label_encodings = torch.eye(NUMBER_LABELS).float().to(device)
    box_loss = GaussianLogLikelihood().to(device)

    autoencoder = AutoregressiveBoxVariationalAutoencoder(
        NUMBER_LABELS,
        conditioning_size=128,
        representation_size=32).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(args.log_dir, "checkpoints", 'epoch_%d.pth' % int(args.epoch))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Evaluate the model on the test set and visualize results
    test_loss = evaluate_and_visualize(autoencoder, test_loader, box_loss, save_dir=args.save_dir, colors=colors)
    print(f"Test Loss: {test_loss}")
