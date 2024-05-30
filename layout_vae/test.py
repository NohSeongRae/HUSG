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

from box import AutoregressiveBoxEncoder, AutoregressiveBoxDecoder
from layout import BatchCollator, LayoutDataset


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)] for x in palette]
    return rgb_triples


def plot_layout(real_boxes, predicted_boxes, labels, width, height, colors=None):
    blank_image = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    blank_draw = ImageDraw.Draw(blank_image)

    number_boxes = real_boxes.shape[0]
    for i in range(number_boxes):
        real_box = real_boxes[i].tolist()
        predicted_box = predicted_boxes[i].tolist()
        label = int(labels[i])

        real_x1, real_y1 = int(real_box[0] * width), int(real_box[1] * height)
        real_x2, real_y2 = real_x1 + int(real_box[2] * width), real_y1 + int(real_box[3] * height)

        predicted_x1, predicted_y1 = int(predicted_box[0] * width), int(predicted_box[1] * height)
        predicted_x2, predicted_y2 = predicted_x1 + int(predicted_box[2] * width), predicted_y1 + int(
            predicted_box[3] * height)

        real_color = (0, 0, 0)
        if colors is not None:
            real_color = tuple(colors[label])

        blank_draw.rectangle([(real_x1, real_y1), (real_x2, real_y2)], outline=real_color)
        blank_draw.rectangle([(predicted_x1, predicted_y1), (predicted_x2, predicted_y2)], outline=(0, 0, 0))

    return blank_image


def evaluate(model, loader, loss, prefix='', colors=None):
    errors = []
    model.eval()
    losses = None
    box_losses = []
    divergence_losses = []

    for batch_i, (indexes, target) in tqdm(enumerate(loader)):
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
        boxes = self.decoder(z, condition)

        return boxes, kl_divergence, z, state


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Box VAE Test')
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")
    parser.add_argument("--test_json", default="test_dataset.json", required=True, help="/path/to/test/json")
    parser.add_argument("--max_length", type=int, default=128, help="max length for dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epoch", required=True, help="checkpoint to evaluate")

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
    checkpoint_path = os.path.join(args.log_dir, "checkpoints", 'epoch_%d.pth' % args.epoch)
    checkpoint = torch.load(args.evaluate_checkpoint, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded checkpoint from {args.evaluate_checkpoint}")

    # Evaluate the model on the test set
    test_loss = evaluate(autoencoder, test_loader, box_loss, colors=colors)
    print(f"Test Loss: {test_loss}")
