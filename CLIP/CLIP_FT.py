import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import clip
from PIL import Image
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc.cityname import city_name

writer = SummaryWriter('runs/clipft')

def restore(self):
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            assert name in self.backup
            param.data = self.backup[name]
    self.backup = {}


class EMA():
    def __init__(self, model, decay):
        """
        EMA는 큰 데이터에 대해 학습된 모델을 fine-tuning할 때 overfitting 현상을 막아줌 
        :param model: EMA를 계산하고 적용할 대상 model
        :param decay: EMA의 감쇠율 (EMA를 계산할 때 사용)
        shadow: moving average 값이 저장되는 dictionary
        backup: 모델의 원래 파라미터 값을 저장
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """
        학습 시작 전
        모델의 모든 파라미터를 'shadow' dictionary에 등록하고 초기 moving average 값을 설정
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        각 학습 step 마다 호출
        모델의 모든 파라미터에 대해 EMA를 업데이트
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        학습 중 모델의 파라미터를 moving average 값으로 교체
        모델의 모든 파라미터를 'shadow' dictionary에 저장된 moving average 값으로 교체
        기존 파라미터 값은 'backup' dictionary에 저장
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        학습 중 모델의 파라미터를 임시로 moving average 값으로 교체한 후 원래대로 복원할 때 호출
        'backup' dictionary에 저장된 원래의 파라미터 값을 모델에 복원
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

parser = argparse.ArgumentParser(description='Fine-tuning CLIP on custom data')
parser.add_argument('--device', type=str, default="cuda:0", help='device to use for training')
parser.add_argument('--model_name', type=str, default="ViT-B/32", help='name of the clip model to use')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='learning rate for the Adam optimizer')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for the Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for the Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-6, help='epsilon for the Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay for the Adam optimizer')
parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs')
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.model_name, device=device, jit=False)

ema = EMA(model, decay=0.9998)

ema.register()


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(
            list_txt)  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title

# data load

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Image')
files = os.listdir(dir_path)
filenum = len(files)

list_image_path = []

for index in range(1, filenum+1):
    image_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                  f'{city_name}_dataset', 'Image',
                                  f'{city_name}_buildings_image{index}.png')
    list_image_path.append(image_filename)


nld_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                            f'{city_name}_dataset', 'NLD', f'{city_name}_NLD.txt')

with open(nld_filename, 'r') as f:
    list_txt = f.read().split('\n')
    list_txt = [line for line in list_txt if line]

dataset = image_title_dataset(list_image_path, list_txt)

train_dataloader = DataLoader(dataset, batch_size=args.batch_size)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

total_steps = len(train_dataloader) * args.epochs

num_training_samples = len(dataset)
steps_per_epoch = num_training_samples // args.batch_size

"""
warmup ? 학습 초기에 학습률을 점차적으로 증가 / 이후에는 감소 
num_warmup_steps : warmup을 진행할 step 수 (이 동안 학습률이 선형적으로 증가)
num_training_steps : 총 훈련 step 수 
"""
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = args.warmup_epochs * steps_per_epoch,
                                            num_training_steps = total_steps)

for epoch in range(args.epochs):
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()

        # Before the optimizer step, save the model parameters
        ema.apply_shadow()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        # After the optimizer step, restore the model parameters
        ema.restore()
        # Update the EMA of the parameters
        ema.update()

        scheduler.step()

        # Add loss to TensorBoard
        writer.add_scalar('Loss/train', total_loss.item(), epoch * len(train_dataloader) + i)

ema.restore()

torch.save(model.state_dict(), 'CLIP/clipft.pth')
writer.close()