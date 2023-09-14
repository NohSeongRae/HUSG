import torch
import clip
from PIL import Image
print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
test_model, preprocess=clip.load('ViT-B/32', device=device, jit=False)

test_image=preprocess(Image.open('clip/schedule.png')).unsqueeze(0).to(device)
text=clip.tokenize(["a middle school", "a high school", "a college"]).to(device)

with torch.no_grad():
    image_feature=test_model.encode_image(test_image)
    text_feature=test_model.encode_text(text)

    logits_per_image, logits_per_text=test_model(test_image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)

# import torch
# import clip
# from PIL import Image
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# image = preprocess(Image.open("schedule.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a middle school", "a high school", "a college"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# print("Label probs:", probs)