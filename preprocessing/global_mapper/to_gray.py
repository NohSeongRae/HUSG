import os
from PIL import Image
import matplotlib.pyplot as plt

def convert_images_to_grayscale(input_folder, output_folder):
    # 출력 폴더가 존재하지 않으면 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더에서 모든 파일을 순회합니다.
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 이미지를 불러옵니다.
            image = Image.open(input_path)

            # 이미지를 흑백으로 변환합니다.
            gray_image = image.convert('L')

            # 흑백 이미지를 저장합니다.
            gray_image.save(output_path)

            # 변환된 이미지를 표시합니다.
            plt.figure(figsize=(8, 8))

            # 원본 이미지
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(image)
            plt.axis('off')

            # 흑백 이미지
            plt.subplot(1, 2, 2)
            plt.title('Grayscale Image')
            plt.imshow(gray_image, cmap='gray')
            plt.axis('off')

            plt.show()

# 사용 예시
input_folder = './gb_figure_pred_sample'  # 입력 폴더 경로를 입력하세요.
output_folder = './gb_figure_pred_sample_gray'  # 출력 폴더 경로를 입력하세요.

convert_images_to_grayscale(input_folder, output_folder)
