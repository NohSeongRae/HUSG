import cv2
import numpy as np

for i in range(1, num_file + 1):
    image_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'BoundaryImage', f'{city_name}_boundary_image{i}.png')

    if os.path.exists(image_filepath):
        image_path = image_filepath
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # 첫 번째 마스크: 경계선은 검은색, 그 외는 하얀색
        mask1 = cv2.bitwise_not(binary_image)

        # 두 번째 마스크 생성을 위해 이미지의 가장자리에 검은색 테두리 추가
        padded_image = cv2.copyMakeBorder(binary_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

        # 경계선의 내부를 채우기 위해 flood fill 사용
        mask2 = padded_image.copy()
        cv2.floodFill(mask2, None, (0, 0), 0)

        # 테두리 제거
        mask2 = mask2[1:-1, 1:-1]

        save_path_boundary = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'boundarymask', f'{city_name}_boundarymask{i}.png')
        save_path_inside = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'insidemask', f'{city_name}_boundarymask{i}.png')

        cv2.imwrite(save_path_boundary, mask1)
        cv2.imwrite(save_path_inside, mask2)