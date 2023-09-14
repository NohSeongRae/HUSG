import os
import shutil


def copy_files(source_dir, dest_dir, num_files=100):
    # 소스 디렉토리에 있는 파일 목록을 가져옵니다.
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 복사할 파일 수를 제한합니다.
    files_to_copy = files[:num_files]

    # 지정된 파일들을 대상 디렉토리로 복사합니다.
    for file in files_to_copy:
        shutil.copy2(os.path.join(source_dir, file), dest_dir)


# 사용 예:
# source_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarybuildingmask')  # 원본 폴더 경로를 입력하세요.
# destination_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarybuildingmask_sample')   # 복사할 대상 폴더 경로를 입력하세요.
# copy_files(source_directory, destination_directory)
#
# source_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarymask')  # 원본 폴더 경로를 입력하세요.
# destination_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarymask_sample')   # 복사할 대상 폴더 경로를 입력하세요.
# copy_files(source_directory, destination_directory)
#
# source_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'buildingmask')  # 원본 폴더 경로를 입력하세요.
# destination_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'buildingmask_sample')   # 복사할 대상 폴더 경로를 입력하세요.
# copy_files(source_directory, destination_directory)
#
# source_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'insidemask')  # 원본 폴더 경로를 입력하세요.
# destination_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'insidemask_sample')   # 복사할 대상 폴더 경로를 입력하세요.
# copy_files(source_directory, destination_directory)
#
# source_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'inversemask')  # 원본 폴더 경로를 입력하세요.
# destination_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'inversemask_sample')   # 복사할 대상 폴더 경로를 입력하세요.
# copy_files(source_directory, destination_directory)

source_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'centroidmask')  # 원본 폴더 경로를 입력하세요.
destination_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'centroidmask_sample')   # 복사할 대상 폴더 경로를 입력하세요.
copy_files(source_directory, destination_directory)