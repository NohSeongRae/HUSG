import os
import shutil
import paths
from tqdm import tqdm
def copy_files(source_dir, dest_dir, num_files=100):
    # 소스 디렉토리에 있는 파일 목록을 가져옵니다.
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 복사할 파일 수를 제한합니다.
    files_to_copy = files[:num_files]

    # 지정된 파일들을 대상 디렉토리로 복사합니다.
    for file in tqdm(files_to_copy, desc='remaining files'):
        shutil.copy2(os.path.join(source_dir, file), dest_dir)



copy_files(paths.boundarymask_all, paths.boundarymask_sample)
copy_files(paths.insidemask_all, paths.insidemask_sample)
copy_files(paths.centroidmask_all, paths.centroidmask_sample)