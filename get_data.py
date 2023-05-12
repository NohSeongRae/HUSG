import os
from cityname import city_name, location
from data_download import data_download
from extract import extract
from get_boundary import get_boundary
import filepath

# get_data.py에선 data_download, extract, get_boundary를 진행함
# 과정이 분할된 이유는, 주로 오류로 인한 중단시에 중간 단계의 결과물을 이어 진행하기 위함

# 폴더가 없다면, 새로 생성
for directory in filepath.output_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

print(f"get_data start: {location}")
data_download(city_name, location)
extract(city_name)
get_boundary(city_name, location)
