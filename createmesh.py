import numpy as np

# 9x592 차원의 가상의 데이터 생성
feature_data = np.random.rand(9, 592)

# 데이터를 .dat 파일로 저장
dat_file_path = 'feature_data.dat'
feature_data.tofile(dat_file_path)

