import numpy as np

def load_matrix_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Split each line by whitespace (default) and convert to float
        matrix = [list(map(float, line.strip().split())) for line in lines]
    return np.array(matrix)

# Load matrices from files
LOGRNEW_matrix = load_matrix_from_file("C:/Users/rlaqhdrb/Desktop/Automatic-Unpaired-Shape-Deformation-Transfer/ACAP_linux/test/LOGRNEW.txt")
S_matrix = load_matrix_from_file("C:/Users/rlaqhdrb/Desktop/Automatic-Unpaired-Shape-Deformation-Transfer/ACAP_linux/test/S.txt")

# print(LOGRNEW_matrix[0][1234])
# print(len(S_matrix[0]))

matrix_size = 9  # 3x3 matrix

extracted_logR_values = []
extracted_scale_values = []

logR_data = LOGRNEW_matrix[0]
scale_data = S_matrix[0]

# logR 데이터에서 원하는 값을 추출
for i in range(0, len(logR_data), matrix_size):
    matrix_values = logR_data[i:i+matrix_size]

    # logR_values = matrix_values
    logR_values = [matrix_values[1], matrix_values[2], matrix_values[5]]
    extracted_logR_values.append(logR_values)

# scale 데이터에서 원하는 값을 추출
for i in range(0, len(scale_data), matrix_size):
    matrix_values = scale_data[i:i+matrix_size]

    scale_values = [matrix_values[0], matrix_values[1], matrix_values[2], matrix_values[4], matrix_values[5], matrix_values[8]]
    extracted_scale_values.append(scale_values)

print("Extracted logR values for each matrix:", extracted_logR_values)
# print("Extracted Scale values for each matrix:", extracted_scale_values)
