import os
import re
import csv

mask_type = ['boundary', 'inside', 'centroid']

boundary_mask_folder = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', 'mask_all', 'boundarymask')
inside_mask_folder = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', 'mask_all', 'insidemask')
centroid_mask_folder = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', 'mask_all', 'centroidmask')


def check_extra_files(directory_path):
    all_items = os.listdir(directory_path)
    filtered_files = [f for f in all_items if os.path.isfile(os.path.join(directory_path, f))]

    extra_files = set(filtered_files) - set(all_items)
    print("Extra files:", extra_files)

    return len(filtered_files)


def check_mismatch(directory_path):
    all_items = os.listdir(directory_path)
    filtered_files = [f for f in all_items if os.path.isfile(os.path.join(directory_path, f))]

    if len(all_items) != len(filtered_files):
        # 불일치 항목 출력
        mismatched_items = set(all_items) - set(filtered_files)
        print("Mismatched items:", mismatched_items)

    return len(filtered_files)

# file_count = check_mismatch(centroid_mask_folder)
# print("Total file count:", file_count)



def extract(filename):
    string_part = ''.join(re.findall(r'[A-Za-z]+', filename))

    numbers = re.findall(r'\d+', filename)
    number_part = int(numbers[0]) if numbers else 0

    return (string_part, number_part)


def assign_index(directory_path, csv_filename):
    filenames = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    filenames.sort(key=extract)

    print(len(filenames))

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Index', 'Filename'])

        for index, filename in enumerate(filenames, 1):
            csv_writer.writerow([index, filename])

boundary_csv_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', 'mask_file_index', 'boundary_mask_index.csv')
inside_csv_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', 'mask_file_index', 'inside_mask_index.csv')
centroid_csv_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', 'mask_file_index', 'centroid_mask_index.csv')

# assign_index(boundary_mask_folder, boundary_csv_filename)
# assign_index(inside_mask_folder, inside_csv_filename)
assign_index(centroid_mask_folder, centroid_csv_filename)

