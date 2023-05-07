import os
from cityname import city_name, location
from data_download import data_download
from extract import extract
from get_boundary import get_boundary

output_directories = [
    city_name + "_dataset",
    city_name + "_dataset/Boundaries",
    city_name + "_dataset/Buildings",
    city_name + "_dataset/NLD",
    city_name + "_dataset/Image"
]

for directory in output_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

data_download(city_name, location)
extract(city_name)
get_boundary(city_name, location)


