import os

from data_download import data_download
from extract import extract
from POI import POI
from get_boundary import get_boundary
from get_building import get_building

city_name = "littlerock"
location = "Little Rock, United States"


output_directories = [
    city_name + "_dataset",
    city_name + "_dataset/Boundaries",
    city_name + "_dataset/Buildings",
]

for directory in output_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


# data_download(city_name, location)
# extract(city_name)
# POI(city_name)
filenum = get_boundary(city_name, location)
# get_building(city_name, filenum)
