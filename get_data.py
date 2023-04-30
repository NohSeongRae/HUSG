import os

from data_download import data_download
from extract import extract
from POI import POI
from get_boundary import get_boundary
from get_building import get_building

city_name = ["budapest"]
location = ["Budapest, Hungary"]

for i in range(len(city_name)):
    output_directories = [
        city_name[i] + "_dataset",
        city_name[i] + "_dataset/Boundaries",
        city_name[i] + "_dataset/Buildings",
    ]

    for directory in output_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    data_download(city_name[i], location[i])
    extract(city_name[i])
    POI(city_name[i])
    filenum = get_boundary(city_name[i], location[i])
    get_building(city_name[i], filenum)
