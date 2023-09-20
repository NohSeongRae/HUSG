import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from boundarymask import boundarymask
from buildingmask import buildingmask
from insidemask import insidemask
from boundarybuildingmask import boundarybuildingmask
from buildingcentroid import buildingcentroid

# city_names = ["atlanta", "dallas", "dublin", "houston", "lasvegas", "littlerock", "minneapolis", "phoenix", "philadelphia", "portland", "richmond", "sanfrancisco", "washington"]

city_names = ["barcelona", "budapest", "firenze", "manchester", "milan", "nottingham", "paris", "singapore", "toronto", "vienna", "zurich"]

for i in range(len(city_names)):
    buildingcentroid(city_names[i])

    print("bounarymask start")
    boundarymask(city_names[i])
    print("buildingmask start")
    buildingmask(city_names[i])
    print("insidemask start")
    insidemask(city_names[i])
    # print("inversebuildingmask start")
    # inversebuildingmask(city_names[i])
    # print("boundarybuildingmask start")
    # boundarybuildingmask(city_names[i])
