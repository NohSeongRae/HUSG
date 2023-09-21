import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from boundarymask import boundarymask
from buildingmask import buildingmask
from insidemask import insidemask
# from boundarybuildingmask import boundarybuildingmask
from buildingcentroid import buildingcentroid

# city_names_USA = ["atlanta", "dallas", "dublin", "houston", "lasvegas", "littlerock", "minneapolis", "phoenix", "portland", "richmond", "sanfrancisco", "washington"]

# city_names_all = ["barcelona", "budapest", "firenze", "manchester", "milan", "nottingham", "paris", "singapore", "toronto", "vienna", "zurich"]
# city_names_all = ["losangeles", "miami", "seattle", "boston", "providence", "tampa"]

# city_names_all = ["pittsburgh"]

city_names = ["atlanta", "dallas", "dublin", "houston", "lasvegas", "littlerock", "minneapolis", "phoenix", "portland", "richmond", "sanfrancisco", "washington", "philadelphia",
              "barcelona", "budapest", "firenze", "manchester", "milan", "nottingham", "paris", "singapore", "toronto", "vienna", "zurich",
              "miami", "seattle", "boston", "providence", "tampa", "pittsburgh"]

for i in range(len(city_names)):
    buildingcentroid(city_names[i], image_size=64)
    boundarymask(city_names[i], image_size=512)
    buildingmask(city_names[i], image_size=512)
    insidemask(city_names[i], image_size=512)

    # boundarybuildingmask(city_names[i])