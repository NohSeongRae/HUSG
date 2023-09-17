import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc.cityname import city_name
from lengtharea import lengtharea
from lengthnum import lengthnum
from ratio import ratio
from blocksemantic import block_category
from density import density
from boundarysize import boundarysize
from buildingnum import buildingnum

from etc import filepath as filepath

if not os.path.exists(filepath.stat):
    os.makedirs(filepath.stat)

city_name_list = ["atlanta", "barcelona", "budapest", "dallas", "dublin", "firenze", "houston", "lasvegas", "littlerock", "manchester", "milan", "minneapolis",
                  "nottingham", "paris", "philadelphia", "phoenix", "portland", "richmond", "saintpaul", "sanfrancisco", "singapore", "toronto", "vienna",
                  "washington", "zurich"]

rangenum1 = "_0_15"
rangenum2 = "_15_25"

# lengtharea(city_name_list[:15], rangenum1)

# lengthnum(city_name_list[:15], rangenum1)

# ratio(city_name_list[:15], rangenum1)

# block_category(city_name_list[:15], rangenum1)

# density(city_name_list[:15], rangenum1)

# boundarysize(city_name_list[:15], rangenum1)

# buildingnum(city_name_list[:15], rangenum1)

# lengtharea(city_name_list[15:], rangenum2)

# lengthnum(city_name_list[15:], rangenum2)

# ratio(city_name_list[15:], rangenum2)

# block_category(city_name_list[15:], rangenum2)

# density(city_name_list[15:], rangenum2)

# boundarysize(city_name_list[15:], rangenum2)

# buildingnum(city_name_list[15:], rangenum2)

# print("Step 10: Stat computation completed")
# print(f"{city_name} is ready")