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

from etc import filepath as filepath

if not os.path.exists(filepath.stat):
    os.makedirs(filepath.stat)

lengtharea(city_name)
lengthnum(city_name)
ratio(city_name)
block_category(city_name)
density(city_name)

print("Step 10: Stat computation completed")
print(f"{city_name} is ready")