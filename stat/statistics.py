import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc.cityname import city_name
from lengtharea import lengtharea
from lengthnum import lengthnum
from ratio import ratio
from block import block_category
from stat.density import density

lengtharea(city_name)
lengthnum(city_name)
ratio(city_name)
block_category(city_name)
density(city_name)