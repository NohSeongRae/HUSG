import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc.cityname import city_name
from boundaryimage import boundary_image
from buildingimage import building_image
from etc import filepath as filepath

if not os.path.exists(filepath.boundaryimage):
    os.makedirs(filepath.boundaryimage)

if not os.path.exists(filepath.buildingimage):
    os.makedirs(filepath.buildingimage)

boundary_image(city_name)
print("boundary-only image generated")

building_image(city_name)
print("building-only image generated")
