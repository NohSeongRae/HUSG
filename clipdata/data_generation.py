import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc.cityname import city_name
from Image import image
# from NLDtest import NLD

image(city_name)
print("Step 7: Image generation compeleted")
# NLD(city_name, filelist)
# print("Step 8: NLD generated PASSED")
