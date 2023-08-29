import sys
# sys.path.append('build')
# from get_mesh import *
import numpy as np


# Load the feature data
feature = np.load('temp.npy')

print(feature.shape)

print('output.npy feature: ', len(feature))