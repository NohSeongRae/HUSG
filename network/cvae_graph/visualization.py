import numpy as np
import matplotlib.pyplot as plt
import os

def plot(pos, size, theta):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    print(pos.shape, size.shape, theta.shape)
    for i in range(len(pos)):
        print(pos[i], size[i], theta[i])

    print(1)