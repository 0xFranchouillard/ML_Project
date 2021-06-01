import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

PATH = os.path.realpath('../../DATASET')

if __name__ == '__main__':
    filenames = [f for f in os.listdir(PATH)]
    dataset = []
    x = 500
    y = 500
    for f in filenames:
        f = np.array(Image.open(PATH + '\\' + f).resize((x, y)))
        dataset.append(f)
    for i in dataset:
        plt.imshow(i)
        plt.show()