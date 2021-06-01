import os
import numpy as np
from PIL import Image

PATH = os.path.realpath('../../DATASET')

def resize_dataset(x, y):
    filenames = [f for f in os.listdir(PATH)]
    for f in filenames:
        image = Image.open(PATH + '\\' + f).resize((x, y))
        image.save(PATH + '\\' + f)

def load_dataset():
    filenames = [f for f in os.listdir(PATH)]
    dataset = []
    for f in filenames:
        f = np.array(Image.open(PATH + '\\' + f))
        dataset.append(f)
    return dataset
