import fnmatch
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PATH = os.path.realpath('../../DATASET')
FILENUMBER = 0
CPT = 0


def resize_dataset(x, y):
    filenames = [f for f in os.listdir(PATH)]
    for f in filenames:
        image = Image.open(PATH + '\\' + f).resize((x, y))
        image.save(PATH + '\\' + f)


def load_image(filepath):
    f = np.array(Image.open(filepath))
    return f


def load_images(label, dataset_img, dataset_label):
    filenames = [f for f in os.listdir(PATH + '\\' + label)]
    global CPT
    for f in filenames:
        dataset_img.append(load_image(PATH + '\\' + label + '\\' + f))
        dataset_label.append(label)
        CPT += 1
        print(f'{CPT / FILENUMBER:2.2%} fichiers chargés')


def load_dataset():
    labelnames = [f for f in os.listdir(PATH)]
    dataset_img = []
    dataset_label = []
    threads = [None] * len(labelnames)
    global FILENUMBER
    for i in range(len(threads)):
        FILENUMBER += len(fnmatch.filter(os.listdir(PATH + '\\' + labelnames[i] + '\\'), '*.*'))
        threads[i] = threading.Thread(target=load_images, args=(labelnames[i], dataset_img, dataset_label))
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
    return dataset_img, dataset_label

