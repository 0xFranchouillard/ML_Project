import fnmatch
import os
import threading
import matplotlib.pyplot as plt
import numpy
import numpy as np
from PIL import Image, ImageOps
import random

FILENUMBER = 0
CPT = 0


def resize_image(filepath, x):
    image = Image.open(filepath)
    # width, height = image.size
    # ratio = float(width / height)
    image = image.resize((x, x))
    image = ImageOps.grayscale(image)
    image.save(filepath)


def resize_images(path, label, x):
    filenames = [f for f in os.listdir(path + '\\' + label)]
    global CPT
    for f in filenames:
        print(os.path.join(path, label, f))
        resize_image(os.path.join(path, label, f), x)
        CPT += 1
        print(f'{CPT / FILENUMBER:2.2%} fichiers resizés')


def resize_dataset(path, x):
    global CPT, FILENUMBER
    CPT = 0
    FILENUMBER = 0
    labelnames = [f for f in os.listdir(path)]
    threads = [None] * len(labelnames)
    for i in range(len(threads)):
        FILENUMBER += len(fnmatch.filter(os.listdir(path + '\\' + labelnames[i] + '\\'), '*.*'))
        threads[i] = threading.Thread(target=resize_images, args=(path, labelnames[i], x))
        threads[i].start()


def load_image(filepath):
    return np.array(Image.open(filepath))


def load_images(path, label, dataset_img, dataset_label):
    filenames = [f for f in os.listdir(path + '\\' + label)]
    global CPT
    for f in filenames:
        dataset_img.append(load_image(path + '\\' + label + '\\' + f))
        dataset_label.append(label)
        CPT += 1
        print(f'{CPT / FILENUMBER:2.2%} fichiers chargés')


def load_dataset(path):
    global CPT
    global FILENUMBER
    CPT = 0
    FILENUMBER = 0
    labelnames = [f for f in os.listdir(path)]
    dataset_img = []
    dataset_label = []
    classes = [None] * len(labelnames)

    for i in range(len(classes)):
        FILENUMBER += len(fnmatch.filter(os.listdir(path + '\\' + labelnames[i] + '\\'), '*.*'))
        classes[i] = threading.Thread(target=load_images, args=(
            path, labelnames[i], dataset_img, dataset_label))
        classes[i].start()
    for i in range(len(classes)):
        classes[i].join()

    return dataset_img, dataset_label


"""
training_dataset_img, training_dataset_label = load_dataset(os.path.realpath('../../DATASET_80px_TRAIN'))
testing_dataset_img, testing_dataset_label = load_dataset(os.path.realpath('../../DATASET_80px_TEST'))
size_first_layer = sum([numpy.prod(img.shape) for img in training_dataset_img])
resize_dataset(os.path.realpath('../../DATASET_80px_TEST'), 32)
"""