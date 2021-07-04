import fnmatch
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

PATH = os.path.realpath('../../DATASET')
FILENUMBER = 0
CPT = 0


def resize_image(filepath, x, y):
    image = Image.open(filepath).resize((x, y))
    image.save(filepath)


def resize_images(label, x, y):
    filenames = [f for f in os.listdir(PATH + '\\' + label)]
    global CPT
    for f in filenames:
        resize_image(PATH + '\\' + label + '\\' + f, x, y)
        CPT += 1
        print(f'{CPT / FILENUMBER:2.2%} fichiers resizés')


def resize_dataset(x, y):
    global CPT, FILENUMBER
    CPT = 0
    FILENUMBER = 0
    labelnames = [f for f in os.listdir(PATH)]
    threads = [None] * len(labelnames)
    for i in range(len(threads)):
        FILENUMBER += len(fnmatch.filter(os.listdir(PATH + '\\' + labelnames[i] + '\\'), '*.*'))
        threads[i] = threading.Thread(target=resize_images, args=(labelnames[i], x, y))
        threads[i].start()


def load_image(filepath, train_ratio):
    if random.uniform(0, 1) > train_ratio:
        return np.array(Image.open(filepath)), False
    return np.array(Image.open(filepath)), True


def load_images(label, dataset_img, dataset_label, train_ratio, testing_dataset_img, testing_dataset_label):
    filenames = [f for f in os.listdir(PATH + '\\' + label)]
    global CPT
    for f in filenames:
        image, training = load_image(PATH + '\\' + label + '\\' + f, train_ratio)
        if training:
            dataset_img.append(image)
            dataset_label.append(label)
        else:
            testing_dataset_img.append(image)
            testing_dataset_label.append(label)
        CPT += 1
        print(f'{CPT / FILENUMBER:2.2%} fichiers chargés')


def load_dataset(train_ratio):
    global CPT
    global FILENUMBER
    CPT = 0
    FILENUMBER = 0
    labelnames = [f for f in os.listdir(PATH)]
    training_dataset_img = []
    training_dataset_label = []
    testing_dataset_img = []
    testing_dataset_label = []
    threads = [None] * len(labelnames)

    for i in range(len(threads)):
        FILENUMBER += len(fnmatch.filter(os.listdir(PATH + '\\' + labelnames[i] + '\\'), '*.*'))
        threads[i] = threading.Thread(target=load_images, args=(labelnames[i], training_dataset_img, training_dataset_label,train_ratio, testing_dataset_img, testing_dataset_label))
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()

    return training_dataset_img, training_dataset_label, testing_dataset_img, testing_dataset_label

