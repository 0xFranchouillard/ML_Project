from ctypes import *
import datetime
import random
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from mpl_toolkits.mplot3d import Axes3D
from process_dataset import *

PATH_TO_SHARED_LIBRARY = "../../RUST_LIBRARY/lib_projet_annuel_machine_learning/target/release/lib_projet_annuel_machine_learning.dll"


def create_linear_model(my_lib, size):
    my_lib.create_linear_model.argtypes = [c_int]
    my_lib.create_linear_model.restype = POINTER(c_void_p)
    return my_lib.create_linear_model(size)


def save_linear_model(my_lib, model, path):
    my_lib.save_linear_model.argtypes = [POINTER(c_void_p),
                                         c_char_p]
    my_lib.save_linear_model.restype = None
    my_lib.save_linear_model(model,
                             path.encode("utf-8"))


def load_linear_model(my_lib, path):
    my_lib.load_linear_model.argtypes = [c_char_p]
    my_lib.load_linear_model.restype = POINTER(c_void_p)
    return my_lib.load_linear_model(path.encode("utf-8"))


def destroy_linear_model(my_lib, model):
    my_lib.destroy_linear_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_linear_model.restype = None
    my_lib.destroy_linear_model(model)


def predict_linear_model_classification(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_linear_model_classification.argtypes = [POINTER(c_void_p),
                                                           inputs_type]
    my_lib.predict_linear_model_classification.restype = c_float

    return my_lib.predict_linear_model_classification(model,
                                                      inputs_type(*inputs_float))


def train_rosenblatt_linear_model(my_lib, model, dataset_inputs, dataset_expected_outputs, iteration_count, alpha):
    dataset_inputs_float = [float(i) for i in dataset_inputs]
    dataset_inputs_type = len(dataset_inputs_float) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_rosenblatt_linear_model.argtypes = [POINTER(c_void_p),
                                                     dataset_inputs_type,
                                                     dataset_expected_outputs_type,
                                                     c_int,
                                                     c_float,
                                                     c_int]
    my_lib.train_rosenblatt_linear_model.restype = None

    my_lib.train_rosenblatt_linear_model(model,
                                         dataset_inputs_type(*dataset_inputs_float),
                                         dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                         iteration_count,
                                         float(alpha),
                                         len(dataset_inputs_float))


def test_classification_linear_model(my_lib):
    # Init dataset
    dataset_inputs = [
        [1, 4],
        [1, -4],
        [4, 4],
    ]
    dataset_expected_outputs = [
        1,
        1,
        -1
    ]

    # Création du model
    model = create_linear_model(my_lib, 2)

    for _ in range(5):
        points_x1_blue = []
        points_x2_blue = []

        points_x1_red = []
        points_x2_red = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                if predict_linear_model_classification(my_lib, model, [i, j]) == 1.0:
                    points_x1_blue.append(i)
                    points_x2_blue.append(j)
                else:
                    points_x1_red.append(i)
                    points_x2_red.append(j)

        plt.scatter(points_x1_blue, points_x2_blue, c='blue')
        plt.scatter(points_x1_red, points_x2_red, c='red')

        plt.scatter([p[0] for p in dataset_inputs[:2]], [p[1] for p in dataset_inputs[:2]], c='blue', s=100)
        plt.scatter([p[0] for p in dataset_inputs[2:]], [p[1] for p in dataset_inputs[2:]], c='red', s=100)
        plt.show()

        train_rosenblatt_linear_model(my_lib, model, np.array(dataset_inputs).flatten(), dataset_expected_outputs, 20,
                                      0.1)

    save_linear_model(my_lib, model, "linear_model_classification.json")
    model2 = load_linear_model(my_lib, "linear_model_classification.json")
    destroy_linear_model(my_lib, model)
    destroy_linear_model(my_lib, model2)


def predict_linear_model_regression(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_linear_model_regression.argtypes = [POINTER(c_void_p),
                                                       inputs_type]
    my_lib.predict_linear_model_regression.restype = c_float

    return my_lib.predict_linear_model_regression(model,
                                                  inputs_type(*inputs_float))


def train_regression_linear_model(my_lib, model, dataset_inputs, dataset_expected_outputs):
    dataset_inputs_float = [float(i) for i in dataset_inputs]
    dataset_inputs_type = len(dataset_inputs_float) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_regression_linear_model.argtypes = [POINTER(c_void_p),
                                                     dataset_inputs_type,
                                                     dataset_expected_outputs_type,
                                                     c_int]
    my_lib.train_regression_linear_model.restype = None

    my_lib.train_regression_linear_model(model,
                                         dataset_inputs_type(*dataset_inputs_float),
                                         dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                         len(dataset_inputs_float))


def test_regression_linear_model(my_lib):
    # Init dataset
    dataset_inputs = [
        1.0,
        3.0,
        4.0,
    ]
    dataset_expected_outputs = [
        2.0,
        3.0,
        7.0
    ]

    # Création du model
    model = create_linear_model(my_lib, 1)

    for _ in range(2):
        point_x = []
        point_y = []

        for i in range(-10, 11):
            point_x.append(float(i))
            point_y.append(predict_linear_model_regression(my_lib, model, [i]))

        plt.scatter(point_x, point_y)
        plt.scatter(dataset_inputs, dataset_expected_outputs, c="purple")
        plt.show()

        train_regression_linear_model(my_lib, model, dataset_inputs, dataset_expected_outputs)

    destroy_linear_model(my_lib, model)


def create_mlp_model(my_lib, struct_model):
    struct_model_int = [int(i) for i in struct_model]
    struct_model_type = len(struct_model_int) * c_int

    my_lib.create_mlp_model.argtypes = [struct_model_type,
                                        c_int]
    my_lib.create_mlp_model.restype = POINTER(c_void_p)

    return my_lib.create_mlp_model(struct_model_type(*struct_model_int), len(struct_model_int))


def save_mlp_model(my_lib, model, path):
    my_lib.save_mlp_model.argtypes = [POINTER(c_void_p), c_char_p]
    my_lib.save_mlp_model.restype = None
    my_lib.save_mlp_model(model, path.encode("utf-8"))


def load_mlp_model(my_lib, path):
    my_lib.load_mlp_model.argtypes = [c_char_p]
    my_lib.load_mlp_model.restype = POINTER(c_void_p)
    return my_lib.load_mlp_model(path.encode("utf-8"))


def destroy_mlp_model(my_lib, model):
    my_lib.destroy_mlp_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_mlp_model.restype = None
    my_lib.destroy_mlp_model(model)


def predict_mlp_model_classification(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_mlp_model_classification.argtypes = [POINTER(c_void_p),
                                                        inputs_type,
                                                        c_int]
    my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    result_predict = my_lib.predict_mlp_model_classification(model, inputs_type(*inputs_float), len(inputs_float))

    return np.ctypeslib.as_array(result_predict, (1,))


def train_classification_stochastic_backprop_mlp_model(my_lib, model, dataset_inputs, dataset_expected_outputs, alpha,
                                                       iterations_count):
    dataset_inputs_flattened = []
    for elt in dataset_inputs:
        dataset_inputs_flattened.append(elt[0])
        dataset_inputs_flattened.append(elt[1])
    dataset_inputs_flattened_type = len(dataset_inputs_flattened) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_classification_stochastic_backprop_mlp_model.argtypes = [POINTER(c_void_p),
                                                                          dataset_inputs_flattened_type,
                                                                          c_int,
                                                                          dataset_expected_outputs_type,
                                                                          c_int,
                                                                          c_float,
                                                                          c_int]
    my_lib.train_classification_stochastic_backprop_mlp_model.restype = None

    my_lib.train_classification_stochastic_backprop_mlp_model(model,
                                                              dataset_inputs_flattened_type(*dataset_inputs_flattened),
                                                              len(dataset_inputs_flattened),
                                                              dataset_expected_outputs_type(
                                                                  *dataset_expected_outputs_float),
                                                              len(dataset_expected_outputs_float),
                                                              alpha, iterations_count)


def test_classification_mlp_model(my_lib):
    # Init dataset
    dataset_inputs = [
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1]
    ]
    dataset_expected_outputs = [
        -1,
        -1,
        1,
        1
    ]

    # Création du model
    model = create_mlp_model(my_lib, [2, 3, 1])

    for _ in range(2):
        points_x1_blue = []
        points_x2_blue = []

        points_x1_red = []
        points_x2_red = []

        for i in range(-10, 11):
            for j in range(-10, 11):
                result_predict = predict_mlp_model_classification(my_lib, model, [i, j])
                if result_predict >= 0:
                    points_x1_blue.append(i / 5.0)
                    points_x2_blue.append(j / 5.0)
                else:
                    points_x1_red.append(i / 5.0)
                    points_x2_red.append(j / 5.0)

        plt.scatter(points_x1_blue, points_x2_blue, c='blue')
        plt.scatter(points_x1_red, points_x2_red, c='red')

        plt.scatter([p[0] for p in dataset_inputs[:2]], [p[1] for p in dataset_inputs[:2]], c='red', s=100)
        plt.scatter([p[0] for p in dataset_inputs[2:]], [p[1] for p in dataset_inputs[2:]], c='blue', s=100)
        plt.show()

        train_classification_stochastic_backprop_mlp_model(my_lib, model, dataset_inputs, dataset_expected_outputs,
                                                           float(0.01), 100000)

    save_mlp_model(my_lib, model, "mlp_model_classification.json")
    destroy_mlp_model(my_lib, model)


def predict_mlp_model_regression(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_mlp_model_regression.argtypes = [POINTER(c_void_p),
                                                    inputs_type,
                                                    c_int]
    my_lib.predict_mlp_model_regression.restype = POINTER(c_float)
    result_predict = my_lib.predict_mlp_model_regression(model, inputs_type(*inputs_float), len(inputs_float))

    return np.ctypeslib.as_array(result_predict, (1,))


def train_regression_stochastic_backprop_mlp_model(my_lib, model, dataset_inputs, dataset_expected_outputs):
    dataset_inputs_float = [float(i) for i in dataset_inputs]
    dataset_inputs_type = len(dataset_inputs_float) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_regression_stochastic_backprop_mlp_model.argtypes = [POINTER(c_void_p),
                                                                      dataset_inputs_type,
                                                                      c_int,
                                                                      dataset_expected_outputs_type,
                                                                      c_int,
                                                                      c_float,
                                                                      c_int]
    my_lib.train_regression_stochastic_backprop_mlp_model.restype = None

    my_lib.train_regression_stochastic_backprop_mlp_model(model,
                                                          dataset_inputs_type(*dataset_inputs_float),
                                                          len(dataset_inputs_float),
                                                          dataset_expected_outputs_type(
                                                              *dataset_expected_outputs_float),
                                                          len(dataset_expected_outputs_float),
                                                          float(0.01), 100000)


def test_regression_mlp_model(my_lib):
    # Init dataset
    dataset_inputs = [
        1,
        3,
        4
    ]
    dataset_expected_outputs = [
        2,
        3,
        7
    ]

    # Création du model
    model = create_mlp_model(my_lib, [1, 5, 5, 1])

    for _ in range(2):
        points_x = []
        points_y = []

        for i in range(-10, 11):
            points_x.append(float(i))
            points_y.append(predict_mlp_model_regression(my_lib, model, [i])[0])

        plt.plot(points_x, points_y)
        plt.scatter(dataset_inputs, dataset_expected_outputs, c="purple")
        plt.show()

        train_regression_stochastic_backprop_mlp_model(my_lib, model, dataset_inputs, dataset_expected_outputs)

    save_mlp_model(my_lib, model, "mlp_model_regression.json")
    destroy_mlp_model(my_lib, model)


def predict_mlp_model_classification_3_class(my_lib, model, inputs):
    inputs_float = [float(i) / 127.5 - 1.0 for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_mlp_model_classification.argtypes = [POINTER(c_void_p),
                                                        inputs_type,
                                                        c_int]
    my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    result_predict = my_lib.predict_mlp_model_classification(model, inputs_type(*inputs_float), len(inputs_float))

    return np.ctypeslib.as_array(result_predict, (3,))


def train_classification_stochastic_backprop_mlp_model_3_class(my_lib, model, dataset_inputs, dataset_expected_outputs,
                                                               alpha, iterations_count):
    dataset_inputs_float = [float(i) / 127.5 - 1.0 for i in dataset_inputs]
    dataset_inputs_flattened_type = len(dataset_inputs_float) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_classification_stochastic_backprop_mlp_model.argtypes = [POINTER(c_void_p),
                                                                          dataset_inputs_flattened_type,
                                                                          c_int,
                                                                          dataset_expected_outputs_type,
                                                                          c_int,
                                                                          c_float,
                                                                          c_int]
    my_lib.train_classification_stochastic_backprop_mlp_model.restype = None

    my_lib.train_classification_stochastic_backprop_mlp_model(model,
                                                              dataset_inputs_flattened_type(*dataset_inputs_float),
                                                              len(dataset_inputs_float),
                                                              dataset_expected_outputs_type(
                                                                  *dataset_expected_outputs_float),
                                                              len(dataset_expected_outputs_float),
                                                              alpha, iterations_count)


def test_classification_mlp_model_3_class(my_lib):
    # Init dataset
    dataset_flattened_inputs = [
        0, 0,
        0.5, 0.5,
        0, 1
    ]
    dataset_flattened_outputs = [
        1, -1, -1,
        -1, 1, -1,
        -1, -1, 1
    ]

    model = create_mlp_model(my_lib, [2, 3, 3])

    for _ in range(2):
        points = [[i / 10.0, j / 10.0] for i in range(15) for j in range(15)]

        predicted_values = [predict_mlp_model_classification_3_class(my_lib, model, p) for p in points]

        classes = [np.argmax(v) for v in predicted_values]

        colors = ['blue' if c == 0 else ('red' if c == 1 else 'green') for c in classes]

        plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors)
        plt.scatter(dataset_flattened_inputs[0], dataset_flattened_inputs[1], c="blue", s=200)
        plt.scatter(dataset_flattened_inputs[2], dataset_flattened_inputs[3], c="red", s=200)
        plt.scatter(dataset_flattened_inputs[4], dataset_flattened_inputs[5], c="green", s=200)
        plt.show()

        train_classification_stochastic_backprop_mlp_model_3_class(my_lib, model, dataset_flattened_inputs,
                                                                   dataset_flattened_outputs, float(0.01), 100000)

    destroy_mlp_model(my_lib, model)


def create_svm_model(my_lib, dataset_inputs, dataset_expected_outputs, alpha, max_iterations_count):
    dataset_inputs_double = [c_double(i) for i in dataset_inputs]
    dataset_inputs_type = len(dataset_inputs_double) * c_double
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float
    my_lib.create_svm_model.argtypes = [dataset_inputs_type,
                                        dataset_expected_outputs_type,
                                        c_int,
                                        c_int,
                                        c_double,
                                        c_int]
    my_lib.create_svm_model.restype = POINTER(c_void_p)
    return my_lib.create_svm_model(dataset_inputs_type(*dataset_inputs_double),
                                   dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                   int(len(dataset_inputs_double) / len(dataset_expected_outputs_float)),
                                   len(dataset_expected_outputs_float),
                                   alpha,
                                   max_iterations_count)


def save_svm_model(my_lib, model, path):
    my_lib.save_svm_model.argtypes = [POINTER(c_void_p), c_char_p]
    my_lib.save_svm_model.restype = None
    my_lib.save_svm_model(model, path.encode("utf-8"))


def load_svm_model(my_lib, path):
    my_lib.load_svm_model.argtypes = [c_char_p]
    my_lib.load_svm_model.restype = POINTER(c_void_p)
    return my_lib.load_svm_model(path.encode("utf-8"))


def predict_svm(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float
    my_lib.predict_svm.argtypes = [POINTER(c_void_p),
                                   inputs_type,
                                   c_int]
    my_lib.predict_svm.restype = c_float
    return my_lib.predict_svm(model,
                              inputs_type(*inputs_float),
                              len(inputs_float))


def destroy_svm_model(my_lib, model):
    my_lib.destroy_svm_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_svm_model.restype = None
    my_lib.destroy_svm_model(model)


def test_svm(my_lib):
    dataset_inputs = np.array([
        [1.0, 1],
        [2, 1],
        [2, 2],
        [4, 1],
        [4, 4]
    ])
    dataset_expected_outputs = np.array([
        1,
        1,
        -1,
        -1,
        -1
    ])

    model = create_svm_model(my_lib, dataset_inputs.flatten(), dataset_expected_outputs, 1.0, 10000)

    points_x1_blue = []
    points_x2_blue = []

    points_x1_red = []
    points_x2_red = []

    for i in range(0, 50):
        for j in range(0, 50):
            if predict_svm(my_lib, model, [1.0, i / 10, j / 10]) >= 0:
                points_x1_blue.append(i / 10)
                points_x2_blue.append(j / 10)
            else:
                points_x1_red.append(i / 10)
                points_x2_red.append(j / 10)

    plt.scatter(points_x1_blue, points_x2_blue, c='pink')
    plt.scatter(points_x1_red, points_x2_red, c='cyan')

    plt.scatter([p[0] for p in dataset_inputs[:2]], [p[1] for p in dataset_inputs[:2]], c='red', s=100)
    plt.scatter([p[0] for p in dataset_inputs[2:]], [p[1] for p in dataset_inputs[2:]], c='blue', s=100)

    plt.show()

    save_svm_model(my_lib, model, "svm_model.json")
    model2 = load_svm_model(my_lib, "svm_model.json")
    destroy_svm_model(my_lib, model)
    destroy_svm_model(my_lib, model2)

    # SVM x3

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in X])
    X = np.array(X[[not np.all(arr == [0, 0, 0]) for arr in Y]])
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y1 = np.array([1 if v[0] == 1 else -1 for v in Y])
    Y2 = np.array([1 if v[1] == 1 else -1 for v in Y])
    Y3 = np.array([1 if v[2] == 1 else -1 for v in Y])

    model = create_svm_model(my_lib, X.flatten(), Y1, 1.0, 10000)
    model2 = create_svm_model(my_lib, X.flatten(), Y2, 1.0, 10000)
    model3 = create_svm_model(my_lib, X.flatten(), Y3, 1.0, 10000)

    points = [[1.0, i / 50.0, j / 50.0] for i in range(-50, 51) for j in range(-50, 51)]

    predicted_values = [predict_svm(my_lib, model, p) for p in points]
    predicted_values2 = [predict_svm(my_lib, model2, p) for p in points]
    predicted_values3 = [predict_svm(my_lib, model3, p) for p in points]
    for i in range(len(predicted_values)):
        if predicted_values[i] >= 0:
            predicted_values[i] = 0
        if predicted_values2[i] >= 0:
            predicted_values[i] = 1
        if predicted_values3[i] >= 0:
            predicted_values[i] = 2
        if predicted_values[i] < 0 and predicted_values2[i] < 0 and predicted_values3[i] < 0:
            predicted_values[i] = 3

    colors = ['cyan' if c == 0 else ('pink' if c == 1 else ('orange' if c == 2 else 'yellow')) for c in
              predicted_values]

    plt.scatter([p[1] for p in points], [p[2] for p in points], c=colors)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.show()
    plt.clf()

    destroy_svm_model(my_lib, model)
    destroy_svm_model(my_lib, model2)
    destroy_svm_model(my_lib, model3)


def create_svm_kernel_trick_model(my_lib, dataset_inputs, dataset_expected_outputs, alpha, max_iterations_count):
    dataset_inputs_flattened = []
    for elt in dataset_inputs:
        dataset_inputs_flattened.append(elt[0])
        dataset_inputs_flattened.append(elt[1])
    dataset_inputs_flattened_type = len(dataset_inputs_flattened) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float
    my_lib.create_svm_kernel_trick_model.argtypes = [dataset_inputs_flattened_type,
                                                     dataset_expected_outputs_type,
                                                     c_int,
                                                     c_int,
                                                     c_double,
                                                     c_int]
    my_lib.create_svm_kernel_trick_model.restype = POINTER(c_void_p)
    return my_lib.create_svm_kernel_trick_model(dataset_inputs_flattened_type(*dataset_inputs_flattened),
                                                dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                                int(len(dataset_inputs_flattened) / len(
                                                    dataset_expected_outputs_float)),
                                                len(dataset_expected_outputs_float),
                                                alpha,
                                                max_iterations_count)


def save_svm_kernel_trick_model(my_lib, model, path):
    my_lib.save_svm_kernel_trick_model.argtypes = [POINTER(c_void_p), c_char_p]
    my_lib.save_svm_kernel_trick_model.restype = None
    my_lib.save_svm_kernel_trick_model(model, path.encode("utf-8"))


def load_svm_kernel_trick_model(my_lib, path):
    my_lib.load_svm_kernel_trick_model.argtypes = [c_char_p]
    my_lib.load_svm_kernel_trick_model.restype = POINTER(c_void_p)
    return my_lib.load_svm_kernel_trick_model(path.encode("utf-8"))


def predict_svm_kernel_trick(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float
    my_lib.predict_svm_kernel_trick.argtypes = [POINTER(c_void_p),
                                                inputs_type,
                                                c_int]
    my_lib.predict_svm_kernel_trick.restype = c_float
    return my_lib.predict_svm_kernel_trick(model, inputs_type(*inputs_float), len(inputs_float))


def destroy_svm_kernel_trick_model(my_lib, model):
    my_lib.destroy_svm_kernel_trick_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_svm_kernel_trick_model.restype = None
    my_lib.destroy_svm_kernel_trick_model(model)


def test_svm_kernel_trick(my_lib):
    X = np.array([[random.uniform(0.5, 4.5), random.uniform(0.5, 4.5)] for _ in range(20)])
    Y = [random.randint(0, 1) for _ in range(20)]
    Y = np.array([elt if elt == 1 else -1 for elt in Y])
    # print(X)
    # print(Y)

    model = create_svm_kernel_trick_model(my_lib, X, Y, 1.0, 10000)

    points_x1_blue = []
    points_x2_blue = []

    points_x1_red = []
    points_x2_red = []

    for i in range(0, 50):
        for j in range(0, 50):
            pred = predict_svm_kernel_trick(my_lib, model, [i / 10, j / 10])
            if pred >= 0:
                points_x1_blue.append(i / 10)
                points_x2_blue.append(j / 10)
            else:
                points_x1_red.append(i / 10)
                points_x2_red.append(j / 10)

    plt.scatter(points_x1_blue, points_x2_blue, c='pink')
    plt.scatter(points_x1_red, points_x2_red, c='cyan')

    plt.scatter([X[k][0] for k in range(len(X)) if Y[k] == 1], [X[k][1] for k in range(len(X)) if Y[k] == 1], c='red',
                s=100)
    plt.scatter([X[k][0] for k in range(len(X)) if Y[k] == -1], [X[k][1] for k in range(len(X)) if Y[k] == -1],
                c='blue', s=100)

    plt.show()

    save_svm_kernel_trick_model(my_lib, model, "svm_kernel_trick_model.json")
    model2 = load_svm_kernel_trick_model(my_lib, "svm_kernel_trick_model.json")
    destroy_svm_kernel_trick_model(my_lib, model)
    destroy_svm_kernel_trick_model(my_lib, model2)

    # Multi Cross
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])

    Y1 = np.array([1 if v[0] == 1 else -1 for v in Y])
    Y2 = np.array([1 if v[1] == 1 else -1 for v in Y])
    Y3 = np.array([1 if v[2] == 1 else -1 for v in Y])

    model = create_svm_kernel_trick_model(my_lib, X, Y1, 0.01, 2500000)
    model2 = create_svm_kernel_trick_model(my_lib, X, Y2, 0.01, 2500000)
    model3 = create_svm_kernel_trick_model(my_lib, X, Y3, 0.01, 2500000)

    points = [[i / 50.0, j / 50.0] for i in range(-50, 51) for j in range(-50, 51)]

    predicted_values = [predict_svm_kernel_trick(my_lib, model, p) for p in points]
    predicted_values2 = [predict_svm_kernel_trick(my_lib, model2, p) for p in points]
    predicted_values3 = [predict_svm_kernel_trick(my_lib, model3, p) for p in points]
    for i in range(len(predicted_values)):
        if predicted_values[i] >= 0:
            predicted_values[i] = 0
        if predicted_values2[i] >= 0:
            predicted_values[i] = 1
        if predicted_values3[i] >= 0:
            predicted_values[i] = 2
        if predicted_values[i] < 0 and predicted_values2[i] < 0 and predicted_values3[i] < 0:
            predicted_values[i] = 3

    colors = ['cyan' if c == 0 else ('pink' if c == 1 else ('orange' if c == 2 else 'yellow')) for c in
              predicted_values]

    plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.show()
    plt.clf()

    destroy_svm_kernel_trick_model(my_lib, model)
    destroy_svm_kernel_trick_model(my_lib, model2)
    destroy_svm_kernel_trick_model(my_lib, model3)


def create_rbf_k_center_model(my_lib, input_dim, cluster_num, gamma):
    my_lib.create_rbf_k_center_model.argtypes = [c_int,
                                                 c_int,
                                                 c_float]
    my_lib.create_rbf_k_center_model.restype = POINTER(c_void_p)

    return my_lib.create_rbf_k_center_model(input_dim, cluster_num, float(gamma))


def save_rbf_k_center_model(my_lib, model, path):
    my_lib.save_rbf_k_center_model.argtypes = [POINTER(c_void_p), c_char_p]
    my_lib.save_rbf_k_center_model.restype = None
    my_lib.save_rbf_k_center_model(model, path.encode("utf-8"))


def load_rbf_k_center_model(my_lib, path):
    my_lib.load_rbf_k_center_model.argtypes = [c_char_p]
    my_lib.load_rbf_k_center_model.restype = POINTER(c_void_p)
    return my_lib.load_rbf_k_center_model(path.encode("utf-8"))


def destroy_rbf_k_center_model(my_lib, model):
    my_lib.destroy_rbf_k_center_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_rbf_k_center_model.restype = None
    my_lib.destroy_rbf_k_center_model(model)


def train_regression_rbf_k_center_model(my_lib, model, X, Y):
    dataset_inputs_float = [float(i) for i in X]
    dataset_inputs_type = len(dataset_inputs_float) * c_float
    dataset_expected_outputs_float = [float(i) for i in Y]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_regression_rbf_k_center_model.argtypes = [POINTER(c_void_p),
                                                           dataset_inputs_type,
                                                           dataset_expected_outputs_type,
                                                           c_int,
                                                           c_int]
    my_lib.train_regression_rbf_k_center_model.restype = None
    my_lib.train_regression_rbf_k_center_model(model,
                                               dataset_inputs_type(*dataset_inputs_float),
                                               dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                               int(len(dataset_inputs_float) / len(dataset_expected_outputs_float)),
                                               len(dataset_expected_outputs_float))


def predict_rbf_k_center_model_regression(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float
    my_lib.predict_rbf_k_center_model_regression.argtypes = [POINTER(c_void_p),
                                                             inputs_type]
    my_lib.predict_rbf_k_center_model_regression.restype = c_float
    return my_lib.predict_rbf_k_center_model_regression(model, inputs_type(*inputs_float))


def test_regression_rbf_k_center_model(my_lib):
    # Init dataset
    X = [
        1.0,
        3.0,
        4.0,
    ]
    Y = [
        2.0,
        3.0,
        7.0
    ]

    model = create_rbf_k_center_model(my_lib, 1, 3, 0.01)

    train_regression_rbf_k_center_model(my_lib, model, X, Y)

    point_x = []
    point_y = []

    for i in range(-10, 11):
        point_x.append(float(i))
        point_y.append(predict_rbf_k_center_model_regression(my_lib, model, [i]))

    plt.plot(point_x, point_y)
    plt.scatter(X, Y, c="purple")
    plt.show()

    save_rbf_k_center_model(my_lib, model, "rbf_k_center_model.json")
    model2 = load_rbf_k_center_model(my_lib, "rbf_k_center_model.json")
    destroy_rbf_k_center_model(my_lib, model)
    destroy_rbf_k_center_model(my_lib, model2)

    X = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    Y = np.array([
        2,
        3,
        2.5
    ])

    model = create_rbf_k_center_model(my_lib, 2, 3, 0.1)

    train_regression_rbf_k_center_model(my_lib, model, X.flatten(), Y)

    points_x = []
    points_y = []
    points_z = []

    for i in range(10, 31):
        for j in range(10, 31):
            points_x.append(float(i / 10))
            points_y.append(float(j / 10))
            points_z.append(float(predict_rbf_k_center_model_regression(my_lib, model, [i / 10, j / 10])))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points_x, points_y, points_z)
    ax.scatter(X[:, 0], X[:, 1], Y, c="orange", s=100)
    plt.show()
    plt.clf()

    destroy_rbf_k_center_model(my_lib, model)

    X = [
        1.5,
        1.4,
        1.5,
        1.4,
        2.5,
        2.4,
        2.5,
        2.4
    ]
    Y = [
        1.5,
        1.5,
        1.4,
        1.4,
        2.5,
        2.5,
        2.4,
        2.4
    ]

    model = create_rbf_k_center_model(my_lib, 1, 2, 0.1)

    train_regression_rbf_k_center_model(my_lib, model, X, Y)

    point_x = []
    point_y = []

    for i in range(1, 5):
        point_x.append(float(i))
        point_y.append(predict_rbf_k_center_model_regression(my_lib, model, [i]))

    plt.plot(point_x, point_y)
    plt.scatter(X, Y, c="purple")
    plt.show()

    destroy_rbf_k_center_model(my_lib, model)

    X = np.array([
        [1.5, 1.5],
        [1.4, 1.5],
        [1.5, 1.4],
        [1.4, 1.4],
        [2.5, 2.5],
        [2.4, 2.5],
        [2.5, 2.4],
        [2.4, 2.4]
    ])
    Y = np.array([
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2
    ])

    model = create_rbf_k_center_model(my_lib, 2, 2, 0.1)

    train_regression_rbf_k_center_model(my_lib, model, X.flatten(), Y)

    points_x = []
    points_y = []
    points_z = []

    for i in range(10, 31):
        for j in range(10, 31):
            points_x.append(float(i / 10))
            points_y.append(float(j / 10))
            points_z.append(float(predict_rbf_k_center_model_regression(my_lib, model, [i / 10, j / 10])))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points_x, points_y, points_z)
    ax.scatter(X[:, 0], X[:, 1], Y, c="orange", s=100)
    plt.show()
    plt.clf()

    destroy_rbf_k_center_model(my_lib, model)


def train_rosenblatt_rbf_k_center_model(my_lib, model, X, Y, iterations, alpha):
    dataset_inputs_flattened = [float(i) for i in X]
    dataset_inputs_flattened_type = len(dataset_inputs_flattened) * c_float
    dataset_expected_outputs_float = [float(i) for i in Y]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_rosenblatt_rbf_k_center_model.argtypes = [POINTER(c_void_p),
                                                           dataset_inputs_flattened_type,
                                                           dataset_expected_outputs_type,
                                                           c_int,
                                                           c_float,
                                                           c_int,
                                                           c_int]
    my_lib.train_rosenblatt_rbf_k_center_model.restype = None
    my_lib.train_rosenblatt_rbf_k_center_model(model,
                                               dataset_inputs_flattened_type(*dataset_inputs_flattened),
                                               dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                               int(iterations),
                                               float(alpha),
                                               int(len(dataset_inputs_flattened) / len(dataset_expected_outputs_float)),
                                               len(dataset_expected_outputs_float))


def predict_rbf_k_center_model_classification(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float
    my_lib.predict_rbf_k_center_model_classification.argtypes = [POINTER(c_void_p),
                                                                 inputs_type]
    my_lib.predict_rbf_k_center_model_classification.restype = c_float
    return my_lib.predict_rbf_k_center_model_classification(model, inputs_type(*inputs_float))


def test_classification_rbf_k_center_model(my_lib):
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3]
    ])
    Y = np.array([
        1,
        -1,
        -1
    ])

    model = create_rbf_k_center_model(my_lib, 2, 3, 0.1)

    train_rosenblatt_rbf_k_center_model(my_lib, model, X.flatten(), Y, 10000, 0.01)

    points_x1_blue = []
    points_x2_blue = []

    points_x1_red = []
    points_x2_red = []
    for i in range(5, 36):
        for j in range(5, 36):
            if predict_rbf_k_center_model_classification(my_lib, model, [i / 10, j / 10]) == 1.0:
                points_x1_blue.append(i / 10)
                points_x2_blue.append(j / 10)
            else:
                points_x1_red.append(i / 10)
                points_x2_red.append(j / 10)

    plt.scatter(points_x1_blue, points_x2_blue, c='blue')
    plt.scatter(points_x1_red, points_x2_red, c='red')

    plt.scatter(X[0, 0], X[0, 1], color='blue', s=100)
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red', s=100)
    plt.show()
    plt.clf()

    destroy_rbf_k_center_model(my_lib, model)

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    model = create_rbf_k_center_model(my_lib, 2, 2, 0.1)

    train_rosenblatt_rbf_k_center_model(my_lib, model, X.flatten(), Y, 10000, 0.01)

    points_x1_blue = []
    points_x2_blue = []

    points_x1_red = []
    points_x2_red = []

    for i in range(9, 31):
        for j in range(9, 31):
            if predict_rbf_k_center_model_classification(my_lib, model, [i / 10, j / 10]) >= 0:
                points_x1_blue.append(i / 10)
                points_x2_blue.append(j / 10)
            else:
                points_x1_red.append(i / 10)
                points_x2_red.append(j / 10)

    plt.scatter(points_x1_blue, points_x2_blue, c='blue')
    plt.scatter(points_x1_red, points_x2_red, c='red')

    plt.scatter(X[0:50, 0], X[0:50, 1], color='blue', s=100)
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red', s=100)
    plt.show()
    plt.clf()

    destroy_rbf_k_center_model(my_lib, model)

    # XOR

    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])

    model = create_rbf_k_center_model(my_lib, 2, 4, 0.1)

    train_rosenblatt_rbf_k_center_model(my_lib, model, X.flatten(), Y, 100000, 0.01)

    points_x1_blue = []
    points_x2_blue = []

    points_x1_red = []
    points_x2_red = []

    for i in range(-5, 16):
        for j in range(-5, 16):
            if predict_rbf_k_center_model_classification(my_lib, model, [i / 10, j / 10]) >= 0:
                points_x1_blue.append(i / 10.0)
                points_x2_blue.append(j / 10.0)
            else:
                points_x1_red.append(i / 10.0)
                points_x2_red.append(j / 10.0)

    plt.scatter(points_x1_blue, points_x2_blue, c='blue')
    plt.scatter(points_x1_red, points_x2_red, c='red')

    plt.scatter(X[0:2, 0], X[0:2, 1], color='blue', s=100)
    plt.scatter(X[2:4, 0], X[2:4, 1], color='red', s=100)
    plt.show()
    plt.clf()

    destroy_rbf_k_center_model(my_lib, model)


def IA_RBF(my_lib):
    dataset_train_img, dataset_train_label = load_dataset("../../DATASET_80px_TRAIN")
    dataset_test_img, dataset_test_label = load_dataset("../../DATASET_80px_TEST")

    dataset_expected_outputs = np.array(
        [[1, -1, -1] if p == "Dionea" else [-1, 1, -1] if p == "Sarracenia" else [-1, -1, 1] for p in
         dataset_train_label])
    dataset_expected_outputs_test = np.array(
        [[1, -1, -1] if p == "Dionea" else [-1, 1, -1] if p == "Sarracenia" else [-1, -1, 1] for p in
         dataset_test_label])
    dataset_expected_outputs_sarracenia = np.array([1 if v[1] == 1 else -1 for v in dataset_expected_outputs])
    dataset_expected_outputs_drosera = np.array([1 if v[2] == 1 else -1 for v in dataset_expected_outputs])
    dataset_expected_outputs_dionaea = np.array([1 if v[0] == 1 else -1 for v in dataset_expected_outputs])

    model_sarracenia = create_rbf_k_center_model(my_lib, 9, 3, 0.1)
    model_drosera = create_rbf_k_center_model(my_lib, 9, 3, 0.1)
    model_dionaea = create_rbf_k_center_model(my_lib, 9, 3, 0.1)
    # model_sarracenia = load_rbf_k_center_model(my_lib, "rbf_sarracenia.json")
    # model_drosera = load_rbf_k_center_model(my_lib, "rbf_drosera.json")
    # model_dionaea = load_rbf_k_center_model(my_lib, "rbf_dionaea.json")

    point_x = []
    point_y = []
    point_mse_y = []
    point_y_test = []
    point_mse_y_test = []

    for n in range(1501):
        train_rosenblatt_rbf_k_center_model(my_lib, model_sarracenia, np.array(dataset_train_img).flatten(),
                                            dataset_expected_outputs_sarracenia.flatten(), len(dataset_train_img),
                                            float(0.005))
        train_rosenblatt_rbf_k_center_model(my_lib, model_drosera, np.array(dataset_train_img).flatten(),
                                            dataset_expected_outputs_drosera.flatten(), len(dataset_train_img),
                                            float(0.005))
        train_rosenblatt_rbf_k_center_model(my_lib, model_dionaea, np.array(dataset_train_img).flatten(),
                                            dataset_expected_outputs_dionaea.flatten(), len(dataset_train_img),
                                            float(0.005))

        # Dataset de Train
        true_true_sarracenia = 0
        false_false_sarracenia = 0

        true_true_drosera = 0
        false_false_drosera = 0

        true_true_dionea = 0
        false_false_dionea = 0

        predicted_values_sarracenia = [
            predict_rbf_k_center_model_classification(my_lib, model_sarracenia, np.array(i).flatten()) for i in
            dataset_train_img]
        predicted_values_drosera = [
            predict_rbf_k_center_model_classification(my_lib, model_drosera, np.array(i).flatten()) for i in
            dataset_train_img]
        predicted_values_dionaea = [
            predict_rbf_k_center_model_classification(my_lib, model_dionaea, np.array(i).flatten()) for i in
            dataset_train_img]

        predicted_values = predicted_values_sarracenia
        for i in range(len(predicted_values_sarracenia)):
            if predicted_values_sarracenia[i] >= 0:
                predicted_values[i] = 1
            if predicted_values_drosera[i] >= 0:
                predicted_values[i] = 2
            if predicted_values_dionaea[i] >= 0:
                predicted_values[i] = 0
            if predicted_values_sarracenia[i] < 0 and predicted_values_drosera[i] < 0 and predicted_values_dionaea[
                i] < 0:
                predicted_values[i] = 3

        colors = ['Dionea' if c == 0 else ('Sarracenia' if c == 1 else ('Drosera' if c == 2 else "None")) for c in
                  predicted_values]

        sum_se = 0
        for i in range(len(predicted_values)):
            if np.array(dataset_expected_outputs[i]).argmax() == 0:
                sum_se += (predicted_values_sarracenia[i] - (-1)) ** 2
                sum_se += (predicted_values_drosera[i] - (-1)) ** 2
                sum_se += (predicted_values_dionaea[i] - 1) ** 2
            elif np.array(dataset_expected_outputs[i]).argmax() == 1:
                sum_se += (predicted_values_sarracenia[i] - 1) ** 2
                sum_se += (predicted_values_drosera[i] - (-1)) ** 2
                sum_se += (predicted_values_dionaea[i] - (-1)) ** 2
            else:
                sum_se += (predicted_values_sarracenia[i] - (-1)) ** 2
                sum_se += (predicted_values_drosera[i] - 1) ** 2
                sum_se += (predicted_values_dionaea[i] - (-1)) ** 2
        mse = sum_se / (len(predicted_values) * 3)

        point_mse_y.append(mse)

        for i in range(len(colors)):
            if ((colors[i] == "Dionea") and (dataset_train_label[i] == "Dionea")):
                true_true_dionea += 1
            if ((colors[i] != "Dionea") and (dataset_train_label[i] == "Dionea")):
                false_false_dionea += 1

            if ((colors[i] == "Drosera") and (dataset_train_label[i] == "Drosera")):
                true_true_drosera += 1
            if ((colors[i] != "Drosera") and (dataset_train_label[i] == "Drosera")):
                false_false_drosera += 1

            if ((colors[i] == "Sarracenia") and (dataset_train_label[i] == "Sarracenia")):
                true_true_sarracenia += 1
            if ((colors[i] != "Sarracenia") and (dataset_train_label[i] == "Sarracenia")):
                false_false_sarracenia += 1

        point_x.append(float(n))
        point_y.append(((true_true_dionea / (true_true_dionea + false_false_dionea)) + (
                true_true_drosera / (true_true_drosera + false_false_drosera)) + (
                                true_true_sarracenia / (true_true_sarracenia + false_false_sarracenia))) / 3)

        # Dataset de Test
        true_true_sarracenia_test = 0
        false_false_sarracenia_test = 0

        true_true_drosera_test = 0
        false_false_drosera_test = 0

        true_true_dionea_test = 0
        false_false_dionea_test = 0

        predicted_values_sarracenia = [
            predict_rbf_k_center_model_classification(my_lib, model_sarracenia, np.array(i).flatten()) for i in
            dataset_test_img]
        predicted_values_drosera = [
            predict_rbf_k_center_model_classification(my_lib, model_drosera, np.array(i).flatten()) for i in
            dataset_test_img]
        predicted_values_dionaea = [
            predict_rbf_k_center_model_classification(my_lib, model_dionaea, np.array(i).flatten()) for i in
            dataset_test_img]

        predicted_values = predicted_values_sarracenia
        for i in range(len(predicted_values_sarracenia)):
            if predicted_values_sarracenia[i] >= 0:
                predicted_values[i] = 1
            if predicted_values_drosera[i] >= 0:
                predicted_values[i] = 2
            if predicted_values_dionaea[i] >= 0:
                predicted_values[i] = 0
            if predicted_values_sarracenia[i] < 0 and predicted_values_drosera[i] < 0 and predicted_values_dionaea[
                i] < 0:
                predicted_values[i] = 3

        colors = ['Dionea' if c == 0 else ('Sarracenia' if c == 1 else ('Drosera' if c == 2 else "None")) for c in
                  predicted_values]

        sum_se = 0
        for i in range(len(predicted_values)):
            if np.array(dataset_expected_outputs_test[i]).argmax() == 0:
                sum_se += (predicted_values_sarracenia[i] - (-1)) ** 2
                sum_se += (predicted_values_drosera[i] - (-1)) ** 2
                sum_se += (predicted_values_dionaea[i] - 1) ** 2
            elif np.array(dataset_expected_outputs_test[i]).argmax() == 1:
                sum_se += (predicted_values_sarracenia[i] - 1) ** 2
                sum_se += (predicted_values_drosera[i] - (-1)) ** 2
                sum_se += (predicted_values_dionaea[i] - (-1)) ** 2
            else:
                sum_se += (predicted_values_sarracenia[i] - (-1)) ** 2
                sum_se += (predicted_values_drosera[i] - 1) ** 2
                sum_se += (predicted_values_dionaea[i] - (-1)) ** 2
        mse = sum_se / (len(predicted_values) * 3)

        point_mse_y_test.append(mse)

        for i in range(len(colors)):
            if ((colors[i] == "Dionea") and (dataset_test_label[i] == "Dionea")):
                true_true_dionea_test += 1
            if ((colors[i] != "Dionea") and (dataset_test_label[i] == "Dionea")):
                false_false_dionea_test += 1

            if ((colors[i] == "Drosera") and (dataset_test_label[i] == "Drosera")):
                true_true_drosera_test += 1
            if ((colors[i] != "Drosera") and (dataset_test_label[i] == "Drosera")):
                false_false_drosera_test += 1

            if ((colors[i] == "Sarracenia") and (dataset_test_label[i] == "Sarracenia")):
                true_true_sarracenia_test += 1
            if ((colors[i] != "Sarracenia") and (dataset_test_label[i] == "Sarracenia")):
                false_false_sarracenia_test += 1

        point_y_test.append(((true_true_dionea_test / (true_true_dionea_test + false_false_dionea_test)) + (
                true_true_drosera_test / (true_true_drosera_test + false_false_drosera_test)) + (
                                     true_true_sarracenia_test / (
                                     true_true_sarracenia_test + false_false_sarracenia_test))) / 3)
        print(n)
        if n % 100 == 0 and n != 0:
            plt.clf()
            plt.plot(point_x, point_y)
            plt.plot(point_x, point_y_test, c="orange")
            plt.savefig("rbf_accuracy_" + str(n) + ".png")
            plt.clf()
            plt.plot(point_x, point_mse_y)
            plt.plot(point_x, point_mse_y_test, c="orange")
            plt.savefig("rbf_loss_" + str(n) + ".png")
            save_rbf_k_center_model(my_lib, model_sarracenia, "rbf_sarracenia_" + str(n) + ".json")
            save_rbf_k_center_model(my_lib, model_drosera, "rbf_drosera_" + str(n) + ".json")
            save_rbf_k_center_model(my_lib, model_dionaea, "rbf_dionaea_" + str(n) + ".json")

    plt.clf()
    plt.plot(point_x, point_y)
    plt.show()
    plt.clf()
    plt.plot(point_x, point_mse_y)
    plt.show()

    save_rbf_k_center_model(my_lib, model_sarracenia, "rbf_sarracenia.json")
    save_rbf_k_center_model(my_lib, model_drosera, "rbf_drosera.json")
    save_rbf_k_center_model(my_lib, model_dionaea, "rbf_dionaea.json")
    destroy_rbf_k_center_model(my_lib, model_sarracenia)
    destroy_rbf_k_center_model(my_lib, model_drosera)
    destroy_rbf_k_center_model(my_lib, model_dionaea)


class MySKLearnMLPRawWrapper:
    def __init__(self, lib, npl: [int], classification: bool = True):
        self.lib = lib
        self.model = create_mlp_model(lib, npl)
        self.classification = classification

    def predict(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X)
        return np.array(predict_mlp_model_classification_3_class(self.lib, self.model))


def IA(my_lib):
    # resize_dataset(os.path.realpath('C:/Users/cedri/OneDrive/Bureau/DATASET_PROJET/TRAIN_120'), 120)
    # resize_dataset(os.path.realpath('C:/Users/cedri/OneDrive/Bureau/DATASET_PROJET/TEST_120'), 120)

    dataset_train_img, dataset_train_label = load_dataset("../../TRAIN_120")
    dataset_test_img, dataset_test_label = load_dataset("../../TEST_120")
    size_first_layer = sum([numpy.prod(img.shape) for img in dataset_train_img]) / len(dataset_train_img)

    print(size_first_layer)
    dataset_expected_outputs = np.array(
        [[1, -1, -1] if p == "Dionea" else [-1, 1, -1] if p == "Sarracenia" else [-1, -1, 1] for p in
         dataset_train_label])
    dataset_expected_outputs_test = np.array(
        [[1, -1, -1] if p == "Dionea" else [-1, 1, -1] if p == "Sarracenia" else [-1, -1, 1] for p in
         dataset_test_label])

    model = create_mlp_model(my_lib, [size_first_layer, 64, 3])
    # model = load_mlp_model(my_lib, "dataset_mlp_model.json")

    point_x = []
    point_y = []
    point_mse_x = []
    point_mse_y = []
    point_x_test = []
    point_y_test = []
    point_mse_x_test = []
    point_mse_y_test = []

    for n in range(1001):
        print(n)
        train_classification_stochastic_backprop_mlp_model_3_class(my_lib, model, np.array(dataset_train_img).flatten(),
                                                                   dataset_expected_outputs.flatten(), float(0.00075),
                                                                   len(dataset_train_img))
        # Dataset de Train
        true_true_sarracenia = 0
        true_false_sarracenia = 0
        false_true_sarracenia = 0
        false_false_sarracenia = 0

        true_true_drosera = 0
        true_false_drosera = 0
        false_true_drosera = 0
        false_false_drosera = 0

        true_true_dionea = 0
        true_false_dionea = 0
        false_true_dionea = 0
        false_false_dionea = 0

        predicted_values_train = [predict_mlp_model_classification_3_class(my_lib, model, np.array(i).flatten()) for
                                  i in dataset_train_img]
        classes_train = [np.argmax(v) for v in predicted_values_train]
        classes_train_lbl = [np.argmax(v) for v in dataset_expected_outputs]
        colors_train = ['Dionea' if c == 0 else ('Sarracenia' if c == 1 else 'Drosera') for c in classes_train]

        sum_se = 0
        for i in range(len(predicted_values_train)):
            for j in range(3):
                diff = predicted_values_train[i][j] - dataset_expected_outputs[i][j]
                diff_square = diff ** 2
                sum_se += diff_square
        mse = sum_se / (len(predicted_values_train) * 3)

        point_mse_x.append(float(n))
        point_mse_y.append(mse)

        for i in range(len(colors_train)):
            if ((colors_train[i] == "Dionea") and (dataset_train_label[i] == "Dionea")):
                true_true_dionea += 1
            if ((colors_train[i] == "Dionea") and (dataset_train_label[i] != "Dionea")):
                true_false_dionea += 1
            if ((colors_train[i] != "Dionea") and (dataset_train_label[i] == "Dionea")):
                false_false_dionea += 1
            if ((colors_train[i] != "Dionea") and (dataset_train_label[i] != "Dionea")):
                false_true_dionea += 1

            if ((colors_train[i] == "Drosera") and (dataset_train_label[i] == "Drosera")):
                true_true_drosera += 1
            if ((colors_train[i] == "Drosera") and (dataset_train_label[i] != "Drosera")):
                true_false_drosera += 1
            if ((colors_train[i] != "Drosera") and (dataset_train_label[i] == "Drosera")):
                false_false_drosera += 1
            if ((colors_train[i] != "Drosera") and (dataset_train_label[i] != "Drosera")):
                false_true_drosera += 1

            if ((colors_train[i] == "Sarracenia") and (dataset_train_label[i] == "Sarracenia")):
                true_true_sarracenia += 1
            if ((colors_train[i] == "Sarracenia") and (dataset_train_label[i] != "Sarracenia")):
                true_false_sarracenia += 1
            if ((colors_train[i] != "Sarracenia") and (dataset_train_label[i] == "Sarracenia")):
                false_false_sarracenia += 1
            if ((colors_train[i] != "Sarracenia") and (dataset_train_label[i] != "Sarracenia")):
                false_true_sarracenia += 1

        point_x.append(float(n))
        point_y.append(((true_true_dionea / (true_true_dionea + false_false_dionea)) + (
                true_true_drosera / (true_true_drosera + false_false_drosera)) + (
                                true_true_sarracenia / (true_true_sarracenia + false_false_sarracenia))) / 3)

        # Dataset de Test
        true_true_sarracenia_test = 0
        true_false_sarracenia_test = 0
        false_true_sarracenia_test = 0
        false_false_sarracenia_test = 0

        true_true_drosera_test = 0
        true_false_drosera_test = 0
        false_true_drosera_test = 0
        false_false_drosera_test = 0

        true_true_dionea_test = 0
        true_false_dionea_test = 0
        false_true_dionea_test = 0
        false_false_dionea_test = 0

        predicted_values_test = [predict_mlp_model_classification_3_class(my_lib, model, np.array(i).flatten()) for
                                 i in dataset_test_img]
        classes_test = [np.argmax(v) for v in predicted_values_test]
        colors_test = ['Dionea' if c == 0 else ('Sarracenia' if c == 1 else 'Drosera') for c in classes_test]
        classes_test_lbl = [np.argmax(v) for v in dataset_expected_outputs_test]

        sum_se = 0
        for i in range(len(predicted_values_test)):
            for j in range(3):
                diff = predicted_values_test[i][j] - dataset_expected_outputs_test[i][j]
                diff_square = diff ** 2
                sum_se += diff_square
        mse = sum_se / (len(predicted_values_test) * 3)

        point_mse_x_test.append(float(n))
        point_mse_y_test.append(mse)

        for i in range(len(colors_test)):
            if ((colors_test[i] == "Dionea") and (dataset_test_label[i] == "Dionea")):
                true_true_dionea_test += 1
            if ((colors_test[i] == "Dionea") and (dataset_test_label[i] != "Dionea")):
                true_false_dionea_test += 1
            if ((colors_test[i] != "Dionea") and (dataset_test_label[i] == "Dionea")):
                false_false_dionea_test += 1
            if ((colors_test[i] != "Dionea") and (dataset_test_label[i] != "Dionea")):
                false_true_dionea_test += 1

            if ((colors_test[i] == "Drosera") and (dataset_test_label[i] == "Drosera")):
                true_true_drosera_test += 1
            if ((colors_test[i] == "Drosera") and (dataset_test_label[i] != "Drosera")):
                true_false_drosera_test += 1
            if ((colors_test[i] != "Drosera") and (dataset_test_label[i] == "Drosera")):
                false_false_drosera_test += 1
            if ((colors_test[i] != "Drosera") and (dataset_test_label[i] != "Drosera")):
                false_true_drosera_test += 1

            if ((colors_test[i] == "Sarracenia") and (dataset_test_label[i] == "Sarracenia")):
                true_true_sarracenia_test += 1
            if ((colors_test[i] == "Sarracenia") and (dataset_test_label[i] != "Sarracenia")):
                true_false_sarracenia_test += 1
            if ((colors_test[i] != "Sarracenia") and (dataset_test_label[i] == "Sarracenia")):
                false_false_sarracenia_test += 1
            if ((colors_test[i] != "Sarracenia") and (dataset_test_label[i] != "Sarracenia")):
                false_true_sarracenia_test += 1

        point_x_test.append(float(n))
        point_y_test.append(((true_true_dionea_test / (true_true_dionea_test + false_false_dionea_test)) + (
                true_true_drosera_test / (true_true_drosera_test + false_false_drosera_test)) + (
                                     true_true_sarracenia_test / (
                                         true_true_sarracenia_test + false_false_sarracenia_test))) / 3)

        if (n % 50 == 0) and n != 0:
            plt.clf()
            plt.plot(point_x, point_y)
            plt.plot(point_x_test, point_y_test, c="orange")
            plt.savefig("accuracy_" + str(n) + ".png")
            plt.clf()
            plt.plot(point_mse_x, point_mse_y)
            plt.plot(point_mse_x_test, point_mse_y_test, c="orange")
            plt.savefig("loss_" + str(n) + ".png")
            save_mlp_model(my_lib, model, "dataset_mlp_model_" + str(n) + ".json")
            plt.clf()

            try:
                y_actu_train = pd.Series(classes_train, name='Predicted')
                y_pred_train = pd.Series(classes_train_lbl, name='Actual')
                df_confusion_train = confusion_matrix(y_actu_train, y_pred_train)
                df_cm_train = pd.DataFrame(df_confusion_train, index=[i for i in ["Dionea", "Drosera", "Sarracenia"]],
                                           columns=[i for i in ["Dionea", "Drosera", "Sarracenia"]])
                plt.figure(figsize=(10, 7))
                sn.heatmap(df_cm_train, annot=True, fmt='d')
                plt.show()
                #plt.savefig(str(n) + '.png')
                plt.clf()

                y_actu = pd.Series(classes_test, name='Predicted')
                y_pred = pd.Series(classes_test_lbl, name='Actual')
                df_confusion = confusion_matrix(y_actu, y_pred)
                df_cm = pd.DataFrame(df_confusion, index=[i for i in ["Dionea", "Drosera", "Sarracenia"]],
                                     columns=[i for i in ["Dionea", "Drosera", "Sarracenia"]])
                plt.figure(figsize=(10, 7))
                sn.heatmap(df_cm, annot=True, fmt='d')
                plt.show()
                #plt.savefig(str(n) + 'test.png')
            except:
                pass

    plt.clf()
    plt.plot(point_x, point_y)
    plt.show()
    plt.clf()
    plt.plot(point_mse_x, point_mse_y)
    plt.show()
    plt.clf()


    """cpt_train = 0
    cpt_test = 0
    print('--------------')
    for l in range(len(classes_predict_train)):
        if classes_predict_train[l] == classes_train[l]:
            cpt = cpt_train + 1
    print(cpt_train)
    print("acc TRAIN: " + str(cpt_train / len(classes_predict_train)))
    print('--------------')
    for l in range(len(classes_predict_test)):
        if classes_predict_test[l] == classes_test[l]:
            cpt = cpt_test + 1
    print(cpt_test)
    print("acc TRAIN: " + str(cpt_test / len(classes_predict_test)))
    print('--------------')"""

    save_mlp_model(my_lib, model, "dataset_mlp_model_test_32gray.json")
    destroy_mlp_model(my_lib, model)


if __name__ == "__main__":
    # Load lib
    my_lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    IA(my_lib)
    # my_lib = cdll.LoadLibrary(path_to_shared_library_release)

    # test_classification_linear_model(my_lib)
    # test_regression_linear_model(my_lib)
    # test_classification_mlp_model(my_lib)
    # test_regression_mlp_model(my_lib)
    # test_classification_mlp_model_3_class(my_lib)
    # test_svm(my_lib)
    # test_svm_kernel_trick(my_lib)
    # test_regression_rbf_k_center_model(my_lib)
    # test_classification_rbf_k_center_model(my_lib)

    # dataset_img, dataset_label = load_dataset()
    # dataset_img = np.array(dataset_img[:5])
    # # print(dataset_img[0])
    # # print(len(dataset_img[0].flatten()))
    # print(dataset_label[:5])
    # dataset_expected_outputs = np.array(
    #     [[1, 0, 0] if p == "Dionea" else [0, 1, 0] if p == "Sarracenia" else [0, 0, 1] for p in dataset_label[:5]])
    # print(dataset_expected_outputs)
    #
    # model = create_mlp_model(my_lib, [480000, 3])
    #
    # train_classification_stochastic_backprop_mlp_model_3_class(my_lib, model, dataset_img.flatten(),
    #                                                            dataset_expected_outputs.flatten(), float(0.03),
    #                                                            250000)
    #
    # predicted_values = [predict_mlp_model_classification_3_class(my_lib, model, np.array(dataset_img[i]).flatten()) for
    #                     i in range(3)]
    # print(predicted_values)
    #
    # destroy_mlp_model(my_lib, model)
