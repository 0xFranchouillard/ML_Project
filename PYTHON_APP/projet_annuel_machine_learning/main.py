from ctypes import *

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from projet_annuel_machine_learning.process_dataset import load_dataset, resize_dataset

PATH_TO_SHARED_LIBRARY = "C:/Users/Cyrille Champion/Desktop/ML_Project/RUST_LIBRARY/lib_projet_annuel_machine_learning/target/release/" \
                         "lib_projet_annuel_machine_learning.dll"


def create_linear_model(my_lib, size):
    my_lib.create_linear_model.argtypes = [c_int]
    my_lib.create_linear_model.restype = POINTER(c_float)

    p_model = my_lib.create_linear_model(size)
    model = np.ctypeslib.as_array(p_model, (size + 1,))

    model_size = len(model)
    return p_model, model_size


def destroy_linear_model(my_lib, model, model_size):
    my_lib.destroy_linear_model.argtypes = [POINTER(c_float), c_int]
    my_lib.destroy_linear_model.restype = None
    my_lib.destroy_linear_model(model, model_size)


def predict_linear_model_classification(my_lib, model, model_size, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_linear_model_classification.argtypes = [POINTER(c_float),
                                                           inputs_type,
                                                           c_int]
    my_lib.predict_linear_model_classification.restype = c_float

    return my_lib.predict_linear_model_classification(model, inputs_type(*inputs_float), model_size)


def train_rosenblatt_linear_model(my_lib, model, model_size, dataset_inputs, dataset_expected_outputs, iteration_count,
                                  alpha):
    dataset_inputs_flattened_for_train = []
    for elt in dataset_inputs:
        dataset_inputs_flattened_for_train.append(float(elt[0]))
        dataset_inputs_flattened_for_train.append(float(elt[1]))
    dataset_inputs_flattened_for_train_type = len(dataset_inputs_flattened_for_train) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_rosenblatt_linear_model.argtypes = [POINTER(c_float),
                                                     dataset_inputs_flattened_for_train_type,
                                                     dataset_expected_outputs_type,
                                                     c_int,
                                                     c_float,
                                                     c_int,
                                                     c_int]
    my_lib.train_rosenblatt_linear_model.restype = None

    my_lib.train_rosenblatt_linear_model(model,
                                         dataset_inputs_flattened_for_train_type(*dataset_inputs_flattened_for_train),
                                         dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                         iteration_count, float(alpha), model_size,
                                         len(dataset_inputs_flattened_for_train))


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
    model, model_size = create_linear_model(my_lib, 2)

    for _ in range(5):
        points_x1_blue = []
        points_x2_blue = []

        points_x1_red = []
        points_x2_red = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                if predict_linear_model_classification(my_lib, model, model_size, [i, j]) == 1.0:
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

        train_rosenblatt_linear_model(my_lib, model, model_size, dataset_inputs, dataset_expected_outputs, 20, 0.1)

    destroy_linear_model(my_lib, model, model_size)


def predict_linear_model_regression(my_lib, model, model_size, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_linear_model_regression.argtypes = [POINTER(c_float),
                                                       inputs_type,
                                                       c_int]
    my_lib.predict_linear_model_regression.restype = c_float

    return my_lib.predict_linear_model_regression(model, inputs_type(*inputs_float), model_size)


def train_regression_linear_model(my_lib, model, model_size, dataset_inputs, dataset_expected_outputs):
    dataset_inputs_float = [float(i) for i in dataset_inputs]
    dataset_inputs_type = len(dataset_inputs_float) * c_float
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_regression_linear_model.argtypes = [POINTER(c_float),
                                                     dataset_inputs_type,
                                                     dataset_expected_outputs_type,
                                                     c_int,
                                                     c_int]
    my_lib.train_regression_linear_model.restype = None

    my_lib.train_regression_linear_model(model, dataset_inputs_type(*dataset_inputs_float),
                                         dataset_expected_outputs_type(*dataset_expected_outputs_float), model_size,
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
    model, model_size = create_linear_model(my_lib, 1)

    for _ in range(2):
        point_x = []
        point_y = []

        for i in range(-10, 11):
            point_x.append(float(i))
            point_y.append(predict_linear_model_regression(my_lib, model, model_size, [i]))

        plt.scatter(point_x, point_y)
        plt.scatter(dataset_inputs, dataset_expected_outputs, c="purple")
        plt.show()

        train_regression_linear_model(my_lib, model, model_size, dataset_inputs, dataset_expected_outputs)

    destroy_linear_model(my_lib, model, model_size)


def create_mlp_model(my_lib, struct_model):
    struct_model_int = [int(i) for i in struct_model]
    struct_model_type = len(struct_model_int) * c_int

    my_lib.create_mlp_model.argtypes = [struct_model_type,
                                        c_int]
    my_lib.create_mlp_model.restype = POINTER(c_void_p)

    return my_lib.create_mlp_model(struct_model_type(*struct_model_int), len(struct_model_int))


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

    destroy_mlp_model(my_lib, model)


def predict_mlp_model_classification_3_class(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_mlp_model_classification.argtypes = [POINTER(c_void_p),
                                                        inputs_type,
                                                        c_int]
    my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    result_predict = my_lib.predict_mlp_model_classification(model, inputs_type(*inputs_float), len(inputs_float))

    return np.ctypeslib.as_array(result_predict, (3,))


def train_classification_stochastic_backprop_mlp_model_3_class(my_lib, model, dataset_inputs, dataset_expected_outputs,
                                                               alpha, iterations_count):
    dataset_inputs_float = [float(i) for i in dataset_inputs]
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


def create_svm_model(my_lib, dataset_inputs, dataset_expected_outputs):
    dataset_inputs_flattened = []
    for elt in dataset_inputs:
        dataset_inputs_flattened.append(elt[0])
        dataset_inputs_flattened.append(elt[1])
    dataset_inputs_flattened_type = len(dataset_inputs_flattened) * c_double
    dataset_expected_outputs_float = [float(i) for i in dataset_expected_outputs]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float
    my_lib.create_svm_model.argtypes = [dataset_inputs_flattened_type,
                                        dataset_expected_outputs_type,
                                        c_int,
                                        c_int]
    my_lib.create_svm_model.restype = POINTER(c_float)
    return my_lib.create_svm_model(dataset_inputs_flattened_type(*dataset_inputs_flattened),
                                   dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                   int(len(dataset_inputs_flattened) / len(dataset_expected_outputs_float)),
                                   len(dataset_expected_outputs_float))


def predict_svm(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float
    my_lib.predict_svm.argtypes = [POINTER(c_float),
                                   inputs_type,
                                   c_int]
    my_lib.predict_svm.restype = c_float
    return my_lib.predict_svm(model, inputs_type(*inputs_float), len(inputs_float))


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

    model = create_svm_model(my_lib, dataset_inputs, dataset_expected_outputs)

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

    # SVM x3

    # X = np.random.random((500, 2)) * 2.0 - 1.0
    # Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
    #               [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
    #               [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
    #               [0, 0, 0] for p in X])
    # X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    # Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    # Y1 = np.array([1 if v[0] == 1 else -1 for v in Y])
    # Y2 = np.array([1 if v[1] == 1 else -1 for v in Y])
    # Y3 = np.array([1 if v[2] == 1 else -1 for v in Y])
    #
    # plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
    #             np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
    #             color='blue')
    # plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
    #             np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
    #             color='red')
    # plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
    #             np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
    #             color='green')
    # plt.show()
    # plt.clf()

    # BigMatrix = []
    # for i in range(len(X)):
    #     BigMatrix.append([])
    #     for j in range(len(X)):
    #         BigMatrix[i].append(Y1[i] * Y1[j] * (np.matmul(X[i].T, X[j])))
    # BigMatrix = np.array(BigMatrix)
    # print("BigMatrix:", BigMatrix)
    #
    # # Define problem data
    # P = sparse.csc_matrix(BigMatrix)
    # q = np.ones(len(X)) * -1
    # A = []
    # for i in range(len(X) + 1):
    #     A.append([])
    #     for j in range(len(X)):
    #         if i == 0:
    #             A[i].append(Y1[j])
    #         elif i - 1 == j:
    #             A[i].append(1.0)
    #         else:
    #             A[i].append(0.0)
    # A = sparse.csc_matrix(A)
    # l = np.zeros(len(X) + 1)
    # u = np.zeros(len(X) + 1)
    # for i in range(1, len(X) + 1):
    #     u[i] = 10000000000.0
    #
    # # Create an OSQP object
    # prob = osqp.OSQP()
    #
    # # Setup workspace and change alpha parameter
    # prob.setup(P, q, A, l, u, alpha=0.1)
    #
    # # Solve problem
    # res = prob.solve()
    # print(res.x)
    #
    # model = create_svm_model(my_lib, X, Y1)
    # model2 = create_svm_model(my_lib, X, Y2)
    # model3 = create_svm_model(my_lib, X, Y3)
    #
    # points = [[i / 50.0, j / 50.0] for i in range(-50, 51) for j in range(-50, 51)]
    #
    # predicted_values = [predict_svm(my_lib, model, p) for p in points]
    # predicted_values2 = [predict_svm(my_lib, model2, p) for p in points]
    # predicted_values3 = [predict_svm(my_lib, model3, p) for p in points]
    # for i in range(len(predicted_values)):
    #     if predicted_values[i] >= 0:
    #         predicted_values[i] = 0
    #     if predicted_values2[i] >= 0:
    #         predicted_values[i] = 1
    #     if predicted_values3[i] >= 0:
    #         predicted_values[i] = 2
    #     if predicted_values[i] < 0 and predicted_values2[i] < 0 and predicted_values3[i] < 0:
    #         predicted_values[i] = 3
    #
    # colors = ['cyan' if c == 0 else ('pink' if c == 1 else ('orange' if c == 2 else 'yellow')) for c in
    #           predicted_values]
    #
    # plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors)
    #
    # plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
    #             np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
    #             color='blue')
    # plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
    #             np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
    #             color='red')
    # plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
    #             np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
    #             color='green')
    # plt.show()
    # plt.clf()


def create_svm_kernel_trick_model(my_lib, dataset_inputs, dataset_expected_outputs):
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
                                                     c_int]
    my_lib.create_svm_kernel_trick_model.restype = POINTER(c_void_p)
    return my_lib.create_svm_kernel_trick_model(dataset_inputs_flattened_type(*dataset_inputs_flattened),
                                                dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                                int(len(dataset_inputs_flattened) / len(
                                                    dataset_expected_outputs_float)),
                                                len(dataset_expected_outputs_float))


def predict_svm_kernel_trick(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float
    my_lib.predict_svm_kernel_trick.argtypes = [POINTER(c_void_p),
                                                inputs_type,
                                                c_int]
    my_lib.predict_svm_kernel_trick.restype = c_float
    return my_lib.predict_svm_kernel_trick(model, inputs_type(*inputs_float), len(inputs_float))


def test_svm_kernel_trick(my_lib):
    X = np.array([[random.uniform(0.5, 4.5), random.uniform(0.5, 4.5)] for _ in range(20)])
    Y = [random.randint(0, 1) for _ in range(20)]
    Y = np.array([elt if elt == 1 else -1 for elt in Y])
    # print(X)
    # print(Y)

    model = create_svm_kernel_trick_model(my_lib, X, Y);

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

    # Multi Cross
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])

    Y1 = np.array([1 if v[0] == 1 else -1 for v in Y])
    Y2 = np.array([1 if v[1] == 1 else -1 for v in Y])
    Y3 = np.array([1 if v[2] == 1 else -1 for v in Y])

    model = create_svm_kernel_trick_model(my_lib, X, Y1)
    model2 = create_svm_kernel_trick_model(my_lib, X, Y2)
    model3 = create_svm_kernel_trick_model(my_lib, X, Y3)

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


def create_rbf_k_center_model(my_lib, input_dim, cluster_num, gamma):
    my_lib.create_rbf_k_center_model.argtypes = [c_int,
                                                 c_int,
                                                 c_float]
    my_lib.create_rbf_k_center_model.restype = POINTER(c_void_p)

    return my_lib.create_rbf_k_center_model(input_dim, cluster_num, float(gamma))


def destroy_rbf_k_center_model(my_lib, model):
    my_lib.destroy_rbf_k_center_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_rbf_k_center_model.restype = None
    my_lib.destroy_rbf_k_center_model(model)


def train_regression_rbf_k_center_model(my_lib, model, X, Y):
    # dataset_inputs_flattened = []
    # for elt in X:
    #     dataset_inputs_flattened.append(elt[0])
    #     dataset_inputs_flattened.append(elt[1])
    dataset_inputs_flattened = [float(i) for i in X]
    dataset_inputs_flattened_type = len(dataset_inputs_flattened) * c_float
    dataset_expected_outputs_float = [float(i) for i in Y]
    dataset_expected_outputs_type = len(dataset_expected_outputs_float) * c_float

    my_lib.train_regression_rbf_k_center_model.argtypes = [POINTER(c_void_p),
                                                           dataset_inputs_flattened_type,
                                                           dataset_expected_outputs_type,
                                                           c_int,
                                                           c_int]
    my_lib.train_regression_rbf_k_center_model.restype = None
    my_lib.train_regression_rbf_k_center_model(model,
                                               dataset_inputs_flattened_type(*dataset_inputs_flattened),
                                               dataset_expected_outputs_type(*dataset_expected_outputs_float),
                                               int(len(dataset_inputs_flattened) / len(dataset_expected_outputs_float)),
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

    destroy_rbf_k_center_model(my_lib, model)

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


if __name__ == "__main__":
    # Load lib
    my_lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)

    # test_classification_linear_model(my_lib)
    # test_regression_linear_model(my_lib)
    # test_classification_mlp_model(my_lib)
    # test_regression_mlp_model(my_lib)
    # test_classification_mlp_model_3_class(my_lib)
    # test_svm(my_lib)
    # test_svm_kernel_trick(my_lib)
    # test_regression_rbf_k_center_model(my_lib)
    # test_classification_rbf_k_center_model(my_lib)
    dataset_img, dataset_label = load_dataset()
    print(dataset_label[:5])
    dataset_img = np.array(dataset_img[:5])

    model = create_mlp_model(my_lib, [30000, 3])

    dataset_expected_outputs = np.array([[1, 0, 0] if p == 'Dionea' else [0, 1, 0] if p == 'Sarracenia' else [0, 0, 1] for p in dataset_label[:5]])
    train_classification_stochastic_backprop_mlp_model_3_class(my_lib, model, dataset_img.flatten(),
                                                               dataset_expected_outputs.flatten(), float(0.03), 25)

    predicted = [predict_mlp_model_classification_3_class(my_lib, model, np.array(dataset_img[i].flatten() for i in range(3)))]

    print(predicted)