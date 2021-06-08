from ctypes import *

import numpy as np
import matplotlib.pyplot as plt

PATH_TO_SHARED_LIBRARY = "D:/CLion/PA/lib_projet_annuel_machine_learning/target/debug" \
                         "/lib_projet_annuel_machine_learning.dll "


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
                                                              dataset_expected_outputs_type(*dataset_expected_outputs_float),
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
                                                          dataset_expected_outputs_type(*dataset_expected_outputs_float),
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
                                                              dataset_expected_outputs_type(*dataset_expected_outputs_float),
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


if __name__ == "__main__":
    # Load lib
    my_lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)

    # test_classification_linear_model(my_lib)
    # test_regression_linear_model(my_lib)
    # test_classification_mlp_model(my_lib)
    # test_regression_mlp_model(my_lib)
    test_classification_mlp_model_3_class(my_lib)
