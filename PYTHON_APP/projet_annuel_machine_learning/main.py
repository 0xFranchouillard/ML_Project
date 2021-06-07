from ctypes import *

import numpy as np
import matplotlib.pyplot as plt

path_to_shared_library = "D:/CLion/PA/lib_projet_annuel_machine_learning/target/debug/lib_projet_annuel_machine_learning.dll"


def make_linear_model(size):
    my_lib.create_linear_model.argtypes = [c_int]
    my_lib.create_linear_model.restype = POINTER(c_float)

    p_model = my_lib.create_linear_model(size)
    model = np.ctypeslib.as_array(p_model, (size + 1,))

    model_size = len(model)
    return p_model, model_size


def destroy_model(model, model_size):
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

def train_rosenblatt_linear_model(my_lib, model, model_size, dataset_inputs, dataset_expected_outputs, iteration_count, alpha):
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

    my_lib.train_rosenblatt_linear_model(model, dataset_inputs_flattened_for_train_type(*dataset_inputs_flattened_for_train), dataset_expected_outputs_type(*dataset_expected_outputs_float), iteration_count, float(alpha), model_size, len(dataset_inputs_flattened_for_train))

def test_classification():
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
    model, model_size = make_linear_model(2)

    for _ in range(5):
        points_x1_blue = []
        points_x2_blue = []

        points_x1_red = []
        points_x2_red = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                if predict_linear_model_classification(my_lib, model, model_size, [i,j]) == 1.0:
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

    destroy_model(model, model_size)

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

    my_lib.train_regression_linear_model(model, dataset_inputs_type(*dataset_inputs_float), dataset_expected_outputs_type(*dataset_expected_outputs_float), model_size, len(dataset_inputs_float))

def test_regression():
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
    model, model_size = make_linear_model(1)

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

    destroy_model(model, model_size)

if __name__ == "__main__":
    # Load lib
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    # test_classification()
    # test_regression()

    # arr = np.array([2, 3, 1], int)
    # arr_type = c_int * len(arr)
    #
    # my_lib.create_mlp_model.argtypes = [arr_type, c_int]
    # my_lib.create_mlp_model.restype = POINTER(c_void_p)
    # model = my_lib.create_mlp_model(arr_type(*arr), len(arr))
    #
    # dataset_inputs = [
    #     [float(0), float(0)],
    #     [float(1), float(1)],
    #     [float(1), float(0)],
    #     [float(0), float(1)]
    # ]
    #
    # # dataset_flattened_inputs = [float(0), float(0), float(1), float(1), float(1), float(0), float(0), float(1)]
    # # dataset_flattened_inputs_type = c_float * len(dataset_flattened_inputs)
    # dataset_flattened_outputs = [float(-1), float(-1), float(1), float(1)]
    # dataset_flattened_outputs_type = c_float * len(dataset_flattened_outputs)
    #
    # arr = np.array([0.0, 0.0], float)
    # arr_type = c_float * len(arr)
    #
    # my_lib.predict_mlp_model_classification.argtypes = [POINTER(c_void_p), arr_type, c_int]
    # my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    #
    # for _ in range(2):
    #     points_x1_blue = []
    #     points_x2_blue = []
    #
    #     points_x1_red = []
    #     points_x2_red = []
    #
    #     for i in range(-10, 11):
    #         for j in range(-10, 11):
    #             arr[0] = float(i)
    #             arr[1] = float(j)
    #             result_predict = my_lib.predict_mlp_model_classification(model, arr_type(*arr), len(arr))
    #             result_predict = np.ctypeslib.as_array(result_predict, (1,))
    #             print(result_predict)
    #             if result_predict >= 0:
    #                 points_x1_blue.append(i/5.0)
    #                 points_x2_blue.append(j/5.0)
    #             else:
    #                 points_x1_red.append(i/5.0)
    #                 points_x2_red.append(j/5.0)
    #     plt.scatter(points_x1_blue, points_x2_blue, c='blue')
    #     plt.scatter(points_x1_red, points_x2_red, c='red')
    #
    #     plt.scatter([p[0] for p in dataset_inputs[:2]], [p[1] for p in dataset_inputs[:2]], c='red', s=100)
    #     plt.scatter([p[0] for p in dataset_inputs[2:]], [p[1] for p in dataset_inputs[2:]], c='blue', s=100)
    #
    #     plt.show()
    #
    #     dataset_inputs_for_train = []
    #     for elt in dataset_inputs:
    #         dataset_inputs_for_train.append(elt[0])
    #         dataset_inputs_for_train.append(elt[1])
    #     dataset_inputs_for_train_type = c_float * len(dataset_inputs_for_train)
    #
    #     my_lib.train_classification_stochastic_backprop_mlp_model.argtypes = [POINTER(c_void_p),
    #                                                                           dataset_inputs_for_train_type, c_int,
    #                                                                           dataset_flattened_outputs_type, c_int]
    #     my_lib.train_classification_stochastic_backprop_mlp_model.restype = None
    #     my_lib.train_classification_stochastic_backprop_mlp_model(model,
    #                                                               dataset_inputs_for_train_type(*dataset_inputs_for_train),
    #                                                               len(dataset_inputs_for_train),
    #                                                               dataset_flattened_outputs_type(*dataset_flattened_outputs),
    #                                                               len(dataset_flattened_outputs))

    # my_lib.destroy_mlp_model.argtypes = [POINTER(c_void_p)]
    # my_lib.destroy_mlp_model.restype = None
    # my_lib.destroy_mlp_model(model)
