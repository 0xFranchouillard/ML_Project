from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

path_to_shared_library = "../../RUST_LIBRARY/lib_projet_annuel_machine_learning/target/debug/lib_projet_annuel_machine_learning.dll"


def make_linear_model(size):
    my_lib.create_linear_model.argtypes = [c_int]
    my_lib.create_linear_model.restype = POINTER(c_float)

    p_model = my_lib.create_linear_model(size)
    model = np.ctypeslib.as_array(p_model, (size + 1,))

    model_size = len(model)
    return p_model, model_size


def destroy_model(p_model, model_size):
    my_lib.destroy_linear_model.argtypes = [POINTER(c_float), c_int]
    my_lib.destroy_linear_model.restype = None
    my_lib.destroy_linear_model(p_model, model_size)


def test_classification():
    # Init dataset
    dataset_inputs = [
        [1, 4],
        [1, -4],
        [4, 4],
    ]

    dataset_expected_outputs = [
        float(1),
        float(1),
        float(-1)
    ]

    p_model, model_size = make_linear_model(2)

    # Test Train Classification
    arr = np.array([0.0, 0.0], float)
    arr_size = 2
    arr_type = c_float * arr_size

    my_lib.predict_linear_model_classification.argtypes = [POINTER(c_float), arr_type, c_int]
    my_lib.predict_linear_model_classification.restype = c_float

    for _ in range(50):
        # Init points
        points_x1_blue = []
        points_x2_blue = []

        points_x1_red = []
        points_x2_red = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                arr[0] = float(i)
                arr[1] = float(j)
                if my_lib.predict_linear_model_classification(p_model, arr_type(*arr), model_size) == 1.0:
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
        dataset_inputs_for_train = []
        for elt in dataset_inputs:
            dataset_inputs_for_train.append(float(elt[0]))
            dataset_inputs_for_train.append(float(elt[1]))

        arr_type_dataset_inputs = c_float * len(dataset_inputs_for_train)
        arr_type_dataset_outputs = c_float * len(dataset_expected_outputs)

        my_lib.train_rosenblatt_linear_model.argtypes = [POINTER(c_float), arr_type_dataset_inputs,
                                                         arr_type_dataset_outputs, c_int, c_float, c_int, c_int]
        my_lib.train_rosenblatt_linear_model.restype = None

        my_lib.train_rosenblatt_linear_model(p_model, arr_type_dataset_inputs(*dataset_inputs_for_train),
                                             arr_type_dataset_outputs(*dataset_expected_outputs), 20, float(0.1),
                                             model_size, len(dataset_inputs_for_train))

    destroy_model(p_model, model_size)

def test_regression():
    # Train Regression
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

    p_model, model_size = make_linear_model(1)

    point_x = []
    point_y = []

    arr_type_dataset_inputs = c_float * len(dataset_inputs)
    arr_type_dataset_outputs = c_float * len(dataset_expected_outputs)

    arr = np.array([0.0], float)
    arr_size = 1
    arr_type = c_float * arr_size

    my_lib.predict_linear_model_regression.argtypes = [POINTER(c_float), arr_type, c_int]
    my_lib.predict_linear_model_regression.restype = c_float

    for i in range(-10, 11):
        point_x.append(float(i))
        point_y.append(my_lib.predict_linear_model_regression(p_model, arr_type(*arr), model_size))

    plt.scatter(point_x, point_y)
    plt.scatter(dataset_inputs, dataset_expected_outputs, c="purple")
    plt.show()

    my_lib.train_regression_linear_model.argtypes = [POINTER(c_float), arr_type_dataset_inputs,
                                                     arr_type_dataset_outputs, c_int, c_int]
    my_lib.train_regression_linear_model.restype = None

    my_lib.train_regression_linear_model(p_model, arr_type_dataset_inputs(*dataset_inputs),
                                         arr_type_dataset_outputs(*dataset_expected_outputs), model_size,
                                         len(dataset_inputs))

    point_x = []
    point_y = []

    for i in range(-10, 11):
        point_x.append(float(i))
        arr[0] = float(i)
        point_y.append(my_lib.predict_linear_model_regression(p_model, arr_type(*arr), model_size))

    plt.scatter(point_x, point_y)
    plt.scatter(dataset_inputs, dataset_expected_outputs, c="purple")
    plt.show()

    destroy_model(p_model, model_size)

if __name__ == "__main__":
    # Load lib
    my_lib = cdll.LoadLibrary(path_to_shared_library)
    #test_classification()
    test_regression()
