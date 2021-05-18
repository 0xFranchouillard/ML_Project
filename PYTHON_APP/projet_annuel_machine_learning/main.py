from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

path_to_shared_library = "D:/CLion/PA/lib_projet_annuel_machine_learning/target/debug/lib_projet_annuel_machine_learning.dll"

if __name__ == "__main__":
    # Load lib
    my_lib = cdll.LoadLibrary(path_to_shared_library)

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

    # Create Model
    my_lib.create_linear_model.argtypes = [c_int]
    my_lib.create_linear_model.restype = POINTER(c_float)

    model_size = 2
    p_model = my_lib.create_linear_model(model_size)
    model = np.ctypeslib.as_array(p_model, (model_size+1,))

    print(model)
    model_size = len(model)

    # Init points
    points_x1_blue = []
    points_x2_blue = []

    points_x1_red = []
    points_x2_red = []

    # Test Classification
    arr = np.array([0.0,0.0],float)
    arr_size = 2
    arr_type = c_float * arr_size

    my_lib.predict_linear_model_classification.argtypes = [POINTER(c_float), arr_type, c_int]
    my_lib.predict_linear_model_classification.restype = c_float

    for _ in range(10):
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

    # Destroy Model
    my_lib.destroy_linear_model.argtypes = [POINTER(c_float), c_int]
    my_lib.destroy_linear_model.restype = None

    my_lib.destroy_linear_model(p_model, model_size)