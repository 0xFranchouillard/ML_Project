from ctypes import *

import numpy as np


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


def predict_mlp_model_classification_3_class(my_lib, model, inputs):
    inputs_float = [float(i) / 127.5 - 1.0 for i in inputs]
    inputs_type = len(inputs_float) * c_float

    my_lib.predict_mlp_model_classification.argtypes = [POINTER(c_void_p),
                                                        inputs_type,
                                                        c_int]
    my_lib.predict_mlp_model_classification.restype = POINTER(c_float)
    result_predict = my_lib.predict_mlp_model_classification(model, inputs_type(*inputs_float), len(inputs_float))

    return np.ctypeslib.as_array(result_predict, (3,))
