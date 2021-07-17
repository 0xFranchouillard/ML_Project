from ctypes import *


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
