from ctypes import *


def load_svm_model(my_lib, path):
    my_lib.load_svm_model.argtypes = [c_char_p]
    my_lib.load_svm_model.restype = POINTER(c_void_p)
    return my_lib.load_svm_model(path.encode("utf-8"))


def destroy_svm_model(my_lib, model):
    my_lib.destroy_svm_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_svm_model.restype = None
    my_lib.destroy_svm_model(model)


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
