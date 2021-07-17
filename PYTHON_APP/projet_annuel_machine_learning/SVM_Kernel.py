from ctypes import *


def load_svm_kernel_trick_model(my_lib, path):
    my_lib.load_svm_kernel_trick_model.argtypes = [c_char_p]
    my_lib.load_svm_kernel_trick_model.restype = POINTER(c_void_p)
    return my_lib.load_svm_kernel_trick_model(path.encode("utf-8"))


def destroy_svm_kernel_trick_model(my_lib, model):
    my_lib.destroy_svm_kernel_trick_model.argtypes = [POINTER(c_void_p)]
    my_lib.destroy_svm_kernel_trick_model.restype = None
    my_lib.destroy_svm_kernel_trick_model(model)


def predict_svm_kernel_trick(my_lib, model, inputs):
    inputs_float = [float(i) for i in inputs]
    inputs_type = len(inputs_float) * c_float
    my_lib.predict_svm_kernel_trick.argtypes = [POINTER(c_void_p),
                                                inputs_type,
                                                c_int]
    my_lib.predict_svm_kernel_trick.restype = c_float
    return my_lib.predict_svm_kernel_trick(model,
                                           inputs_type(*inputs_float),
                                           len(inputs_float))
