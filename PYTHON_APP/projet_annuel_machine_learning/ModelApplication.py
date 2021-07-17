import sys
import numpy as np

from process_dataset import *
from LinearModel import *
from MLP import *
from SVM import *
from SVM_Kernel import *
from RBF import *


def main():
    if len(sys.argv) != 2:
        print("1 argument attendu")
        return

    if os.path.isfile("../image/photoTest.jpg"):
        image = load_image("../image/photoTest.jpg")
    else:
        image = load_image("../image/photoTest.png")

    my_lib = cdll.LoadLibrary("lib_projet_annuel_machine_learning.dll")
    if sys.argv[1] == "Linear":
        model_sarracenia = load_linear_model(my_lib, "models/linear_model_sarracenia.json")
        model_drosera = load_linear_model(my_lib, "models/linear_model_drosera.json")
        model_dionaea = load_linear_model(my_lib, "models/linear_model_dionaea.json")
        value_predict_sarracenia = predict_linear_model_classification(my_lib, model_sarracenia, image)
        value_predict_drosera = predict_linear_model_classification(my_lib, model_drosera, image)
        value_predict_dionaea = predict_linear_model_classification(my_lib, model_dionaea, image)
        destroy_linear_model(my_lib, model_sarracenia)
        destroy_linear_model(my_lib, model_drosera)
        destroy_linear_model(my_lib, model_dionaea)
        if value_predict_sarracenia > 0.0:
            return "Sarracenia"
        elif value_predict_drosera > 0.0:
            return "Drosera"
        elif value_predict_dionaea > 0.0:
            return "Dionaea"
        else:
            return "None"
    elif sys.argv[1] == "MLP":
        model = load_mlp_model(my_lib, "models/mlp.json")
        value_predict = predict_mlp_model_classification_3_class(my_lib, model, image)
        destroy_mlp_model(my_lib, model)
        if np.argmax(value_predict) == 0:
            return "Dionaea"
        elif np.argmax(value_predict) == 1:
            return "Sarracenia"
        else:
            return "Drosera"
    elif sys.argv[1] == "SVM":
        print("SVM")
    elif sys.argv[1] == "SVM Kernel":
        print("SVM Kernel")
    elif sys.argv[1] == "RBF":
        print("RBF")
