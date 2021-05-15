import ctypes
from ctypes import *

PATH_RUST_LIBRARY = '../RUST_LIBRARY/target/debug/RUST_LIBRARY.dll'

if __name__ == '__main__':
    my_lib = cdll.LoadLibrary(PATH_RUST_LIBRARY)

    print(my_lib.toto())
