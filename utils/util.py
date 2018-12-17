# coding:utf-8
import os
from collections import namedtuple

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def dict_to_namedtuple(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
