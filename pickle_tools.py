#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:35:05 2023

@author: nurmelaj
"""

import pickle
import os

def save_obj(obj, path, filename):
    '''
    Saves an object in a pickle format.

    Parameters
    ----------
    obj : object
        Object to be saved.
    path : str
        Relative or absolute path for saving.
    filename : str
        Name for the file to be saved in as pkl-file.

    Returns
    -------
    None.
    '''
    if os.path.exists(path + filename + '.pkl'):
        os.remove(path + filename + '.pkl')
    with open(path + filename + '.pkl', 'wb') as file:
        pickle.dump(obj, file)

def load_obj(path, filename):
    '''
    Loads a pickle formatted object.

    Parameters
    ----------
    path : str
        Relative or absolute path of the file.
    filename : str
        Name of the pickle file to be loaded.

    Returns
    -------
    object
        Loaded pkl-file.
    '''
    with open(path + filename + '.pkl', 'rb') as file:
        return pickle.load(file)