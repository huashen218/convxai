import os
import torch
import pandas as pd
import numpy as np
import json
import h5py
from transformers import BertTokenizer, BertModel


def h5_load(filename, data_list, dtype=None, verbose=False):
    with h5py.File(filename, 'r') as infile:
        data = []
        for data_name in data_list:
            if dtype is not None:
                temp = np.empty(infile[data_name].shape, dtype=dtype)
            else:
                temp = np.empty(infile[data_name].shape, dtype=infile[data_name].dtype)
            infile[data_name].read_direct(temp)
            data.append(temp)
          
        if verbose:
            print("\n".join(
                "{} = {} [{}]".format(data_name, str(real_data.shape), str(real_data.dtype))
                for data_name, real_data in zip(data_list, data)
            ))
            print()
        return data

