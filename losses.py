"""
Created on Tue Feb 18 2020

@author: Alban Petit
"""

import torch as th

def CrossEntropy_Multi_Class(out_data, labels):
    l = th.nn.CrossEntropyLoss()
    return l(out_data, labels.squeeze())
