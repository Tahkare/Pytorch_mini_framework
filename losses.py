"""
Created on Tue Feb 18 2020

@author: Alban Petit
"""

import torch as th

class CrossEntropy_Multi_Class:
    
    def __init__(self):
        self.loss = th.nn.CrossEntropyLoss()
        
    def compute(self, out_data, labels):
        return self.loss(out_data, labels.squeeze())
