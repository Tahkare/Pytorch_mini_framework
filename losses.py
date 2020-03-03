"""
Created on Tue Feb 18 2020

@author: Alban Petit
"""
import torch as th

class Loss:
    def __init__(self):
        pass
    def compute(self, in_data, out_data, labels):
        pass

class CrossEntropy_Multi_Class(Loss):
    def __init__(self):
        self.loss = th.nn.CrossEntropyLoss()
        
    def compute(self, in_data, out_data, labels):
        return self.loss(out_data, labels.squeeze())

class Loss_1LSTM_6Linear(Loss):
    def __init__(self):
        self.l0 = th.nn.CrossEntropyLoss(weight=th.Tensor([1.,10.]))
        self.l1 = th.nn.CrossEntropyLoss(weight=th.Tensor([1.,99.]))
        self.l2 = th.nn.CrossEntropyLoss(weight=th.Tensor([1.,19.]))
        self.l3 = th.nn.CrossEntropyLoss(weight=th.Tensor([1.,299.]))
        self.l4 = th.nn.CrossEntropyLoss(weight=th.Tensor([1.,20.]))
        self.l5 = th.nn.CrossEntropyLoss(weight=th.Tensor([1.,124.]))
        
    def compute(self, in_data, out_data, labels):
        return self.l0(out_data[0], labels[:,0].squeeze())+self.l1(out_data[1], labels[:,1].squeeze()) + self.l2(out_data[2], labels[:,2].squeeze())+self.l3(out_data[3], labels[:,3].squeeze()) + self.l4(out_data[4], labels[:,4].squeeze())+self.l5(out_data[5], labels[:,5].squeeze())

class Loss_1LSTM_1Linear6(Loss):
    def __init__(self):
        self.loss = th.nn.BCEWithLogitsLoss()
        
    def compute(self, in_data, out_data, labels):
        return self.loss(out_data, labels.float())