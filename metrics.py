"""
Created on Tue Feb 18 2020

@author: Alban Petit
"""
import numpy as np

class Metric:
    
    def __init__(self):
        self.reset()
        
    def step(self, out_data, labels):
        pass
    def score(self):
        pass
    def reset(self):
        pass
    
class Accuracy_Multi_Class(Metric):
    
    def step(self, out, labels):
        for i,j in zip(out,labels):
            self.total += 1
            best = np.argmax(i.detach().numpy())
            if best == j:
                self.correct += 1
        
    def score(self):
        return self.correct/self.total
    
    def reset(self):
        self.total = 0
        self.correct = 0
    