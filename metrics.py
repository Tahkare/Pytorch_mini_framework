"""
Created on Tue Feb 18 2020

@author: Alban Petit
"""
import numpy as np

class Metric:
    def __init__(self, higher_is_better=True):
        self.higher_is_better = higher_is_better
        self.reset()   
    def step(self, in_data, out_data, labels):
        pass
    def score(self):
        pass
    def reset(self):
        pass
    
class Accuracy_Multi_Class(Metric):
    
    def step(self, in_data, out_data, labels):
        for i,j in zip(out_data,labels):
            self.total += 1
            try:
                best = np.argmax(i.detach().cpu().numpy())
            except:
                best = np.argmax(i.detach().numpy())
            if best == j:
                self.correct += 1
        
    def score(self):
        return self.correct/self.total
    
    def reset(self):
        self.total = 0
        self.correct = 0
        
class Accuracy_Binary(Metric):
    def step(self, in_data, out_data, labels):
        for i,j in zip(out_data, labels):
            self.total += 1
            if (i>0) == (j==1):
                self.correct += 1
        
    def score(self):
        return self.correct/self.total
    
    def reset(self):
        self.total = 0
        self.correct = 0

class BAC_6_classes_2_out(Metric):
    def reset(self):
        self.totals = [0 for i in range(12)]
        self.corrects = [0 for i in range(12)]
    
    def step(self, in_data, out_data, labels):
        for l in range(6):
            for i,j in zip(out_data[l], labels):
                self.totals[2*l+int(j[l])] += 1
                if (i[1] > i[0]) == (j[l]==1):
                    self.corrects[2*l+int(j[l])] += 1
        
    def score(self):
        s = 0
        t = 0
        for i in range(12):
            if self.totals[i] > 0:
                t += self.corrects[i] / self.totals[i]
                s += 1
        return t/s
    
class BAC_6_classes_1_out(Metric):
    def reset(self):
        self.totals = [0 for i in range(12)]
        self.corrects = [0 for i in range(12)]
    
    def step(self, in_data, out_data, labels):
        for l in range(6):
            for i,j in zip(out_data, labels):
                self.totals[2*l+int(j[l])] += 1
                if (i[l] > 0) == (j[l]==1):
                    self.corrects[2*l+int(j[l])] += 1
        
    def score(self):
        s = 0
        t = 0
        for i in range(12):
            if self.totals[i] > 0:
                t += self.corrects[i] / self.totals[i]
                s += 1
        return t/s