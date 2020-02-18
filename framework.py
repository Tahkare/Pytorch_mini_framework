""" 
Created on Tue Feb 18 2020

@author : Alban Petit
"""

import torch as th
import torch.utils.data as data

class Dataset:
    
    def __merge__(self, datas, labels, data_type='float', label_type='long', batch_size=64, shuffle=True):
        tensor_type = {'int' : th.IntTensor, 'float' : th.FloatTensor, 'long' : th.LongTensor}
        return data.DataLoader([(tensor_type[data_type](i), tensor_type[label_type](j)) for i,j in zip(datas, labels)],
                               batch_size=batch_size, shuffle=shuffle)
    
    def __init__(self, name="default", train_set=None, train_data=None, train_labels=None, dev_set=None, dev_data=None, dev_labels=None,
                 test_set=None, test_data=None, test_labels=None, data_type='float', label_type='long', batch_size=64):
        self.name = name
        if train_set == None:
            self.train_set = self.__merge__(train_data, train_labels, data_type, label_type, batch_size)
        else:
            self.train_set = train_set
        if dev_set == None:
            self.dev_set = self.__merge__(dev_data, dev_labels, data_type, label_type, batch_size)
        else:
            self.dev_set = dev_set
        if test_set == None:
            self.test_set = self.__merge__(test_data, test_labels, data_type, label_type, batch_size, shuffle=False)
        else:
            self.test_set = test_set
        if self.train_set != None:
            self.__reset_train__()
        if self.dev_set != None:
            self.__reset_dev__()
        if self.test_set != None:
            self.__reset_test__()   
    
    def __next__(self, iterator):
        try:
            res = iterator.next()
        except:
            res = None
        return res
    
    def __next_train__(self):
        return self.__next__(self.train_iter)
        
    def __next_dev__(self):
        return self.__next__(self.dev_iter)
        
    def __next_test__(self):
        return self.__next__(self.test_iter)
    
    def __reset_train__(self):
        self.train_iter = iter(self.train_set)
        
    def __reset_dev__(self):
        self.dev_iter = iter(self.dev_set)
        
    def __reset_test__(self):
        self.test_iter = iter(self.test_set)
        
class Model:
    
    def __init__(self, name="default", model=None, loss=None, optimizer=None, dataset=None, metric=None):
        self.name = name
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.dataset = dataset
        self.metric = metric
        
    def train(self, epochs=10, evaluate=True, verbose=1):
        best_score = None
        self.dataset.__reset_train__()
        if evaluate:
            scores = []
        for e in range(epochs):
            self.model.train()
            loss = 0
            total_batches = len(self.dataset.train_iter)
            batch_size = None
            total_elements = 0
            for b in range(total_batches):
                in_data, labels = self.dataset.__next_train__()
                batch_size = in_data.size(0)
                total_elements += batch_size
                out_data = self.model(in_data)
                self.optimizer.zero_grad()
                error = self.loss.compute(out_data, labels)
                error.backward()
                self.optimizer.step()
                loss += error / batch_size
                if verbose >= 2:
                    print("Batch "+str(b+1)+"/"+str(total_batches)+" done - Avg. loss = "+str(loss/total_elements))
            self.dataset.__reset_train__()
            th.save(self.model.state_dict(), "latest_"+self.name+"_"+self.dataset.name+".th")
            if verbose >= 1:
                print("Epoch "+str(e+1)+"/"+str(epochs)+" done - Avg. loss = "+str(loss/total_elements))
            if evaluate:
                self.model.eval()
                score = self.__evaluate__(verbose=verbose)
                scores += [score]
                if best_score == None or score > best_score:
                    best_score = score
                    th.save(self.model.state_dict(), "best_"+self.name+"_"+self.dataset.name+".th")
                self.model.train()
        if evaluate:
            return scores
            
    def __evaluate__(self, verbose=1):
        total_batches = len(self.dataset.dev_iter)
        self.dataset.__reset_dev__()
        for b in range(total_batches):
            in_data, labels = self.dataset.__next_dev__()
            out_data = self.model(in_data)
            self.metric.step(out_data, labels)
        score = self.metric.score()
        self.metric.reset()
        self.dataset.__reset_dev__()
        if verbose >= 1:
            print("Metric evaluation score is :",score)
        return score
    
    def score(self, verbose=1):
        total_batches = len(self.dataset.test_iter)
        self.dataset.__reset_test__()
        for b in range(total_batches):
            in_data, labels = self.dataset.__next_test__()
            out_data = self.model(in_data)
            self.metric.step(out_data, labels)
        score = self.metric.score()
        self.metric.reset()
        self.dataset.__reset_test__()
        if verbose >= 1:
            print("Metric score is :",score)
        return score