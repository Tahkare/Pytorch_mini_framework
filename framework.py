""" 
Created on Tue Feb 18 2020

@author : Alban Petit
"""

import torch as th
import torch.utils.data as data
from matplotlib import pyplot as plt
import itertools

device = th.device("cpu")

class Dataset:
    
    def __merge__(self, datas, labels, data_type='float', label_type='long', batch_size=64, shuffle=True):
        tensor_type = {'int' : th.IntTensor, 'float' : th.FloatTensor, 'long' : th.LongTensor, 'double' : th.DoubleTensor}
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
        self.model = model.to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.dataset = dataset
        self.metric = metric
        self.train_scores = []
        self.dev_scores = []
    
    def hyper_parameters(self, parameters=None, epochs=1, verbose=1):
        if parameters != None:
            default_values = {}
            for param_set in parameters:
                for param in param_set.keys():
                    default_values[param] = param_set[param][0]
            for param_set in parameters:
                if verbose >= 1:
                    print("Selecting values for parameters :",list(param_set.keys()))
                best_score = None
                combinations = list(itertools.product(*(param_set[param] for param in param_set.keys())))
                for combination in combinations:
                    tmp = {}
                    for label,value in zip(param_set.keys(), combination):
                        tmp[label] = value
                    for label in default_values.keys():
                        if label not in tmp.keys():
                            tmp[label] = default_values[label]
                    if verbose >= 2:
                        display = {param : tmp[param] for param in param_set.keys()}
                        print("Testing combination",display)
                    self.model.__init__(**tmp)
                    for param_group in [{'params' : list(self.model.parameters())}]:
                        self.optimizer.add_param_group(param_group)
                    self.train(epochs=epochs, verbose=0)
                    train_scores, dev_scores = self.get_scores()
                    if verbose >= 2:
                        print("Metric score on dev set is :", dev_scores[-1])
                    if best_score == None or dev_scores[-1] > best_score:
                        best_score = dev_scores[-1]
                        for label in param_set.keys():
                            default_values[label] = tmp[label]
                if verbose >= 1:
                    display = {param : default_values[param] for param in param_set.keys()}
                    print("Best combination is",display)
            if verbose >= 1:
                print("Combination selected :",default_values)
            self.model.__init__(**default_values)
            for param_group in [{'params' : list(self.model.parameters())}]:
                self.optimizer.add_param_group(param_group)
                
    def train(self, epochs=10, evaluate=True, verbose=1):
        best_score = None
        self.__reset_scores__()
        self.dataset.__reset_train__()
        for e in range(epochs):
            self.model.train()
            self.metric.reset()
            loss = 0
            total_batches = len(self.dataset.train_iter)
            batch_size = None
            total_elements = 0
            for b in range(total_batches):
                in_data, labels = self.dataset.__next_train__()
                in_data, labels = in_data.to(device), labels.to(device)
                batch_size = in_data.size(0)
                total_elements += batch_size
                out_data = self.model(in_data)
                self.metric.step(in_data, out_data, labels)
                self.optimizer.zero_grad()
                error = self.loss.compute(in_data, out_data, labels)
                error.backward()
                self.optimizer.step()
                loss += error
                if verbose >= 2:
                    print("Batch "+str(b+1)+"/"+str(total_batches)+" done - Avg. loss = "+str(loss/total_elements))
            self.dataset.__reset_train__()
            th.save(self.model.state_dict(), "latest_"+self.name+"_"+self.dataset.name+".th")
            if verbose >= 1:
                print("Epoch "+str(e+1)+"/"+str(epochs)+" done - Avg. loss = "+str(loss/total_elements))
            if evaluate:
                score = self.metric.score()
                if verbose >= 1:
                    print("Metric score on train set is :",score)
                self.train_scores += [score]
                self.metric.reset()
                score = self.__evaluate__(verbose=verbose)
                self.dev_scores += [score]
                if best_score == None or score > best_score:
                    best_score = score
                    th.save(self.model.state_dict(), "best_"+self.name+"_"+self.dataset.name+".th")
            self.metric.reset()
            
    def __evaluate__(self, verbose=1):
        self.model.eval()
        total_batches = len(self.dataset.dev_iter)
        self.dataset.__reset_dev__()
        for b in range(total_batches):
            in_data, labels = self.dataset.__next_dev__()
            in_data, labels = in_data.to(device), labels.to(device)
            out_data = self.model(in_data)
            self.metric.step(in_data, out_data, labels)
        score = self.metric.score()
        self.metric.reset()
        self.dataset.__reset_dev__()
        if verbose >= 1:
            print("Metric score on dev set is :",score)
        return score
    
    def score(self, verbose=1):
        self.model.eval()
        total_batches = len(self.dataset.test_iter)
        self.dataset.__reset_test__()
        for b in range(total_batches):
            in_data, labels = self.dataset.__next_test__()
            in_data, labels = in_data.to(device), labels.to(device)
            out_data = self.model(in_data)
            self.metric.step(in_data, out_data, labels)
        score = self.metric.score()
        self.metric.reset()
        self.dataset.__reset_test__()
        if verbose >= 1:
            print("Metric score on test set is :",score)
        return score
    
    def restore_best(self):
        self.model.load_state_dict(th.load("best_"+self.name+"_"+self.dataset.name+".th"))
        
    def __reset_scores__(self):
        self.train_scores = []
        self.dev_scores = []
        
    def plot_scores(self, train=True, dev=True):
        plt.clf()
        tmp = [i for i in range(len(self.train_scores))]
        plt.plot(tmp, self.train_scores, label='Train score')
        plt.plot(tmp, self.dev_scores, label='Dev score')
        plt.xlabel('Epochs')
        plt.ylabel('Metric score')
        plt.title("Metric score of model "+self.name+" for dataset "+self.dataset.name)
        plt.legend()
        plt.show()
        
    def get_scores(self):
        return self.train_scores, self.dev_scores
        