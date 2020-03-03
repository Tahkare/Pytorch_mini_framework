import pickle
import gzip
import torch.nn as nn
import torch.optim as optim

import framework as fk
import metrics as M
import losses as L

tr_set, val_set, te_set = pickle.load(gzip.open("mnist.pkl.gz", "rb"), encoding="latin-1")
dataset = fk.Dataset(name='MNIST', train_data = tr_set[0], train_labels=[[i] for i in tr_set[1]],
                     dev_data=val_set[0],dev_labels=[[i] for i in val_set[1]],
                     test_data=te_set[0],test_labels=[[i] for i in te_set[1]])

class Linear(nn.Module):
    def __init__(self, a, b, c):
        super(Linear, self).__init__()
        self.linear = nn.Linear(784, a)
        self.linear2 = nn.Linear(a, b)
        self.linear3 = nn.Linear(b, c)
        self.linear4 = nn.Linear(c, 10)
    def forward(self, inputs):
        return self.linear4(self.linear3(self.linear2(self.linear(inputs))))

        
model = Linear(450, 350, 150)
metric = M.Accuracy_Multi_Class()
loss = L.CrossEntropy_Multi_Class()
optimizer = optim.SGD(model.parameters(), lr=0.05)
encaps = fk.Model(name="4-MLP", model=model, loss=loss, optimizer=optimizer, metric=metric, dataset=dataset)

encaps.hyper_parameters(parameters=[{'a':[450,500,550]},{'b':[250,300,350],'c':[100,150,200]}], epochs=3, verbose=2, use_loss=True)
encaps.train(epochs=10)
encaps.restore_best()
encaps.score()
encaps.plot_scores()