# Pytorch Mini Framework
A small framework to facilitate training of models created using Pytorch

#### Dataset
The Dataset class expects a name and both training, developement and testing datasets. <br> Each dataset can be passed directly in the Dataloader format using the xxxxx_set argument or as two n-dimensional arrays where each sample must be in an array using xxxxx_data and xxxxx_labels.

You can also specify the batch size and the types of the input and the labels using data_type and label_type with either 'int', 'float' or 'long'.

#### Model
The Model class expects a name, a Pytorch model, a loss function (either from the losses module or manually defined), an optimizer, a metric (either from the metrics module or manually defined) and a dataset. <br>
The functions train and score can be called to either train the model with a given number of epochs or evaluate the metric on the test set.

#### Losses
The losses module defines a series of loss functions. <br>
A loss class must define a single function compute(out_data, labels) taking as arguments the output of the model and the associated labels for a batch of input data and returning the result of the loss function.

#### Metrics
The metrics module defines a series of evaluation functions to evaluate the performance of the model. <br>
A metric class must define three functions : score() which returns the score, reset() which initializes the parameters needed to compute the metric and step(out_data, labels) which updates the metric based on the output of the model and the associated labels for a batch of input data
