# Neograd
### A Deep Learning framework created from scratch with Python and NumPy

<br>
<div align="center">
  <img width="251" alt="image" src="https://github.com/pranftw/neograd/raw/main/ng.png">
</div>
<br>

[![Neograd Tests](https://github.com/pranftw/neograd/actions/workflows/python-app.yml/badge.svg)](https://github.com/pranftw/neograd/actions/workflows/python-app.yml)
[![Downloads](https://static.pepy.tech/personalized-badge/neograd?period=month&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)](https://pepy.tech/project/neograd)

## Motivation
I firmly believe that in order to understand something completely, you have to build it on your own from scratch. I used to do gradient calculation analytically, and thought that autograd was some kind of magic. So this was initially built to understand autograd but later on its scope was extended. You might be wondering, there are already many frameworks like TensorFlow and PyTorch that are very popular, and why did I have to create another one? The answer is that these have very complex codebases that are difficult to grasp. So I intend that this repository be used as an educational tool in order to understand how things work under the hood in these giant frameworks, with code that is intuitive and easily readable.

## Installation
`pip install neograd`

## Features
### Automatic Differentiation
`autograd` offers automatic differentiation, implemented for the most commonly required operations for vectors of any dimension, with broadcasting capabilities
```
import neograd as ng
a = ng.tensor(3, requires_grad=True)
b = ng.tensor([1,2,3], requires_grad=True)
c = a+b
c.backward([1,1,1])
print(a.grad)
print(b.grad)
```
### Custom autograd operations
If you wanted a custom operation to have `autograd` capabilities, those can be defined with very simple interface each having a forward method and a backward method
```
class Custom(Operation):
  def forward(self):
    pass
  def backward(self):
    pass
```
### Gradient Checking
Debug your models/functions with Gradient Checking, to ensure that the gradients are getting propagated correctly
### Highly customizable
Create your own custom layers, optimizers, loss functions which provides more flexibility to create anything you
desire
### PyTorch like API
PyTorch's API is one of the best and one the most elegant API designs, so we've leveraged the same
### Neural Network module
`nn` contains some of the most commonly used optimizers, activations and loss functions required to train a Neural Network
### Save and Load weights
Trained a model already? Then save the weights onto a file and load them whenever required
### Checkpoints
Let's say you're training a model and your computer runs out of juice and if you'd waited until training was finished, to save the weights, then you'd lose all the weights. To prevent this, checkpoint your model with various sessions to save the weights during regular intervals with additional supporting data

## Example
```
import neograd as ng
import numpy as np
from neograd.nn.loss import BCE
from neograd.nn.optim import Adam
from neograd.autograd.utils import grad_check
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X,y)

num_train = 750
num_test = 250
num_iter = 50

X_train, X_test = ng.tensor(X_train[:num_train,:]), ng.tensor(X_test[:num_test,:])
y_train, y_test = ng.tensor(y_train[:num_train].reshape(num_train,1)), ng.tensor(y_test[:num_test].reshape(num_test,1))

class NN(ng.nn.Model):
  def __init__(self):
    self.stack = ng.nn.Sequential(
      ng.nn.Linear(2,100),
      ng.nn.ReLU(),
      ng.nn.Linear(100,1),
      ng.nn.Sigmoid()
    )
  
  def forward(self, inputs):
    return self.stack(inputs)

model = NN()
loss_fn = BCE()
optim = Adam(model.get_params(), 0.05)

for iter in range(num_iter):
  optim.zero_grad()
  outputs = model(X_train)
  loss = loss_fn(outputs, y_train)
  loss.backward()
  optim.step()
  print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")

with model.eval():
  test_outputs = model(X_test)
  preds = np.where(test_outputs.data>=0.5, 1, 0)

print(classification_report(y_test.data.astype(int).flatten(), preds.flatten()))
print(accuracy_score(y_test.data.astype(int).flatten(), preds.flatten()))

grad_check(model, X_train, y_train, loss_fn)
```

## Resources
- A big thank you to Andrej Karpathy for his CS231n lecture on Backpropagation which was instrumental in helping me gain a good grasp of the basic mechanisms of autograd
[Lecture](https://youtu.be/i94OvYb6noo)
- Thanks to Terance Parr and Jeremy Howard for their paper The Matrix Calculus You Need For Deep Learning which helped me get rid of my fear for matrix calculus, that is beautifully written starting from the very fundamentals and slowly transitioning into advanced topics
[Paper](https://arxiv.org/abs/1802.01528)
