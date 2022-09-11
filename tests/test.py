# import neograd as ng
# import numpy as np
# from neograd.nn.loss_fns import BinaryCrossEntropy
# from neograd.nn.optim import GD
# from neograd.nn.utils import get_batches
# from sklearn.datasets import make_circles
# from sklearn.model_selection import train_test_split
# import time

# X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
# X_train, X_test, y_train, y_test = train_test_split(X,y)

# X_train, X_test = ng.tensor(X_train.T), ng.tensor(X_test.T)
# y_train, y_test = ng.tensor(y_train.T.reshape(1,750)), ng.tensor(y_test.T.reshape(1,250))

# num_train = 750
# num_test = 250
# num_iter = 1

# class NN(ng.nn.Model):
#   def __init__(self):
#     super().__init__(self)
#     self.stack = ng.nn.Sequential(
#       ng.nn.Linear(2,100),
#       ng.nn.ReLU(),
#       ng.nn.Linear(100,1),
#       ng.nn.Sigmoid()
#     )
  
#   def forward(self, inputs):
#     return self.stack(inputs)

# model = NN()
# loss_fn = BinaryCrossEntropy()
# optim = GD(model.get_params(), 0.5)

# # train_data_gen = get_batches(X_train, y_train, num_train, 50) TODO: ISSUE WRT DIMS

# for iter in range(num_iter):
#   optim.zero_grad()
#   outputs = model(X_train)
#   loss = loss_fn(outputs, y_train)
#   loss.backward()
#   optim.step()
#   print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")


class Graph:
  def __init__(self):
    self.nodes_dict = {} # Tensor is the key, its Node is the value
  
  def add_edge(self, *operands, result):
    self.add_node(result) # Result is always a new tensor, so add it
    result_node = self.get_node(result)
    for operand in operands:
      if self.get_node(operand) is None:
        self.add_node(operand)
      operand_node = self.get_node(operand)
      result_node.add_parent(operand_node)
      operand_node.add_child(result_node)
  
  def get_node(self, tens):
    return self.nodes_dict.get(tens)
  
  def add_node(self, tens):
    self.nodes_dict[tens] = Node()
  
  def zero_grad(self):
    for tens in self.nodes_dict:
      tens.zero_grad()
  
  def reset_graph(self):
    self.nodes_dict = {}


class Node:
  def __init__(self):
    self.children = []
    self.parents = []
  
  def add_child(self, other):
    self.children.append(other)
  
  def add_parent(self, other):
    self.parents.append(other)


class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = data
    self.requires_grad = requires_grad
    self.grad = 0. if requires_grad else None
    self.grad_fn = None
  
  def zero_grad(self):
    self.grad = 0. if self.requires_grad else None
  
  @property
  def data(self):
    return self._data
  
  @data.setter
  def data(self, data):
    self._data = process_data(data)

  @property
  def shape(self):
    return self.data.shape

  def __add__(self, other):
    return add(self, other)
  
  def __radd__(self, other):
    return add(other, self)