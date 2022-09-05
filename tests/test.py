# import _setup
# import neograd as ng
# import numpy as np
# from neograd.nn.loss_fns import BinaryCrossEntropy
# from neograd.nn.optim import GD
# from neograd.nn.utils import get_batches
# from sklearn.datasets import make_circles
# from sklearn.model_selection import train_test_split

# X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
# X_train, X_test, y_train, y_test = train_test_split(X,y)

# X_train, X_test = ng.tensor(X_train.T), ng.tensor(X_test.T)
# y_train, y_test = ng.tensor(y_train.T.reshape(1,750)), ng.tensor(y_test.T.reshape(1,250))

# num_train = 750
# num_test = 250
# num_iter = 100

# class NN(ng.nn.Model):
#   def __init__(self):
#     super().__init__(self)
#     self.stack = ng.nn.Sequential(
#       ng.nn.Linear(2,10),
#       ng.nn.ReLU(),
#       ng.nn.Linear(10,1),
#       ng.nn.Sigmoid()
#     )
  
#   def forward(self, inputs):
#     return self.stack(inputs)

# model = NN()
# loss_fn = BinaryCrossEntropy()
# optim = GD(model.get_params, 0.005)

# # train_data_gen = get_batches(X_train, y_train, num_train, 50) TODO: ISSUE WRT DIMS

# for iter in range(num_iter):
#   optim.zero_grad()
#   outputs = model(X_train)
#   loss = loss_fn(outputs, y_train)
#   loss.backward()
#   optim.step()
#   print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")

class Operation:
  def __init__(self, operation, *operands):
    self.tensors = self.process_operands(operands)
    self.operation = operation
    self.broadcast_shape = self.get_broadcast_shape()

  def process_operands(self, operands):
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_broadcast_shape(self):
    return np.broadcast_shapes(*(tens.data.shape for tens in self.tensors))
  
  def check_result_requires_grad(self):
    for tens in self.tensors:
      if tens.requires_grad:
        return True
    return False
  
  def add_children(self, result_node):
    for tens in self.tensors:
      if tens.node is not None:
        tens.node.add_child(result_node)
  
  def get_result_tensor(self, result):
    result = result.astype(np.ndarray)
    result_tensor = Tensor(result, requires_grad=self.check_result_requires_grad())
    result_tensor.node = Node(self.operation)
    self.add_children(result_tensor.node)
    return result_tensor

class Node:
  def __init__(self, operation):
    self.operation = operation
    self.result_tensor = None
    self.children = []
    self.is_visited = False
  
  def add_child(self, child):
    self.children.append(child)
  
  def top_sort(self):
    sorted_nodes = []
    for tens in self.tensors:
      if tens.node is not None and not(tens.node.is_visited):
        if len(tens.node.children)==0 or self.check_if_all_children_visited(tens.node): # check if it has no children or all the children are already visited
          tens.node.is_visited = True        
          sorted_nodes.append(tens.node)
        else:
          for child in tens.node.children:
            sorted_nodes+=child.top_sort()
    self.is_visited = True
    sorted_nodes.append(self)
    return sorted_nodes
  
  def reset_is_visited(self):
    self.is_visited = False
    for tens in self.tensors:
      if tens.node is not None:
        self.reset_is_visited(tens.node)
  
  def check_if_all_children_visited(self, node):
    for child in node.children:
      if not(child.is_visited):
        return False
    return True
  
  def backward(self):
    self.reset_is_visited()
    for node in self.top_sort():
      node._backward()
  
  def _backward(self):
    self.operation.backward(self.result_tensor.grad) # do the gradient addition in backward pass

class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = data
    self.requires_grad = requires_grad
    self.node = None
    self.grad = 0.0
  
  def zero_grad(self):
    self.grad = 0.0
  
  def backward(self, upper_grad):
    self.grad = upper_grad
    self.node.backward()
  
  @property
  def data(self):
    return self._data
  
  @property.setter
  def data(self):
    self._data = process_data(data)