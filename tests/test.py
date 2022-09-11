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

import weakref
import _setup
import neograd as ng
import numpy as np
from neograd.autograd.utils import process_data, unflatten_data, mul_shape_dims


class Graph:
  def __init__(self):
    self.nodes_dict = {} # Tensor is the key, its Node is the value
  
  def add_edge(self, result_node, operands):
    self.add_node(result_node)
    for operand in operands:
      if self.get_node(operand) is None:
        self.add_tensor(operand)
      operand_node = self.get_node(operand)
      result_node.add_parent(operand_node)
      operand_node.add_child(result_node)
  
  def add_node(self, node):
    self.nodes_dict[node.tens] = node

  def get_node(self, tens):
    return self.nodes_dict.get(tens)
  
  def add_tensor(self, tens):
    self.nodes_dict[tens] = Node(tens)
  
  def zero_grad(self):
    for tens in self.nodes_dict:
      tens.zero_grad()
  
  def reset_graph(self):
    self.nodes_dict = {}

NG_GRAPH = Graph()


class Node:
  def __init__(self, tens):
    self.tens = tens
    self.children = []
    self.parents = []
    self.parent_broadcast_shape = None
    self.needs_broadcasting = None
    self.backward_fn = None
    self.visited = False
  
  def top_sort(self):
    sorted_tensors = []
    if self.are_children_visited():
      self.visited = True
      sorted_tensors.append(self.tens)
      for parent in self.parents:
        if not(parent.visited):
          sorted_tensors+=parent.top_sort()
    else:
      for child in self.children:
        if not(child.visited):
          sorted_tensors+=child.top_sort()
      self.visited = True
      sorted_tensors.append(self.tens)
    return sorted_tensors
  
  def backward(self):
    self.reset_visited()
    sorted_tensors = self.top_sort()
    for tens in sorted_tensors:
      if tens.requires_grad:
        tens._backward()

  def visit_all_children(self):
    for child in self.children:
      child.visited = True

  def are_children_visited(self):
    for child in self.children:
      if not(child.visited):
        return False
    return True

  def reset_visited(self):
    self.visited = False
    for parent in self.parents:
      parent.visited = False
  
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
  
  def backward(self, upper_grad=1.):
    upper_grad = process_data(upper_grad)
    if self.shape!=upper_grad.shape:
      raise ValueError("Shapes of grad and Tensor data must match!")
    self.grad+=upper_grad
    node = NG_GRAPH.get_node(self)
    node.backward()
    NG_GRAPH.reset_graph() # if some operations are done after backward and then backward is again called, then it results in incomplete graph
  
  def _backward(self):
    node = NG_GRAPH.get_node(self)
    for child in node.children:
      child.backward_fn(*[node.tens for node in child.parents])
      upper_grad = child.tens.grad
      if child.needs_broadcasting:
        upper_grad = upper_grad.flatten()
      grad = self.grad_fn(upper_grad)
      grad = unflatten_data(grad, self.shape, child.parent_broadcast_shape)
      grad = grad.reshape(self.shape)
      self.grad+=grad
  
  def set_grad_fn(self, grad_fn):
    if self.requires_grad:
      self.grad_fn = grad_fn
    else:
      self.grad_fn = None
  
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


class Operation:
  def __init__(self, operation, needs_broadcasting):
    self.operation = weakref.proxy(operation)
    self.needs_broadcasting = needs_broadcasting
  
  def process_operands(self, operands):
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_tensors(self, *operands):
    tensors = self.process_operands(operands)
    if len(tensors)==0:
      return None
    elif len(tensors)==1:
      return tensors[0]
    else:
      return tensors
  
  def get_broadcast_shape(self, *tensors):
    if self.needs_broadcasting:
      try:
        return np.broadcast_shapes(*(tens.data.shape for tens in tensors))
      except ValueError:
        return None
    else:
      return None
  
  def check_result_requires_grad(self, tensors):
    for tens in tensors:
      if tens.requires_grad:
        return True
    return False
  
  def get_result_tensor(self, result, *tensors):
    graph = NG_GRAPH
    result = result.astype(np.ndarray)
    result_tensor = Tensor(result, self.check_result_requires_grad(tensors))
    result_node = Node(result_tensor)
    result_node.needs_broadcasting = self.needs_broadcasting
    result_node.backward_fn = self.operation.backward
    result_node.parent_broadcast_shape = self.get_broadcast_shape(*tensors)
    graph.add_edge(result_node, tensors)
    return result_tensor


class Add(Operation):
  def __init__(self):
    super().__init__(self, True)

  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data+tens2.data, tens1, tens2)

  def backward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    broadcast_shape = self.get_broadcast_shape(tens1, tens2)
    tens1.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(broadcast_shape)), ug))
    tens2.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(broadcast_shape)), ug))

def add(tens1, tens2):
  return Add().forward(tens1, tens2)


a = Tensor(1, requires_grad=True)
b = Tensor([1,2,3], requires_grad=True)
c = a+b
c.backward([1,1,1])
print(a.grad)
print(b.grad)