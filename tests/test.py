import _setup
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

from neograd.autograd.utils import mul_shape_dims, unflatten_data, process_data
import numpy as np

class Operation:
  def __init__(self, operation, needs_broadcasting, *operands):
    self.tensors = self.process_operands(operands)
    self.result_tensor = None
    self.operation = operation
    self.needs_broadcasting = needs_broadcasting
    self.broadcast_shape = self.get_broadcast_shape()

  def process_operands(self, operands):
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_processed_tensors(self):
    if len(self.tensors)==1:
      return self.tensors[0]
    return self.tensors
  
  def get_broadcast_shape(self):
    return np.broadcast_shapes(*(tens.data.shape for tens in self.tensors))
  
  def check_result_requires_grad(self):
    for tens in self.tensors:
      if tens.requires_grad:
        return True
    return False
  
  def add_children(self):
    for tens in self.tensors:
      if tens.node is not None:
        tens.node.add_child(self.result_tensor.node)
  
  def get_result_tensor(self, result):
    result = result.astype(np.ndarray)
    self.result_tensor = Tensor(result, requires_grad=self.check_result_requires_grad())
    self.result_tensor.node = Node(self.operation)
    self.add_children()
    return self.result_tensor

class Node:
  def __init__(self, operation):
    self.operation = operation
    self.children = []
    self.is_visited = False
  
  def add_child(self, child):
    self.children.append(child)
  
  def top_sort(self, is_start=False):
    sorted_nodes = []
    if len(self.children)==0 or self.check_if_all_children_visited() or is_start: # children are all resolved, add the current node and go to its operands or this is the starting node
      self.is_visited = True        
      sorted_nodes.append(self)
      for tens in self.operation.tensors:
        if tens.node is not None:
          sorted_nodes+=tens.node.top_sort()
    else: # First resolve the children, then add current node
      for child in self.children:
        sorted_nodes+=child.top_sort()
      self.is_visited = True
      sorted_nodes.append(self)
    return sorted_nodes
  
  def reset_is_visited(self):
    self.is_visited = False
    for tens in self.operation.tensors:
      if tens.node is not None:
        tens.node.reset_is_visited()
  
  def check_if_all_children_visited(self):
    for child in self.children:
      if not(child.is_visited):
        return False
    return True
  
  def backward(self):
    self.reset_is_visited()
    for node in self.top_sort():
      node._backward()
  
  def _backward(self):
    self.operation.backward() # do the gradient addition in backward pass
    upper_grad = self.operation.result_tensor.grad
    if self.operation.needs_broadcasting:
      upper_grad = upper_grad.flatten()
    broadcast_shape = self.operation.broadcast_shape
    for tens in self.operation.tensors:
      if tens.requires_grad:
        grad = tens.grad_fn(upper_grad)
        grad = unflatten_data(grad, tens.shape, broadcast_shape)
        grad = grad.reshape(tens.shape)
        tens.grad+=grad

class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = process_data(data)
    self.requires_grad = requires_grad
    self.node = None
    self.grad_fn = None
    self.grad = 0.0 if requires_grad else None
  
  def zero_grad(self):
    self.grad = 0.0
  
  def backward(self, upper_grad):
    self.grad = process_data(upper_grad)
    self.node.backward()
  
  def set_grad_fn(self, grad_fn):
    if self.requires_grad:
      self.grad_fn = grad_fn
  
  @property
  def shape(self):
    return self.data.shape
  
  def __add__(self, other):
    return Add(self, other).forward()
  
  def __radd__(self, other):
    return Add(other, self).forward()

class Add(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(self.tens1.data+self.tens2.data)
  
  def backward(self):
    self.tens1.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(self.broadcast_shape)), ug))
    self.tens2.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(self.broadcast_shape)), ug))
    
a = Tensor(1, requires_grad=True)
b = Tensor([1,2,3], requires_grad=True)
c = a+b
d = c+b
d.backward([1,1,1])
print(a.grad_fn)
print(a.grad)
print(b.grad)