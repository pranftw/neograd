import _setup
from neograd.autograd.utils import process_data, mul_shape_dims, unflatten_data
import weakref
import numpy as np


"""
  When operation occurs add operands, set backward_fn
"""

class Node:
  def __init__(self):
    self.operands = [] # can this be weakref
    self.children = []
    self.needs_broadcasting = True
    self.visited = False
  
  def top_sort(self):
    sorted_tensors = []
    if self.are_children_visited(): # All children are resolved
      self.visited = True
      sorted_tensors.append(self)
      for operand in self.operands:
        if not(operand.visited):
          sorted_tensors+=operand.top_sort()
    else:
      for child in self.children: # Resolve children first
        if not(child.visited):
          sorted_tensors+=child.top_sort()
      self.visited = False
      sorted_tensors.append(self)
    return sorted_tensors
  
  def add_child(self, child):
    self.children.append(child)
  
  def add_operand(self, operand):
    self.operands.append(operand)
  
  def node_backward(self):
    self.reset_visited()
    self.visit_all_children()
    sorted_tensors = self.top_sort()
    self.reset_visited()
    self.visit_all_children()
    for tens in sorted_tensors:
      tens.visited = True
      upper_grad = tens.grad
      tens._backward(upper_grad)
  
  def _backward(self, upper_grad):
    if len(self.operands)!=0:
      if self.needs_broadcasting:
        upper_grad = upper_grad.flatten()
      self.backward_fn()
      for operand in self.operands:
        if operand.requires_grad and operand.are_children_visited():
          operand.visited = True
          grad = operand.grad_fn(upper_grad)
          grad = unflatten_data(grad, operand.shape, self.shape)
          grad = grad.reshape(operand.shape)
          operand.grad+=grad
  
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
    for operand in self.operands:
      operand.visited = False 


class Tensor(Node):
  def __init__(self, data, requires_grad=False):
    super().__init__()
    self.data = data
    self.requires_grad = requires_grad
    self.grad = 0. if requires_grad else None
    self.grad_fn = None
    self.backward_fn = None
  
  def backward(self, upper_grad=1.0):
    upper_grad = process_data(upper_grad)
    self.grad+=upper_grad
    self.node_backward()
  
  def zero_grad(self):
    if self.grad:
      self.grad = 0.
  
  def set_grad_fn(self, grad_fn):
    if self.requires_grad:
      self.grad_fn = grad_fn
    else:
      self.grad_fn = None
  
  def __add__(self, other):
    return Add(self, other).forward()
  
  def __radd__(self, other):
    return Add(other, self).forward()
  
  @property
  def shape(self):
    return self.data.shape
  
  @property
  def data(self):
    return self._data
  
  @data.setter
  def data(self, data):
    self._data = process_data(data)


class Operation:
  def __init__(self, operation, needs_broadcasting, *operands):
    tensors = self.process_operands(operands)
    self.tensors = [weakref.proxy(tens) for tens in tensors]
    self.operation = weakref.proxy(operation)
    self.needs_broadcasting = needs_broadcasting

  def process_operands(self, operands):
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_tensors(self):
    if len(self.tensors)==0:
      return None
    elif len(self.tensors)==1:
      return self.tensors[0]
    else:
      return self.tensors
  
  def get_broadcast_shape(self):
    if self.needs_broadcasting:
      try:
        return np.broadcast_shapes(*(tens.data.shape for tens in self.tensors))
      except ValueError:
        return None
    else:
      return None
  
  def check_result_requires_grad(self):
    for tens in self.tensors:
      if tens.requires_grad:
        return True
    return False
  
  def add_edges(self, result):
    for operand in self.tensors:
      operand.add_child(result)
      result.add_operand(operand)
  
  def get_result_tensor(self, result):
    result = result.astype(np.ndarray)
    result = Tensor(result, requires_grad=self.check_result_requires_grad())
    result.needs_broadcasting = self.needs_broadcasting
    result.backward_fn = self.operation.backward
    self.add_edges(result)
    return result


class Add(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)

  def forward(self):
    tens1, tens2 = self.get_tensors()
    return self.get_result_tensor(tens1.data+tens2.data)

  def backward(self):
    broadcast_shape = self.get_broadcast_shape()
    tens1, tens2 = self.get_tensors()
    tens1.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(broadcast_shape)), ug))
    tens2.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(broadcast_shape)), ug))

a = Tensor(1, requires_grad=True)
b = Tensor([1,1,1], requires_grad=True)
c = a+b
c.backward([1,1,1])
print(a.grad)
print(b.grad)