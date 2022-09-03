import numpy as np
from ..node import Node
from ..tensor import Tensor

class Operation:
  def __init__(self, operation, needs_broadcasting=False, *operands):
    self.tensors = self.process_operands(operands)
    self.needs_broadcasting = needs_broadcasting
    self.operation = operation
    self.result_tensor = None
    self.node = Node(self.operation)
    self.add_grad_fn()
    self.add_broadcast_shape()
  
  def process_operands(self, operands):
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_broadcast_shape(self):
    if self.needs_broadcasting:
      final_broadcasted_shape = np.broadcast_shapes(*(tens.data.shape for tens in self.tensors))
    else:
      final_broadcasted_shape = None
    return final_broadcasted_shape
  
  def add_broadcast_shape(self):
    broadcasted_shape = self.get_broadcast_shape()
    for tens in self.tensors:
      if broadcasted_shape is None:
        tens.broadcasted_shape = tens.data.shape
      else:
        tens.broadcasted_shape = broadcasted_shape
  
  def add_grad_fn(self):
    operation_cls = self.operation.__class__
    operation_cls_dict = operation_cls.__dict__
    if ('backward' in operation_cls_dict) and ('backward_fns' in operation_cls_dict):
      raise ValueError("Only one among backward and backward_fns must be defined")
    elif 'backward' in operation_cls_dict:
      for tens in self.tensors:
        tens.grad_fn = operation_cls.backward
    elif 'backward_fns' in operation_cls_dict:
      backward_fns = operation_cls.backward_fns(self.operation)
      if len(backward_fns)!=len(self.tensors):
        raise ValueError("Number of functions in backward_fns must be equal to the number of operands")
      else:
        for i,tens in enumerate(self.tensors):
          tens.grad_fn = backward_fns[i]
    else:
      raise ValueError("backward or backward_fns must be defined")
  
  def check_result_requires_grad(self):
    requires_grad = False
    for tens in self.tensors:
      if tens.requires_grad:
        requires_grad = True
        break
    return requires_grad
  
  def get_result_tensor(self, result_data):
    result_data = result_data.astype(np.ndarray)
    self.result_tensor = Tensor(result_data, self.check_result_requires_grad())
    self.result_tensor.node = self.node
    return self.result_tensor