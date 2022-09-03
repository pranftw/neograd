import numpy as np
from .node import Node
from .tensor import Tensor
from .utils import mul_shape_dims


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


# <------------ADD------------>

class Add(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors
  
  def forward(self):
    self.tens1.local_grad = np.eye(mul_shape_dims(self.get_broadcast_shape()))
    self.tens2.local_grad = np.eye(mul_shape_dims(self.get_broadcast_shape()))
    return self.get_result_tensor(self.tens1.data+self.tens2.data)
  
  def backward(self, local_grad, upper_grad):
    return np.dot(local_grad, upper_grad)

def add(tens1, tens2):
  return Add(tens1, tens2).forward()


# <------------SUB------------>

class Sub(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors
  
  def forward(self):
    self.tens1.local_grad = np.eye(mul_shape_dims(self.get_broadcast_shape()))
    self.tens2.local_grad = -np.eye(mul_shape_dims(self.get_broadcast_shape()))
    return self.get_result_tensor(self.tens1.data-self.tens2.data)
  
  def backward(self, local_grad, upper_grad):
    return np.dot(local_grad, upper_grad)

def sub(tens1, tens2):
  return Sub(tens1, tens2).forward()


# <------------MUL------------>

class Mul(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors

  def forward(self):
    self.tens1.local_grad = np.diag(np.ndarray.flatten(np.broadcast_to(self.tens2.data, self.tens2.broadcasted_shape)))
    self.tens2.local_grad = np.diag(np.ndarray.flatten(np.broadcast_to(self.tens1.data, self.tens1.broadcasted_shape)))
    return self.get_result_tensor(self.tens1.data*self.tens2.data)
  
  def backward(self, local_grad, upper_grad):
    return np.dot(local_grad, upper_grad)

def mul(tens1, tens2):
  return Mul(tens1, tens2).forward()


# <------------DIV------------>

class Div(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors

  def forward(self):
    self.tens1.local_grad = np.diag(np.ndarray.flatten(np.broadcast_to(1/self.tens2.data, self.tens2.broadcasted_shape)))
    self.tens2.local_grad = np.diag(np.ndarray.flatten(np.broadcast_to((-1*self.tens1.data)/np.power(self.tens2.data, 2), self.tens1.broadcasted_shape)))
    return self.get_result_tensor(self.tens1.data/self.tens2.data)
  
  def backward(self, local_grad, upper_grad):
    return np.dot(local_grad, upper_grad)

def div(tens1, tens2):
  return Div(tens1, tens2).forward()


# <------------DOT------------>

class Dot(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, False, tens1, tens2)
    self.tens1, self.tens2 = self.tensors
  
  def forward(self):
    self.tens1.local_grad = self.tens2.data
    self.tens2.local_grad = self.tens1.data
    return self.get_result_tensor(np.dot(self.tens1.data, self.tens2.data))
  
  def backward_fns(self):
    def backward1(self, local_grad, upper_grad):
      return np.dot(upper_grad, local_grad.T)
    def backward2(self, local_grad, upper_grad):
      return np.dot(local_grad.T, upper_grad)
    return backward1, backward2

def dot(tens1, tens2):
  return Dot(tens1, tens2).forward()


# <------------EXP------------>

class Exp(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.tensors[0]
  
  def forward(self):
    self.tens.local_grad = np.exp(self.tens.data)
    return self.get_result_tensor(self.tens.local_grad)
  
  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad)

def exp(tens):
  return Exp(tens).forward()


# <------------LOG------------>

class Log(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.tensors[0]
  
  def forward(self):
    self.tens.local_grad = 1/self.tens.data
    return self.get_result_tensor(np.log(self.tens.data))
  
  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad)

def log(tens):
  return Log(tens).forward()


# <------------POW------------>

class Pow(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors
  
  def forward(self):
    result = np.power(self.tens1.data, self.tens2.data)
    self.tens1.local_grad = (np.power(self.tens1.data, self.tens2.data-1) * self.tens2.data).flatten()
    self.tens2.local_grad = (result*np.log(self.tens1.data)).flatten()
    return self.get_result_tensor(result)
  
  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad)

def pow(tens1, tens2):
  return Pow(tens1, tens2).forward()


# <------------SUM------------>

class Sum(Operation):
  def __init__(self, tens, axis=None):
    super().__init__(self, True, tens)
    self.tens = self.tensors[0]
    self.axis = axis
  
  def forward(self):
    tens_shape = list(self.tens.shape)
    if self.axis is not None:
      try:
        tens_shape[self.axis] = 1
      except IndexError:
        pass
    self.tens.local_grad = np.eye(mul_shape_dims(tuple(tens_shape)))
    return self.get_result_tensor(np.sum(self.tens.data, axis=self.axis))
  
  def backward(self, local_grad, upper_grad):
    if self.axis is not None:
      grads = np.dot(local_grad, upper_grad)
      try:
        num_repeat = self.tens.shape[self.axis]
      except IndexError:
        num_repeat = 1
      grads = grads[np.newaxis]
      grads = np.concatenate([grads]*num_repeat)
    else:
      grads = np.dot(local_grad, upper_grad*np.ones(mul_shape_dims(self.tens.shape)))
    return grads

def sum(tens, axis=0):
  return Sum(tens, axis).forward()


# <------------TRANSPOSE------------>

class Transpose(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.tensors[0]
  
  def forward(self):
    self.tens.local_grad = 1
    return self.get_result_tensor(self.tens.data.T)

  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad).T

def transpose(tens):
  return Transpose(tens).forward()