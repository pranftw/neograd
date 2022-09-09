import numpy as np
from .node import Node
from .utils import mul_shape_dims
import weakref


class Operation:
  def __init__(self, operation, needs_broadcasting, *operands):
    self.tensors = self.process_operands(operands)
    self.operation = weakref.proxy(operation)
    self.needs_broadcasting = needs_broadcasting

  def process_operands(self, operands):
    from .tensor import Tensor
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
      operand.add_child(weakref.proxy(result))
      result.add_operand(operand)
  
  def get_result_tensor(self, result):
    from .tensor import Tensor
    result = result.astype(np.ndarray)
    result = Tensor(result, requires_grad=self.check_result_requires_grad())
    result.needs_broadcasting = self.needs_broadcasting
    result.backward_fn = self.operation.backward
    result.operand_broadcast_shape = self.get_broadcast_shape()
    self.add_edges(result)
    return result


# <------------ADD------------>

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

def add(tens1, tens2):
  return Add(tens1, tens2).forward()


# <------------SUB------------>

class Sub(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
  
  def forward(self):
    tens1, tens2 = self.get_tensors()
    return self.get_result_tensor(tens1.data-tens2.data)
  
  def backward(self):
    broadcast_shape = self.get_broadcast_shape()
    tens1, tens2 = self.get_tensors()
    tens1.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(broadcast_shape)), ug))
    tens2.set_grad_fn(lambda ug:np.dot(-np.eye(mul_shape_dims(broadcast_shape)), ug))

def sub(tens1, tens2):
  return Sub(tens1, tens2).forward()


# <------------MUL------------>

class Mul(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
  
  def forward(self):
    tens1, tens2 = self.get_tensors()
    return self.get_result_tensor(tens1.data*tens2.data)
  
  def backward(self):
    broadcast_shape = self.get_broadcast_shape()
    tens1, tens2 = self.get_tensors()
    tens1.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(tens2.data, broadcast_shape))), ug))
    tens2.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(tens1.data, broadcast_shape))), ug))

def mul(tens1, tens2):
  return Mul(tens1, tens2).forward()


# <------------DIV------------>

class Div(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
  
  def forward(self):
    tens1, tens2 = self.get_tensors()
    return self.get_result_tensor(tens1.data/tens2.data)
  
  def backward(self):
    broadcast_shape = self.get_broadcast_shape()
    tens1, tens2 = self.get_tensors()
    tens1.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(1/tens2.data, broadcast_shape))), ug))
    tens2.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to((-1*tens1.data)/np.power(tens2.data, 2), broadcast_shape))), ug))

def div(tens1, tens2):
  return Div(tens1, tens2).forward()


# <------------DOT------------>

class Dot(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, False, tens1, tens2)
  
  def forward(self):
    tens1, tens2 = self.get_tensors()
    return self.get_result_tensor(np.dot(tens1.data, tens2.data))
  
  def backward(self):
    tens1, tens2 = self.get_tensors()
    tens1.set_grad_fn(lambda ug:np.dot(ug, tens2.data.T))
    tens2.set_grad_fn(lambda ug:np.dot(tens1.data.T, ug))

def dot(tens1, tens2):
  return Dot(tens1, tens2).forward()


# <------------EXP------------>

class Exp(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
  
  def forward(self):
    tens = self.get_tensors()
    return self.get_result_tensor(tens.local_grad)
  
  def backward(self):
    tens = self.get_tensors()
    tens.set_grad_fn(lambda ug:np.exp(tens.data)*ug)

def exp(tens):
  return Exp(tens).forward()


# <------------LOG------------>

class Log(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
  
  def forward(self):
    tens = self.get_tensors()
    return self.get_result_tensor(np.log(tens.data))
  
  def backward(self):
    tens = self.get_tensors()
    tens.set_grad_fn(lambda ug:(1/tens.data)*ug)

def log(tens):
  return Log(tens).forward()


# <------------POW------------>

class Pow(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
  
  def forward(self):
    tens1, tens2 = self.get_tensors()
    return self.get_result_tensor(np.power(tens1.data, tens2.data))
  
  def backward(self):
    result = np.power(tens1.data, tens2.data)
    tens1, tens2 = self.get_tensors()
    tens1.set_grad_fn(lambda ug:(np.power(tens1.data, tens2.data-1) * tens2.data).flatten()*ug)
    tens2.set_grad_fn(lambda ug:(result*np.log(tens1.data)).flatten()*ug)

def pow(tens1, tens2):
  return Pow(tens1, tens2).forward()


# <------------SUM------------>

class Sum(Operation):
  def __init__(self, tens, axis=None):
    super().__init__(self, True, tens)
    self.axis = axis
  
  def forward(self):
    tens = self.get_tensors()
    return self.get_result_tensor(np.sum(tens.data, axis=self.axis))
  
  def backward(self):
    tens = self.get_tensors()
    def grad_backward(ug):
      tens_shape = list(tens.shape)
      if self.axis is not None:
        try:
          tens_shape[self.axis] = 1
        except IndexError:
          pass
        lg = np.eye(mul_shape_dims(tuple(tens_shape)))
      else:
        lg = np.ones(tens.shape)

      if self.axis is not None:
        grads = np.dot(lg, ug)
        try:
          num_repeat = tens.shape[self.axis]
        except IndexError:
          num_repeat = 1
        grads = grads[np.newaxis]
        grads = np.concatenate([grads]*num_repeat)
      else:
        grads = lg*ug
      return grads
    tens.set_grad_fn(grad_backward)

def sum(tens, axis=None):
  return Sum(tens, axis).forward()


# <------------TRANSPOSE------------>

class Transpose(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
  
  def forward(self):
    tens = self.get_tensors()
    return self.get_result_tensor(tens.data.T)

  def backward(self):
    tens = self.get_tensors()
    tens.set_grad_fn(lambda ug:ug.T)

def transpose(tens):
  return Transpose(tens).forward()


# <------------RELU------------>

class ReLU(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
  
  def forward(self):
    tens = self.get_tensors()
    return self.get_result_tensor(np.maximum(0, tens.data))
  
  def backward(self):
    tens = self.get_tensors()
    tens.set_grad_fn(lambda ug:np.where(tens.data>=0, 1, 0)*ug)

def relu(tens):
  return ReLU(tens).forward()


# <------------SIGMOID------------>

class Sigmoid(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
  
  def forward(self):
    tens = self.get_tensors()
    return self.get_result_tensor(1/(1+np.exp(-tens.data)))
  
  def backward(self):
    tens = self.get_tensors()
    result = 1/(1+np.exp(-tens.data))
    tens.set_grad_fn(lambda ug:(result*(1-result))*ug)

def sigmoid(tens):
  return Sigmoid(tens).forward()


# <------------TANH------------>

class Tanh(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
  
  def forward(self):
    tens = self.get_tensors()
    return self.get_result_tensor(np.tanh(tens.data))
  
  def backward(self):
    tens = self.get_tensors()
    result = np.tanh(tens.data)
    self.tens.set_grad_fn(lambda ug:(1-np.power(result,2))*ug)

def tanh(tens):
  return Tanh(tens).forward()