import numpy as np
from .node import Node
from .utils import mul_shape_dims


class Operation:
  def __init__(self, operation, needs_broadcasting, *operands):
    self.tensors = self.process_operands(operands)
    self.result_tensor = None
    self.operation = operation
    self.needs_broadcasting = needs_broadcasting
    self.broadcast_shape = self.get_broadcast_shape()

  def process_operands(self, operands):
    from .tensor import Tensor
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
    from .tensor import Tensor
    result = result.astype(np.ndarray)
    self.result_tensor = Tensor(result, requires_grad=self.check_result_requires_grad())
    self.result_tensor.node = Node(self.operation)
    self.add_children()
    return self.result_tensor


# <------------ADD------------>

class Add(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(self.tens1.data+self.tens2.data)
  
  def backward(self):
    self.tens1.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(self.broadcast_shape)), ug))
    self.tens2.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(self.broadcast_shape)), ug))

def add(tens1, tens2):
  return Add(tens1, tens2).forward()


# <------------SUB------------>

class Sub(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(self.tens1.data-self.tens2.data)
  
  def backward(self):
    self.tens1.set_grad_fn(lambda ug:np.dot(np.eye(mul_shape_dims(self.broadcast_shape)), ug))
    self.tens2.set_grad_fn(lambda ug:np.dot(-np.eye(mul_shape_dims(self.broadcast_shape)), ug))

def sub(tens1, tens2):
  return Sub(tens1, tens2).forward()


# <------------MUL------------>

class Mul(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(self.tens1.data*self.tens2.data)
  
  def backward(self):
    self.tens1.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(self.tens2.data, self.tens2.broadcasted_shape))), ug))
    self.tens2.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(self.tens1.data, self.tens1.broadcasted_shape))), ug))

def mul(tens1, tens2):
  return Mul(tens1, tens2).forward()


# <------------DIV------------>

class Div(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(self.tens1.data/self.tens2.data)
  
  def backward(self):
    self.tens1.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(1/self.tens2.data, self.tens2.broadcasted_shape))), ug))
    self.tens2.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to((-1*self.tens1.data)/np.power(self.tens2.data, 2), self.tens1.broadcasted_shape))), ug))

def div(tens1, tens2):
  return Div(tens1, tens2).forward()


# <------------DOT------------>

class Dot(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, False, tens1, tens2)
    self.tens1, self.tens2 = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(np.dot(self.tens1.data, self.tens2.data))
  
  def backward(self):
    self.tens1.set_grad_fn(lambda ug:np.dot(ug, self.tens2.data.T))
    self.tens2.set_grad_fn(lambda ug:np.dot(self.tens1.data.T, ug))

def dot(tens1, tens2):
  return Dot(tens1, tens2).forward()


# <------------EXP------------>

class Exp(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(self.tens.local_grad)
  
  def backward(self):
    self.tens.set_grad_fn(lambda ug:np.exp(self.tens.data)*ug)

def exp(tens):
  return Exp(tens).forward()


# <------------LOG------------>

class Log(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(np.log(self.tens.data))
  
  def backward(self):
    self.tens.set_grad_fn(lambda ug:(1/self.tens.data)*ug)

def log(tens):
  return Log(tens).forward()


# <------------POW------------>

class Pow(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(np.power(self.tens1.data, self.tens2.data))
  
  def backward(self):
    self.tens1.set_grad_fn(lambda ug:(np.power(self.tens1.data, self.tens2.data-1) * self.tens2.data).flatten()*ug)
    self.tens2.set_grad_fn(lambda ug:(self.result_tensor.data*np.log(self.tens1.data)).flatten()*ug)

def pow(tens1, tens2):
  return Pow(tens1, tens2).forward()


# <------------SUM------------>

class Sum(Operation):
  def __init__(self, tens, axis=None):
    super().__init__(self, True, tens)
    self.tens = self.get_processed_tensors()
    self.axis = axis
  
  def forward(self):
    return self.get_result_tensor(np.sum(self.tens.data, axis=self.axis))
  
  def backward(self):
    def grad_backward(ug):
      tens_shape = list(self.tens.shape)
      if self.axis is not None:
        try:
          tens_shape[self.axis] = 1
        except IndexError:
          pass
        lg = np.eye(mul_shape_dims(tuple(tens_shape)))
      else:
        lg = np.ones(self.tens.shape)

      if self.axis is not None:
        grads = np.dot(lg, ug)
        try:
          num_repeat = self.tens.shape[self.axis]
        except IndexError:
          num_repeat = 1
        grads = grads[np.newaxis]
        grads = np.concatenate([grads]*num_repeat)
      else:
        grads = lg*ug
      return grads
    self.tens.set_grad_fn(grad_backward)

def sum(tens, axis=None):
  return Sum(tens, axis).forward()


# <------------TRANSPOSE------------>

class Transpose(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.get_processed_tensors()
  
  def forward(self):
    self.tens.local_grad = 1
    return self.get_result_tensor(self.tens.data.T)

  def backward(self):
    self.tens.set_grad_fn(lambda ug:ug.T)

def transpose(tens):
  return Transpose(tens).forward()


# <------------RELU------------>

class ReLU(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(np.maximum(0, self.tens.data))
  
  def backward(self):
    self.tens.set_grad_fn(lambda ug:np.where(self.tens.data>=0, 1, 0)*ug)

def relu(tens):
  return ReLU(tens).forward()


# <------------SIGMOID------------>

class Sigmoid(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(1/(1+np.exp(-self.tens.data)))
  
  def backward(self):
    self.tens.set_grad_fn(lambda ug:(self.result_tensor.data*(1-self.result_tensor.data))*ug)

def sigmoid(tens):
  return Sigmoid(tens).forward()


# <------------TANH------------>

class Tanh(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.get_processed_tensors()
  
  def forward(self):
    return self.get_result_tensor(np.tanh(self.tens.data))
  
  def backward(self):
    self.tens.set_grad_fn(lambda ug:(1-np.power(self.result_tensor.data,2))*ug)

def tanh(tens):
  return Tanh(tens).forward()