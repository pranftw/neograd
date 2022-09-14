import numpy as np
from .node import Node


class Operation:
  def __init__(self, operation, needs_broadcasting):
    self.operation = operation
    self.needs_broadcasting = needs_broadcasting
  
  def process_operands(self, operands):
    from .tensor import Tensor
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
    from .tensor import Tensor
    from .. import _NG_GRAPH
    graph = _NG_GRAPH
    result = result.astype(np.ndarray)
    result_tensor = Tensor(result, self.check_result_requires_grad(tensors))
    result_node = Node(result_tensor)
    result_node.needs_broadcasting = self.needs_broadcasting
    result_node.backward_fn = self.operation.backward
    result_node.parent_broadcast_shape = self.get_broadcast_shape(*tensors)
    graph.add_edge(result_node, tensors)
    return result_tensor


# <------------ADD------------>

class Add(Operation):
  def __init__(self):
    super().__init__(self, True)

  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data+tens2.data, tens1, tens2)

  def backward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:ug)

def add(tens1, tens2):
  return Add().forward(tens1, tens2)


# <------------SUB------------>

class Sub(Operation):
  def __init__(self):
    super().__init__(self, True)
  
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data-tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:-ug)

def sub(tens1, tens2):
  return Sub().forward(tens1, tens2)


# <------------MUL------------>

class Mul(Operation):
  def __init__(self):
    super().__init__(self, True)
  
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data*tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    broadcast_shape = self.get_broadcast_shape(tens1, tens2)
    tens1.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(tens2.data, broadcast_shape))), ug))
    tens2.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(tens1.data, broadcast_shape))), ug))

def mul(tens1, tens2):
  return Mul().forward(tens1, tens2)


# <------------DIV------------>

class Div(Operation):
  def __init__(self):
    super().__init__(self, True)
  
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data/tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    broadcast_shape = self.get_broadcast_shape(tens1, tens2)
    tens1.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to(1/tens2.data, broadcast_shape))), ug))
    tens2.set_grad_fn(lambda ug:np.dot(np.diag(np.ndarray.flatten(np.broadcast_to((-1*tens1.data)/np.power(tens2.data, 2), broadcast_shape))), ug))

def div(tens1, tens2):
  return Div().forward(tens1, tens2)


# <------------DOT------------>

class Dot(Operation):
  def __init__(self):
    super().__init__(self, False)
  
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.dot(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    tens1.set_grad_fn(lambda ug:np.dot(ug, tens2.data.T))
    tens2.set_grad_fn(lambda ug:np.dot(tens1.data.T, ug))

def dot(tens1, tens2):
  return Dot().forward(tens1, tens2)


# <------------EXP------------>

class Exp(Operation):
  def __init__(self):
    super().__init__(self, False)
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.exp(tens.data), tens)
  
  def backward(self, tens):
    tens = self.get_tensors(tens)
    tens.set_grad_fn(lambda ug:np.exp(tens.data)*ug)

def exp(tens):
  return Exp().forward(tens)


# <------------LOG------------>

class Log(Operation):
  def __init__(self):
    super().__init__(self, False)
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.log(tens.data), tens)
  
  def backward(self, tens):
    tens = self.get_tensors(tens)
    tens.set_grad_fn(lambda ug:(1/tens.data)*ug)

def log(tens):
  return Log().forward(tens)


# <------------POW------------>

class Pow(Operation):
  def __init__(self):
    super().__init__(self, True)
  
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.power(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    result = np.power(tens1.data, tens2.data)
    tens1, tens2 = self.get_tensors(tens1, tens2)
    tens1.set_grad_fn(lambda ug:(np.power(tens1.data, tens2.data-1) * tens2.data).flatten()*ug)
    tens2.set_grad_fn(lambda ug:(result*np.log(tens1.data)).flatten()*ug)

def pow(tens1, tens2):
  return Pow().forward(tens1, tens2)


# <------------SUM------------>

class Sum(Operation):
  def __init__(self, axis=None):
    super().__init__(self, True)
    self.axis = axis
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.sum(tens.data, axis=self.axis), tens)
  
  def backward(self, tens):
    tens = self.get_tensors(tens)
    
    def grad_backward(ug):
      tens_shape = list(tens.shape)
      if self.axis is not None:
        try:
          tens_shape[self.axis] = 1
        except IndexError:
          pass
        lg = 1
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
  return Sum(axis).forward(tens)


# <------------TRANSPOSE------------>

class Transpose(Operation):
  def __init__(self):
    super().__init__(self, False)
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.T, tens)

  def backward(self, tens):
    tens = self.get_tensors(tens)
    tens.set_grad_fn(lambda ug:ug.T)

def transpose(tens):
  return Transpose().forward(tens)


# <------------RELU------------>

class ReLU(Operation):
  def __init__(self):
    super().__init__(self, False)
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.maximum(0, tens.data), tens)
  
  def backward(self, tens):
    tens = self.get_tensors(tens)
    tens.set_grad_fn(lambda ug:np.where(tens.data>=0, 1, 0)*ug)

def relu(tens):
  return ReLU().forward(tens)


# <------------SIGMOID------------>

class Sigmoid(Operation):
  def __init__(self):
    super().__init__(self, False)
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(1/(1+np.exp(-tens.data)), tens)
  
  def backward(self, tens):
    tens = self.get_tensors(tens)
    result = 1/(1+np.exp(-tens.data))
    tens.set_grad_fn(lambda ug:(result*(1-result))*ug)

def sigmoid(tens):
  return Sigmoid().forward(tens)


# <------------TANH------------>

class Tanh(Operation):
  def __init__(self):
    super().__init__(self, False)
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.tanh(tens.data), tens)
  
  def backward(self, tens):
    tens = self.get_tensors(tens)
    result = np.tanh(tens.data)
    self.tens.set_grad_fn(lambda ug:(1-np.power(result,2))*ug)

def tanh(tens):
  return Tanh().forward(tens)