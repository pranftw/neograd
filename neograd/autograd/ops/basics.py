import numpy as np
from .operation import Operation


# <------------ADD------------>
class Add(Operation):
  '''Element wise addition between two Tensors or Tensor-like
  '''
  def forward(self, tens1, tens2):
    '''Calculates element wise addition

    Args:
      tens1 (Tensor or int or float or list or np.ndarray): First operand
      tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
    Returns:
      Tensor of the result
    '''
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data+tens2.data, tens1, tens2)

  def backward(self, tens1, tens2):
    '''Sets grad_fn of operands

    Local gradient is an identity matrix, that should be dotted with the upper gradient
    which results in upper gradient

    Args:
      tens1 (Tensor): First operand
      tens2 (Tensor): Second operand
    '''
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:ug)

def add(tens1, tens2):
  '''Abstraction for Add.forward

  Args:
    tens1 (Tensor or int or float or list or np.ndarray): First operand
    tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
  Returns:
    Tensor of the result
  '''
  return Add().forward(tens1, tens2)


# <------------SUB------------>
class Sub(Operation):
  '''Element wise subtraction between two Tensors or Tensor-like
  '''
  def forward(self, tens1, tens2):
    '''Calculates element wise subtraction

    Args:
      tens1 (Tensor or int or float or list or np.ndarray): First operand
      tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
    Returns:
      Tensor of the result
    '''
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data-tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    '''Sets grad_fn of operands

    Local gradient is an identity matrix, that should be dotted with the upper gradient
    which results in upper gradient, for the other one local gradient is a negative identity
    matrix which results in negative upper gradient

    Args:
      tens1 (Tensor): First operand
      tens2 (Tensor): Second operand
    '''
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:-ug)

def sub(tens1, tens2):
  '''Abstraction for Sub.forward

  Args:
    tens1 (Tensor or int or float or list or np.ndarray): First operand
    tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
  Returns:
    Tensor of the result
  '''
  return Sub().forward(tens1, tens2)


# <------------MUL------------>
class Mul(Operation):
  '''Element wise multiplication between two Tensors or Tensor-like
  '''
  def forward(self, tens1, tens2):
    '''Calculates element wise multiplication

    Args:
      tens1 (Tensor or int or float or list or np.ndarray): First operand
      tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
    Returns:
      Tensor of the result
    '''
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data*tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    '''Sets grad_fn of operands

    Local gradient for each Tensor is the other Tensor's data, which is element-wise
    multiplied with upper gradient

    Args:
      tens1 (Tensor): First operand
      tens2 (Tensor): Second operand
    '''
    tens1.set_grad_fn(lambda ug:tens2.data*ug)
    tens2.set_grad_fn(lambda ug:tens1.data*ug)

def mul(tens1, tens2):
  '''Abstraction for Mul.forward

  Args:
    tens1 (Tensor or int or float or list or np.ndarray): First operand
    tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
  Returns:
    Tensor of the result
  '''
  return Mul().forward(tens1, tens2)


# <------------DIV------------>
class Div(Operation):
  '''Element wise division between two Tensors or Tensor-like
  '''
  def forward(self, tens1, tens2):
    '''Calculates element wise division

    Args:
      tens1 (Tensor or int or float or list or np.ndarray): First operand
      tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
    Returns:
      Tensor of the result
    '''
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data/tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    '''Sets grad_fn of operands

    Local gradient of tens1 is 1/tens2.data, local gradient of tens2 is
    -1*tens1.data/tens2.data^2, which is element wise multiplied with upper
    gradient

    Args:
      tens1 (Tensor): First operand
      tens2 (Tensor): Second operand
    '''
    tens1.set_grad_fn(lambda ug:(1/tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:((-1*tens1.data)/np.power(tens2.data, 2))*ug)

def div(tens1, tens2):
  '''Abstraction for Div.forward

  Args:
    tens1 (Tensor or int or float or list or np.ndarray): First operand
    tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
  Returns:
    Tensor of the result
  '''
  return Div().forward(tens1, tens2)


# <------------DOT------------>
class Dot(Operation):
  '''Dot product between two Tensors or Tensor-like
  '''
  def forward(self, tens1, tens2):
    '''Calculates dot product

    Args:
      tens1 (Tensor or int or float or list or np.ndarray): First operand
      tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
    Returns:
      Tensor of the result
    '''
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.dot(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    '''Sets grad_fn of operands

    Local gradient of tens1 is transpose of tens2.data, local gradient of tens2 is
    transpose of tens1.data, which is dotted with upper gradient

    Args:
      tens1 (Tensor): First operand
      tens2 (Tensor): Second operand
    '''
    tens1.set_grad_fn(lambda ug:np.dot(ug, tens2.data.T))
    tens2.set_grad_fn(lambda ug:np.dot(tens1.data.T, ug))

def dot(tens1, tens2):
  '''Abstraction for Dot.forward

  Args:
    tens1 (Tensor or int or float or list or np.ndarray): First operand
    tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
  Returns:
    Tensor of the result
  '''
  return Dot().forward(tens1, tens2)


# <------------EXP------------>
class Exp(Operation):
  '''Exponentiates the Tensor or Tensor-like
  '''
  def forward(self, tens):
    '''Calculates exponentiation

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.exp(tens.data), tens)
  
  def backward(self, tens):
    '''Sets grad_fn of operand

    Local gradient is exponentiation of tens.data itself

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    '''
    tens.set_grad_fn(lambda ug:np.exp(tens.data)*ug)

def exp(tens):
  '''Abstraction for Exp.forward

  Args:
    tens (Tensor): Operand
  
  Returns:
    Tensor of the result
  '''
  return Exp().forward(tens)


# <------------LOG------------>
class Log(Operation):
  '''Natural Logarithm of the Tensor or Tensor-like
  '''
  def forward(self, tens):
    '''Calculates natural logarithm

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.log(tens.data), tens)
  
  def backward(self, tens):
    '''Sets grad_fn of operand

    Local gradient is exponentiation of 1/tens.data itself

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    '''
    tens.set_grad_fn(lambda ug:(1/tens.data)*ug)

def log(tens):
  '''Abstraction for Log.forward

  Args:
    tens (Tensor): Operand
  
  Returns:
    Tensor of the result
  '''
  return Log().forward(tens)


# <------------POW------------>
class Pow(Operation):
  '''Raises one Tensor or Tensor-like to the power of another Tensor or Tensor-like
  '''
  def forward(self, tens1, tens2):
    '''Calculates raising to a power

    Args:
      tens1 (Tensor or int or float or list or np.ndarray): First operand
      tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
    Returns:
      Tensor of the result
    '''
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.power(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    '''Sets grad_fn of operands

    Local gradient of tens1 is tens1.data^(tens2.data-1), local gradient of tens2 is
    log(tens1.data), which is element wise multiplied with upper gradient

    Args:
      tens1 (Tensor): First operand
      tens2 (Tensor): Second operand
    '''
    result = np.power(tens1.data, tens2.data)
    tens1.set_grad_fn(lambda ug:(np.power(tens1.data, tens2.data-1) * tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:(result*np.log(tens1.data))*ug)

def pow(tens1, tens2):
  '''Abstraction for Pow.forward

  Args:
    tens1 (Tensor or int or float or list or np.ndarray): First operand
    tens2 (Tensor or int or float or list or np.ndarray): Second operand
    
  Returns:
    Tensor of the result
  '''
  return Pow().forward(tens1, tens2)


# <------------SUM------------>
class Sum(Operation):
  '''Performs sum along a specified axis

  If axis is None, then the sum of the entire Tensor is calculated

  Parameters:
    axis (None or int or tuple of int): Axis along which it should be summed
  '''
  def __init__(self, axis=None):
    '''
    Args:
      axis (None or int or tuple of int): Axis along which it should be summed
        Defaults to None
    '''
    self.axis = axis
  
  def forward(self, tens):
    '''Calculates sum along an axis

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.sum(tens.data, axis=self.axis), tens)
  
  def backward(self, tens):
    '''Sets grad_fn of operand

    Local gradient is all ones and the upper gradient must be added a new axis
    along the axis attribute if axis is not None, for broadcasting of upper_gradient
    as during forward pass the dimension will be reduced along the axis it is summed

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    '''
    def sum_backward(ug):
      if self.axis is not None:
        ug = np.expand_dims(ug, axis=self.axis)
      grads = np.ones(tens.shape)*ug
      return grads
    tens.set_grad_fn(sum_backward)

def sum(tens, axis=None):
  '''Abstraction for Sum.forward

  Args:
    tens (Tensor): Operand
    axis (None or int or tuple of int): Axis along which it should be summed
      Defaults to None
  
  Returns:
    Tensor of the result
  '''
  return Sum(axis).forward(tens)


# <------------TRANSPOSE------------>
class Transpose(Operation):
  '''Performs transpose of Tensor or Tensor-like
  '''
  def forward(self, tens):
    '''Performs transpose

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.T, tens)

  def backward(self, tens):
    '''Sets grad_fn of operand

    No local gradient, upper gradient is just transposed

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    '''
    tens.set_grad_fn(lambda ug:ug.T)

def transpose(tens):
  '''Abstraction for Transpose.forward

  Args:
    tens (Tensor): Operand
  
  Returns:
    Tensor of the result
  '''
  return Transpose().forward(tens)


# <------------FLATTEN------------>
class Flatten(Operation):
  '''Performs flattening of Tensor or Tensor-like
  '''
  def forward(self, tens):
    '''Performs flattening from any dimension to 1D

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    flattened = tens.data.flatten()
    return self.get_result_tensor(flattened.reshape(flattened.shape[0],1), tens)
  
  def backward(self, tens):
    '''Sets grad_fn of operand

    No local gradient, upper gradient is reshaped to original shape

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    '''
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def flatten(tens):
  '''Abstraction for Flatten.forward

  Args:
    tens (Tensor): Operand
  
  Returns:
    Tensor of the result
  '''
  return Flatten().forward(tens)


# <------------RESHAPE------------>
class Reshape(Operation):
  '''Performs reshaping of Tensor or Tensor-like
  '''
  def forward(self, tens, new_shape):
    '''Performs reshaping of Tensor to a new shape

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
      new_shape (tuple): New shape to be reshaped into
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.reshape(new_shape), tens)
  
  def backward(self, tens):
    '''Sets grad_fn of operand

    No local gradient, upper gradient is reshaped to original shape

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    '''
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def reshape(tens, new_shape):
  '''Abstraction for Reshape.forward

  Args:
    tens (Tensor): Operand
  
  Returns:
    Tensor of the result
  '''
  return Reshape().forward(tens, new_shape)