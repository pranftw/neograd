import numpy as np
from .operation import Operation


# <------------RELU------------>

class ReLU(Operation):
  '''Performs Rectified Linear Unit
  '''
  def forward(self, tens):
    '''Only allows positive values to flow through

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.maximum(0, tens.data), tens)
  
  def backward(self, tens):
    '''Sets the grad_fn of the Tensor

    If element in data is greater than zero, its local gradient will be 1
    else 0

    Args:
      tens (Tensor): Operand
    '''
    tens.set_grad_fn(lambda ug:np.where(tens.data>=0, 1, 0)*ug)

def relu(tens):
  '''Abstraction for ReLU.forward

  Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
  Returns:
    Tensor of the result
  '''
  return ReLU().forward(tens)


# <------------SIGMOID------------>

class Sigmoid(Operation):
  '''Performs Sigmoid
  '''
  def forward(self, tens):
    '''Squishes the Tensor to value between 0 and 1

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(1/(1+np.exp(-tens.data)), tens)
  
  def backward(self, tens):
    '''Sets the grad_fn of the Tensor

    Args:
      tens (Tensor): Operand
    '''
    result = 1/(1+np.exp(-tens.data))
    tens.set_grad_fn(lambda ug:(result*(1-result))*ug)

def sigmoid(tens):
  '''Abstraction for Sigmoid.forward

  Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
  Returns:
    Tensor of the result
  '''
  return Sigmoid().forward(tens)


# <------------TANH------------>

class Tanh(Operation):
  '''Performs Hyperbolic Tangent
  '''
  def forward(self, tens):
    '''Squishes the Tensor to value between -1 and 1

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.tanh(tens.data), tens)
  
  def backward(self, tens):
    '''Sets the grad_fn of the Tensor

    Args:
      tens (Tensor): Operand
    '''
    result = np.tanh(tens.data)
    tens.set_grad_fn(lambda ug:(1-np.power(result,2))*ug)

def tanh(tens):
  '''Abstraction for Tanh.forward

  Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
  Returns:
    Tensor of the result
  '''
  return Tanh().forward(tens)


# <------------SOFTMAX------------>

class Softmax(Operation):
  '''Performs Softmax

  Attributes:
    axis (None or int or tuple of int): Axis along which it should be calculated
  '''
  def __init__(self, axis):
    '''
    Args:
      axis (None or int or tuple of int): Axis along which it should be calculated
    '''
    self.axis = axis

  def forward(self, tens):
    '''Maps the Tensor to probabilities, whose sum is 1

    Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
    Returns:
      Tensor of the result
    '''
    tens = self.get_tensors(tens)
    result = self.calc_softmax(tens.data, axis=self.axis)
    return self.get_result_tensor(result, tens)
  
  def backward(self, tens):
    '''Sets the grad_fn of the Tensor

    Quite a tricky one, first the Jacobian of each of the slices along
    the specified axis of the result is taken, which is then dotted with the
    corresponding slice of the upper gradient

    Args:
      tens (Tensor): Operand
    '''
    def softmax_grad(arr, ug_slices): # arr will always be 1d array
      local_grad = -np.broadcast_to(arr, (arr.size, arr.size))
      np.fill_diagonal(local_grad, 1+np.diagonal(local_grad))
      local_grad = local_grad*arr.reshape(arr.size, 1)
      result = np.dot(local_grad, ug_slices.pop(0))
      return result
    
    def get_ug_slices(arr, ug_slices):
      ug_slices.append(arr)

    def grad_backward(ug):
      result = np.apply_along_axis(self.calc_softmax, self.axis, tens.data)
      ug_slices = []
      np.apply_along_axis(get_ug_slices, self.axis, ug, ug_slices)
      grads = np.apply_along_axis(softmax_grad, self.axis, result, ug_slices)
      return grads

    tens.set_grad_fn(grad_backward)

  @staticmethod
  def calc_softmax(arr, axis=None):
    '''Calculates stable Softmax

    Args:
      arr (np.ndarray): Array whose Softmax is to be calculated
      axis (int or tuple of int): Axis along which to calculate the Softmax
        Defaults to None
    
    Returns:
      Softmax of the array
    '''
    exponentiated = np.exp(arr-np.max(arr, axis=axis, keepdims=True))
    sum_val = np.sum(exponentiated, axis=axis, keepdims=True)
    return exponentiated/sum_val

def softmax(tens, axis):
  '''Abstraction for Softmax.forward

  Args:
      tens (Tensor or int or float or list or np.ndarray): Operand
    
  Returns:
    Tensor of the result
  '''
  return Softmax(axis).forward(tens)