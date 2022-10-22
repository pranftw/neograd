from .layers import Layer
from ..autograd.ops.operation import Operation
import numpy as np


# <------------RELU------------>
class ReLU(Layer, Operation):
  '''ReLU Layer
  '''
  def forward(self, inputs):
    '''Calculates ReLU of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    inputs = self.get_tensors(inputs)
    return self.get_result_tensor(np.maximum(0, inputs.data), inputs)
  
  def backward(self, inputs):
    '''Sets the grad_fn of the Tensor

    If element in data is greater than zero, its local gradient will be 1
    else 0

    Args:
      inputs (Tensor): Operand
    '''
    inputs.set_grad_fn(lambda ug:np.where(inputs.data>=0, 1, 0)*ug)

  def __repr__(self):
    return 'ReLU()'
  
  def __str__(self):
    return 'ReLU'


# <------------SIGMOID------------>
class Sigmoid(Layer, Operation):
  '''Sigmoid Layer
  '''
  def forward(self, inputs):
    '''Calculates Sigmoid of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    inputs = self.get_tensors(inputs)
    return self.get_result_tensor(1/(1+np.exp(-inputs.data)), inputs)
  
  def backward(self, inputs):
    '''Sets the grad_fn of the Tensor

    Args:
      inputs (Tensor): Operand
    '''
    result = 1/(1+np.exp(-inputs.data))
    inputs.set_grad_fn(lambda ug:(result*(1-result))*ug)

  def __repr__(self):
    return 'Sigmoid()'
  
  def __str__(self):
    return 'Sigmoid'


# <------------TANH------------>
class Tanh(Layer, Operation):
  '''Tanh Layer
  '''
  def forward(self, inputs):
    '''Calculates Tanh of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    inputs = self.get_tensors(inputs)
    return self.get_result_tensor(np.tanh(inputs.data), inputs)
  
  def backward(self, inputs):
    '''Sets the grad_fn of the Tensor

    Args:
      inputs (Tensor): Operand
    '''
    result = np.tanh(inputs.data)
    inputs.set_grad_fn(lambda ug:(1-np.power(result,2))*ug)
  
  def __repr__(self):
    return 'Tanh()'
  
  def __str__(self):
    return 'Tanh'


# <------------SOFTMAX------------>
class Softmax(Layer, Operation):
  '''Softmax Layer

  Parameters:
    axis (None or int or tuple of int): Axis along which it should be calculated
  '''
  def __init__(self, axis):
    '''
    Args:
      axis (None or int or tuple of int): Axis along which it should be calculated
    '''
    self.axis = axis

  def forward(self, inputs):
    '''Calculates Softmax of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    inputs = self.get_tensors(inputs)
    result = self.calc_softmax(inputs.data, axis=self.axis)
    return self.get_result_tensor(result, inputs)
  
  def backward(self, inputs):
    '''Sets the grad_fn of the Tensor

    Quite a tricky one, first the Jacobian of each of the slices along
    the specified axis of the result is taken, which is then dotted with the
    corresponding slice of the upper gradient

    Args:
      inputs (Tensor): Operand
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
      result = np.apply_along_axis(self.calc_softmax, self.axis, inputs.data)
      ug_slices = []
      np.apply_along_axis(get_ug_slices, self.axis, ug, ug_slices)
      grads = np.apply_along_axis(softmax_grad, self.axis, result, ug_slices)
      return grads

    inputs.set_grad_fn(grad_backward)
  
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
  
  def __repr__(self):
    return f'Softmax(axis={self.axis})'
  
  def __str__(self):
    return 'Softmax'


# <------------LEAKYRELU------------>
class LeakyReLU(Layer, Operation):
  '''LeakyReLU Layer
  '''
  def __init__(self, leak=0.01):
    '''
    Args:
      leak (float): leak value
    '''
    self.leak = leak

  def forward(self, inputs):
    '''Calculates LeakyReLU of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    inputs = self.get_tensors(inputs)
    arr = inputs.data
    return self.get_result_tensor(np.where(arr>=0, arr, self.leak*arr), inputs)
  
  def backward(self, inputs):
    '''Sets the grad_fn of the Tensor

    If element in data is greater than zero, its local gradient will be 1
    else will be leak value

    Args:
      inputs (Tensor): Operand
    '''
    inputs.set_grad_fn(lambda ug: np.where(inputs.data>=0, 1, self.leak)*ug)

  def __repr__(self):
    return f'LeakyReLU(leak={self.leak})'
  
  def __str__(self):
    return 'LeakyReLU'