from .layers import Layer
from ..autograd.ops import relu, sigmoid, tanh, softmax
import numpy as np


class ReLU(Layer):
  '''ReLU Layer
  '''
  def __init__(self):
    super().__init__()

  def forward(self, inputs):
    '''Calculates ReLU of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    return relu(inputs)

  def __repr__(self):
    return 'ReLU()'
  
  def __str__(self):
    return 'ReLU'


class Sigmoid(Layer):
  '''Sigmoid Layer
  '''
  def __init__(self):
    super().__init__()

  def forward(self, inputs):
    '''Calculates Sigmoid of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    return sigmoid(inputs)

  def __repr__(self):
    return 'Sigmoid()'
  
  def __str__(self):
    return 'Sigmoid'


class Tanh(Layer):
  '''Tanh Layer
  '''
  def __init__(self):
    super().__init__()

  def forward(self, inputs):
    '''Calculates Tanh of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    return tanh(inputs)
  
  def __repr__(self):
    return 'Tanh()'
  
  def __str__(self):
    return 'Tanh'


class Softmax(Layer):
  '''Softmax Layer
  '''
  def __init__(self, axis):
    '''
    Args:
      axis (int): Axis along which softmax should be calculated
    '''
    super().__init__()
    self.axis = axis

  def forward(self, inputs):
    '''Calculates Softmax of inputs

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of result
    '''
    return softmax(inputs, self.axis)
  
  def __repr__(self):
    return f'Softmax(axis={self.axis})'
  
  def __str__(self):
    return 'Softmax'
