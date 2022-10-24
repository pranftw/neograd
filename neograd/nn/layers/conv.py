import numpy as np
from ..layers import Layer, Param
from ...autograd.ops import conv2d, conv3d, maxpool2d, maxpool3d


class Conv2D(Layer):
  '''Implements Conv2D

  Parameters:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    weights (Param): Kernel for the Convolution
    bias (Param): Bias for the Convolution
  
  Raises:
    ValueError: If kernel_shape isn't 2D tuple
  '''
  def __init__(self, kernel_shape, padding=0, stride=1):
    '''
    Args:
      kernel_shape (tuple of int): Shape of the kernel
    '''
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.weights = Param(np.random.randn(*kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(0, requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    '''Forward pass of Conv2D

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    return conv2d(inputs, self.weights, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    return f'Conv2D(kernel_shape={self.weights.shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'Conv2D(kernel_shape={self.weights.shape}, padding={self.padding}, stride={self.stride})'


class Conv3D(Layer):
  '''Implements Conv3D

  Parameters:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    weights (Param): Kernel for the Convolution
    bias (Param): Bias for the Convolution
  
  Raises:
    ValueError: If kernel_shape isn't 2D tuple
  '''
  def __init__(self, in_channels, out_channels, kernel_shape, padding=0, stride=1):
    '''
    Args:
      in_channels (int): Number of channels in the inputs
      out_channels (int): Number of channels in the outputs
      kernel_shape (tuple of int): Shape of the kernel
    '''
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.weights = Param(np.random.randn(out_channels, in_channels, *kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(np.zeros(out_channels), requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    '''Forward pass of Conv3D

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    return conv3d(inputs, self.weights, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    kernel_shape = self.weights.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    kernel_shape = self.weights.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'


class MaxPool2D(Layer):
  '''Implements MaxPool2D

  Parameters:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    kernel_shape (tuple of int): Shape of the kernel
  
  Raises:
    ValueError: If kernel_shape isn't 2D tuple
  '''
  def __init__(self, kernel_shape, padding=0, stride=1):
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel_shape = kernel_shape
  
  def forward(self, inputs):
    '''Forward pass of MaxPool2D

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    return maxpool2d(inputs, self.kernel_shape, self.padding, self.stride)
  
  def __repr__(self):
    return f'MaxPool2D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'MaxPool2D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'


class MaxPool3D(Layer):
  '''Implements MaxPool2D

  Parameters:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    kernel_shape (tuple of int): Shape of the kernel
  
  Raises:
    ValueError: If kernel_shape isn't 2D tuple
  '''
  def __init__(self, kernel_shape, padding=0, stride=1):
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel_shape = kernel_shape
  
  def forward(self, inputs):
    '''Forward pass of MaxPool3D

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    return maxpool3d(inputs, self.kernel_shape, self.padding, self.stride)
  
  def __repr__(self):
    return f'MaxPool3D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'MaxPool3D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'