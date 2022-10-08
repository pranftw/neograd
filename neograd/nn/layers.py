import numpy as np
from ..autograd import tensor, dot
from ..autograd.ops import conv2d, conv3d, maxpool2d, maxpool3d


class Container:
  '''Contains many Layers

  Attributes:
    eval (bool): Whether the Container is in eval mode
    layers (list of Layer/Container): Layer to be included in the container
    frozen (bool): Whether the layer is frozen or not. If frozen, then all of its parameters
      won't be updated during optimization, but allows the gradients to flow through
  '''
  def __init__(self):
    self.eval = False
    self.layers = None
    self.frozen = False

  def __call__(self, inputs):
    '''Calls the forward method

    Args:
      inputs (Tensor): Inputs to the container
    
    Returns:
      Tensor of the result
    '''
    return self.forward(inputs)

  def get_params(self, as_dict, return_frozen):
    '''Recursively goes through all the layers in the Container and gets the params of each Layer

    If return_frozen is True or layer is not frozen, then the params are taken from
    that layer

    Args:
      as_dict (bool): Whether params need to be returned as a dict
      return_frozen (bool): Whether frozen layers' params need to be returned
    
    Returns:
      list of Params
    '''
    params = []
    for layer in self.layers:
      if return_frozen or not(layer.frozen):
        if as_dict:
          params.append(layer.get_params(as_dict, return_frozen))
        else:
          params+=layer.get_params(as_dict, return_frozen)
    return params
  
  def set_eval(self, eval):
    '''Sets eval

    Sets its eval to the eval argument also recursively sets the eval
    of its layers

    Args:
      eval (bool): Whether Container is in eval mode or not
    '''
    self.eval = eval
    for layer in self.layers:
      layer.set_eval(eval)
  
  def set_params(self, container_params):
    '''Sets the params for all the layers
    '''
    for layer_param, layer in zip(container_params, self.layers):
      layer.set_params(layer_param)
  
  def freeze(self):
    '''Sets frozen to True
    '''
    self.frozen = True
  
  def unfreeze(self):
    '''Sets frozen to False
    '''
    self.frozen = False
  
  def __repr__(self):
    layers = []
    for layer in self.layers:
      layers.append(f'{layer.__str__()}')
    layers_repr = ', '.join(layers)
    return layers_repr
  
  def __str__(self):
    layers = []
    for layer in self.layers:
      layers.append(f'{layer.__repr__()}')
    layers_str = ', '.join(layers)
    return layers_str


class Layer:
  '''
    Fundamental building block of the model
  
  Attributes:
    eval (bool): Whether the Container is in eval mode
    frozen (bool): Whether the layer is frozen or not. If frozen, then all of its parameters
      won't be updated during optimization, but allows the gradients to flow through
  '''
  def __init__(self):
    self.eval = False
    self.frozen = False

  def __call__(self, inputs):
    '''Calls the forward method

    Args:
      inputs (Tensor): Inputs to the container
    
    Returns:
      Tensor of the result
    '''
    return self.forward(inputs)

  def get_params(self, as_dict, return_frozen):
    '''Returns the parameters in the Layer

    If any of the attributes in a Layer is instance of Param, then it is automatically
    considered as a param for the model

    Args:
      as_dict (bool): Whether params need to be returned as a dict
      return_frozen (bool): Whether frozen layers' params need to be returned
    
    Returns:
      list of Params or dict
    '''
    params = {}
    for attr in dir(self):
      val = self.__getattribute__(attr)
      if isinstance(val, Param):
        if return_frozen or not(val.frozen):
          params[attr] = val.data if as_dict else val
    return params if as_dict else list(params.values())
  
  def set_eval(self, eval):
    '''Sets eval

    Sets its eval to the eval argument

    Args:
      eval (bool): Whether Container is in eval mode or not
    '''
    self.eval = eval
  
  def set_params(self, layer_params):
    '''Sets the params for the current layer
    '''
    for attr, param_data in layer_params.items():
      param = self.__getattribute__(attr)
      param.data = param_data
  
  def freeze(self):
    '''Sets frozen to True
    '''
    self.frozen = True
  
  def unfreeze(self):
    '''Sets frozen to False
    '''
    self.frozen = False
  
  def __setattr__(self, attr, val):
    '''Sets attributes for the Layer

    Doesn't allow modification/re-assigning of Param attributes

    Args:
      attr (str): Attribute to set the value to
      val (object): The value to be set to

    Raises:
      AttributeError: If a Param attr has already been defined/assigned
    '''
    if (isinstance(val, Param)) and (attr in self.__dict__):
      raise AttributeError(f"Attribute {attr} has already been defined, it cannot be defined again for a Param")
    object.__setattr__(self, attr, val)


class Param(tensor):
  '''Alias for Tensor

  Just an alias for Tensor, so that when params are gathered for a Layer, only these
  are automatically considered for param, while ignoring some helper Tensors which aren't
  necessarily param

  Attributes:
    frozen (bool): Whether the param is frozen or not
  '''

  def __init__(self, data, requires_grad=False, requires_broadcasting=True):
    super().__init__(data, requires_grad, requires_broadcasting)
    self.frozen = False
  
  def freeze(self):
    '''Sets frozen to True
    '''
    self.frozen = True
  
  def unfreeze(self):
    '''Sets frozen to False
    '''
    self.frozen = False
  
  def __str__(self):
    return f'Param({super().__str__()})'
  
  def __repr__(self):
    return f'Param({super().__repr__()})'


class Sequential(Container):
  '''Sequential Container
  
  Outputs of one layer are passed as inputs to the next layer, sequentially
  '''
  
  def __init__(self, *args):
    super().__init__()
    self.layers = args
  
  def forward(self, inputs):
    '''Forward pass of Sequential

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    for layer in self.layers:
      output = layer(inputs)
      inputs = output
    return output
  
  def __str__(self):
    return f'Sequential(\n{super().__str__()}\n)'
  
  def __repr__(self):
    return f'Sequential(\n{super().__repr__()}\n)'


class Linear(Layer):
  '''Implements a fully connected Layer

  Parameters:
    num_in (int): Number of inputs to the Layer
    num_out (int): Number of outputs from the Layer
    weights (Param): Weights of the Layer
    bias (Param): Bias of the Layer
  '''
  __slots__ = ['num_in', 'num_out', 'weights', 'bias']

  def __init__(self, num_in, num_out):
    super().__init__()
    self.num_in = num_in
    self.num_out = num_out
    self.weights = Param(np.random.randn(num_in, num_out), requires_grad=True)
    self.bias = Param(np.zeros((1, num_out)), requires_grad=True)
  
  def forward(self, inputs):
    '''Forward pass of Linear

    The inputs are dotted with weights and then bias is added

    Args:
      inputs (Tensor): Inputs to the Linear
    
    Returns:
      Tensor of the result
    '''
    return dot(inputs, self.weights) + self.bias
  
  def __repr__(self):
    return f'Linear({self.num_in}, {self.num_out})'
  
  def __str__(self):
    return f'Linear in:{self.num_in} out:{self.num_out}'


class Dropout(Layer):
  '''Dropout Layer
  
  https://youtu.be/D8PJAL-MZv8
  
  Attributes:
    prob (float): Probability with which to turn off the inputs
  '''
  def __init__(self, prob):
    super().__init__()
    self.prob = prob
  
  def forward(self, inputs):
    '''Forward pass of Dropout

    The inputs are turned off with the given prob

    If in eval mode, then all inputs are always on

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    if not(self.eval):
      inputs.data = (inputs.data * np.where(np.random.randn(*(inputs.shape))<self.prob, 1, 0)) / self.prob
    return inputs
  
  def __repr__(self):
    return f'Dropout(prob={self.prob})'
  
  def __str__(self):
    return f'Dropout(prob={self.prob})'


class Conv2D(Layer):
  '''Implements Conv2D

  Attributes:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    kernel (Param): Kernel for the Convolution
    bias (Param): Bias for the Convolution
  
  Raises:
    ValueError: If kernel_shape isn't 2D tuple
  '''
  def __init__(self, kernel_shape, padding=0, stride=1):
    '''
    Args:
      kernel_shape (tuple of int): Shape of the kernel
    '''
    super().__init__()
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel = Param(np.random.randn(*kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(0, requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    '''Forward pass of Conv2D

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    return conv2d(inputs, self.kernel, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    return f'Conv2D(kernel_shape={self.kernel.shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'Conv2D(kernel_shape={self.kernel.shape}, padding={self.padding}, stride={self.stride})'


class Conv3D(Layer):
  '''Implements Conv3D

  Attributes:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    kernel (Param): Kernel for the Convolution
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
    super().__init__()
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel = Param(np.random.randn(out_channels, in_channels, *kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(np.zeros(out_channels), requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    '''Forward pass of Conv3D

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    return conv3d(inputs, self.kernel, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    kernel_shape = self.kernel.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    kernel_shape = self.kernel.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'


class MaxPool2D(Layer):
  '''Implements MaxPool2D

  Attributes:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    kernel_shape (tuple of int): Shape of the kernel
  
  Raises:
    ValueError: If kernel_shape isn't 2D tuple
  '''
  def __init__(self, kernel_shape, padding=0, stride=1):
    super().__init__()
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

  Attributes:
    padding (int): Padding value to be applied. Defaults to 0
    stride (int): Stride to be taken. Defaults to 1
    kernel_shape (tuple of int): Shape of the kernel
  
  Raises:
    ValueError: If kernel_shape isn't 2D tuple
  '''
  def __init__(self, kernel_shape, padding=0, stride=1):
    super().__init__()
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