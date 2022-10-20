import numpy as np
from copy import deepcopy
from ..autograd import tensor, dot
from ..autograd.ops import conv2d, conv3d, maxpool2d, maxpool3d
from ..autograd.ops.operation import Operation


class Container:
  '''Contains many Layers

  Parameters:
    eval (bool): Whether the Container is in eval mode
    layers (list of Layer/Container): Layer to be included in the container
  '''
  def __init__(self):
    self.eval = False
    self.layers = None

  def __call__(self, inputs):
    '''Calls the forward method

    Args:
      inputs (Tensor): Inputs to the container
    
    Returns:
      Tensor of the result
    '''
    return self.forward(inputs)

  def parameters(self, as_dict=False):
    '''Recursively goes through all the layers in the Container and gets the params of each Layer

    Args:
      as_dict (bool): Whether params need to be returned as a dict,
        Defaults to False
    
    Returns:
      list of Params
    '''
    params = []
    for layer in self.layers:
      if as_dict:
        params.append(layer.parameters(as_dict))
      else:
        params+=layer.parameters(as_dict)
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
    '''Freezes all the layers present in the Container
    '''
    for layer in self.layers:
      layer.freeze()
  
  def unfreeze(self):
    '''Unfreezes all the layers present in the Container
    '''
    for layer in self.layers:
      layer.unfreeze()
  
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
  
  Parameters:
    eval (bool): Whether the Container is in eval mode
  '''
  def __init__(self):
    self.eval = False

  def __call__(self, inputs):
    '''Calls the forward method

    Args:
      inputs (Tensor): Inputs to the container
    
    Returns:
      Tensor of the result
    '''
    return self.forward(inputs)

  def parameters(self, as_dict=False):
    '''Returns the parameters in the Layer

    If any of the attributes in a Layer is instance of Param, then it is automatically
    considered as a param for the model

    Args:
      as_dict (bool): Whether params need to be returned as a dict,
        Defaults to False
    
    Returns:
      list of Params or dict
    '''
    params = {}
    for attr, val in self.__dict__.items():
      if isinstance(val, Param):
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
    '''Freezes all the Params in the Layer
    '''
    for param in self.parameters(as_dict=False):
      param.freeze()
  
  def unfreeze(self):
    '''Unfreezes all the Params in the Layer
    '''
    for param in self.parameters(as_dict=False):
      param.unfreeze()
  
  def __getstate__(self):
    '''Returns the state for the object that is to be pickled

    The instances of Param data are set to 0, to prevent saving of all the params
    in the Layer.
    If required, then the weights can be saved and loaded separately

    Returns:
      state of the current Layer
    '''
    state = deepcopy(self.__dict__)
    for param_attr in self.parameters(as_dict=True).keys():
      state[param_attr].data = 0 # Wanted to set it to None, but it isnt supported by Tensor, so set it to the next best 0
    return state
  
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

  Parameters:
    __frozen (bool): Whether current Param is frozen or not. This is required because we
      need to know if it has been frozen before unfreeze is called
  '''

  def __init__(self, data, requires_grad=False, requires_broadcasting=True):
    super().__init__(data, requires_grad, requires_broadcasting)
    self.__frozen = False
  
  def freeze(self):
    '''Sets requires_grad=False
    '''
    self.requires_grad = False
    self.__frozen = True
  
  def unfreeze(self):
    '''Sets requires_grad=True only if its frozen

    frozen condition is checked because only if it was previously frozen, we can set requires_grad
    = True, if we don't check for it and requires_grad is False originally, then we might set it to
    True, which would be incorrect
    '''
    if self.__frozen:
      self.requires_grad = True
    self.__frozen = False
  
  def __str__(self):
    return f'Param({super().__str__()})'
  
  def __repr__(self):
    return f'Param({super().__repr__()})'


class Sequential(Container):
  '''Sequential Container
  
  Outputs of one layer are passed as inputs to the next layer, sequentially
  '''
  
  def __init__(self, *args):
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
  def __init__(self, num_in, num_out):
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


class Dropout(Layer, Operation):
  '''Dropout Layer
  
  https://youtu.be/D8PJAL-MZv8
  
  Parameters:
    prob (float): Probability with which to keep the inputs
      With probability=prob, the units are kept and with probability=1-prob,
      they are shut off
  '''
  def __init__(self, prob):
    Layer.__init__(self)
    assert prob>0 and prob<=1, 'Probability should be between 0 and 1'
    self.prob = prob
  
  def forward(self, inputs):
    '''Forward pass of Dropout

    The inputs are turned on with the given prob
    If in eval mode, then all inputs are always on

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    if self.eval:
      filter = np.ones(inputs.shape) # dont discard anything, just dummy because if eval or not, backward needs to have filter arg
    else:
      filter = np.where(np.random.random(inputs.shape)<self.prob, 1, 0)
    inputs, filter = self.get_tensors(inputs, filter)
    if not(self.eval): # Deliberately repeated condition check for eval, because if in eval, it shouldnt be scaled by prob
      result = (inputs.data*filter.data)/self.prob
    else:
      result = inputs.data
    return self.get_result_tensor(inputs.data, inputs, filter)
  
  def backward(self, inputs, filter):
    '''Sets the grad_fn of inputs only because filter doesnt have requires_grad=True

    Just multiplies the upper gradient with the filter set during forward pass and scales
    it by the probability

    Args:
      inputs (Tensor): Inputs to the Layer
      filter (Tensor): The dropout filter that was applied during forward pass
    '''
    if not(self.eval):
      inputs.set_grad_fn(lambda ug:(ug*filter.data)/self.prob)
    inputs.set_grad_fn(lambda ug:ug)
  
  def __repr__(self):
    return f'Dropout(prob={self.prob})'
  
  def __str__(self):
    return f'Dropout(prob={self.prob})'


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