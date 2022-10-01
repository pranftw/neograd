import numpy as np
from ..autograd import tensor, dot
from ..autograd.ops import conv2d, conv3d, maxpool2d, maxpool3d


class Container:
  '''
    This acts as a container for Layer/s
  '''
  def __init__(self):
    self.layers = None
    self.frozen = False

  def __call__(self, inputs):
    return self.forward(inputs)

  def get_params(self, as_dict, return_frozen):
    '''
      Goes through all the layers in the Container and gets the params of each Layer
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
    for layer in self.layers:
      layer.set_eval(eval)
  
  def set_params(self, container_params):
    for layer_param, layer in zip(container_params, self.layers):
      layer.set_params(layer_param)
  
  def freeze(self):
    self.frozen = True
  
  def unfreeze(self):
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
    Performs some kind of computation taking in inputs and giving out outputs
  '''
  def __init__(self):
    self.eval = False
    self.frozen = False

  def __call__(self, inputs):
    return self.forward(inputs)

  def get_params(self, as_dict, return_frozen):
    '''
      If any of the attributes in a Layer is instance of Param, then it is automatically
        considered as a param for the whole model
    '''
    params = {}
    for attr in dir(self):
      val = self.__getattribute__(attr)
      if isinstance(val, Param):
        if return_frozen or not(val.frozen):
          params[attr] = val.data if as_dict else val
    return params if as_dict else params.values()
  
  def set_eval(self, eval):
    self.eval = eval
  
  def set_params(self, layer_params):
    for attr, param_data in layer_params.items():
      param = self.__getattribute__(attr)
      param.data = param_data
  
  def freeze(self):
    self.frozen = True
  
  def unfreeze(self):
    self.frozen = False
  
  def __setattr__(self, attr, val):
    if (isinstance(val, Param)) and (attr in self.__dict__):
      raise AttributeError(f"Attribute {attr} has already been defined, it cannot be defined again for a Param")
    object.__setattr__(self, attr, val)


class Param(tensor):
  '''
    Just an alias for Tensor, so that when params are gathered for a Layer, only these
      are automatically considered for param, while ignoring some helper Tensors which aren't
      necessarily param
  '''

  def __init__(self, data, requires_grad=False, requires_broadcasting=True):
    super().__init__(data, requires_grad, requires_broadcasting)
    self.frozen = False
  
  def freeze(self):
    self.frozen = True
  
  def unfreeze(self):
    self.frozen = False
  
  def __str__(self):
    return f'Param({super().__str__()})'
  
  def __repr__(self):
    return f'Param({super().__repr__()})'


class Sequential(Container):
  '''
    Outputs of one layer are passed as inputs to the next layer, sequentially
  '''
  
  def __init__(self, *args):
    super().__init__()
    self.layers = args
  
  def forward(self, inputs):
    for layer in self.layers:
      output = layer(inputs)
      inputs = output
    return output
  
  def __str__(self):
    return f'Sequential(\n{super().__str__()}\n)'
  
  def __repr__(self):
    return f'Sequential(\n{super().__repr__()}\n)'


class Linear(Layer):
  __slots__ = ['num_in', 'num_out', 'weights', 'bias']
  '''
    Implements a fully connected layer
  '''

  def __init__(self, num_in, num_out):
    super().__init__()
    self.num_in = num_in
    self.num_out = num_out
    self.weights = Param(np.random.randn(num_in, num_out), requires_grad=True)
    self.bias = Param(np.zeros((1, num_out)), requires_grad=True)
  
  def forward(self, inputs):
    return dot(inputs, self.weights) + self.bias
  
  def __repr__(self):
    return f'Linear({self.num_in}, {self.num_out})'
  
  def __str__(self):
    return f'Linear in:{self.num_in} out:{self.num_out}'


class Dropout(Layer):
  '''
    Dropout
    https://youtu.be/D8PJAL-MZv8
  '''

  def __init__(self, prob):
    super().__init__()
    self.prob = prob
  
  def forward(self, inputs):
    if not(self.eval):
      inputs.data = (inputs.data * np.where(np.random.randn(*(inputs.shape))<self.prob, 1, 0)) / self.prob
    return inputs
  
  def __repr__(self):
    return f'Dropout(prob={self.prob})'
  
  def __str__(self):
    return f'Dropout(prob={self.prob})'


class Conv2D(Layer):
  '''
    Conv2D
  '''

  def __init__(self, kernel_shape, padding=0, stride=1):
    super().__init__()
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel = Param(np.random.randn(*kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(0, requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    return conv2d(inputs, self.kernel, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    return f'Conv2D(kernel_shape={self.kernel.shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'Conv2D(kernel_shape={self.kernel.shape}, padding={self.padding}, stride={self.stride})'


class Conv3D(Layer):
  '''
    Conv3D
  '''

  def __init__(self, in_channels, out_channels, kernel_shape, padding=0, stride=1):
    super().__init__()
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel = Param(np.random.randn(out_channels, in_channels, *kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(np.zeros(out_channels), requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    return conv3d(inputs, self.kernel, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    kernel_shape = self.kernel.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    kernel_shape = self.kernel.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'


class MaxPool2D(Layer):
  '''
    MaxPool2D
  '''
  
  def __init__(self, kernel_shape, padding=0, stride=1):
    super().__init__()
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel_shape = kernel_shape
  
  def forward(self, inputs):
    return maxpool2d(inputs, self.kernel_shape, self.padding, self.stride)
  
  def __repr__(self):
    return f'MaxPool2D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'MaxPool2D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'


class MaxPool3D(Layer):
  '''
    MaxPool3D
  '''

  def __init__(self, kernel_shape, padding=0, stride=1):
    super().__init__()
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel_shape = kernel_shape
  
  def forward(self, inputs):
    return maxpool3d(inputs, self.kernel_shape, self.padding, self.stride)
  
  def __repr__(self):
    return f'MaxPool3D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'MaxPool3D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'