from ..autograd import tensor, dot
import numpy as np


class Container:
  __slots__ = ['layers']
  '''
    This acts as a container for Layer/s
  '''

  def __call__(self, inputs):
    return self.forward(inputs)

  def get_params(self):
    '''
      Goes through all the layers in the Container and gets the params of each Layer
    '''
    params = []
    for layer in self.layers:
      params+=layer.get_params()
    return params
  
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
  __slots__ = []
  '''
    Performs some kind of computation taking in inputs and giving out outputs
  '''

  def __call__(self, inputs):
    return self.forward(inputs)

  def get_params(self):
    '''
      If any of the attributes in a Layer is instance of Param, then it is automatically
        considered as a param for the whole model
    '''
    params = []
    for attr in dir(self):
      attr_val = self.__getattribute__(attr)
      if isinstance(attr_val, Param):
        params.append(attr_val)
    return params


class Param(tensor):
  __slots__ = []
  '''
    Just an alias for Tensor, so that when params are gathered for a Layer, only these
      are automatically considered for param, while ignoring some helper Tensors which aren't
      necessarily param
  '''

  def __init__(self, data, requires_grad=False):
    super().__init__(data, requires_grad)
  
  def __str__(self):
    return f'Param({super().__str__()})'
  
  def __repr__(self):
    return f'Param({super().__repr__()})'


class Sequential(Container):
  __slots__ = ['layers']
  '''
    Outputs of one layer are passed as inputs to the next layer, sequentially
  '''
  
  def __init__(self, *args):
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
    self.num_in = num_in
    self.num_out = num_out
    self.weights = Param(np.random.randn(num_out, num_in), requires_grad=True)
    self.bias = Param(np.zeros((num_out, 1)), requires_grad=True)
  
  def forward(self, inputs):
    return dot(self.weights, inputs) + self.bias
  
  def __repr__(self):
    return f'Linear({self.num_in}, {self.num_out})'
  
  def __str__(self):
    return f'Linear in:{self.num_in} out:{self.num_out}'