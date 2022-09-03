from ..autograd import tensor


class Container:
  def __call__(self, inputs):
    return self.forward(inputs)

  def get_params(self):
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
  def __call__(self, inputs):
    return self.forward(inputs)

  def get_params(self):
    params = []
    for attr in dir(self):
      attr_val = self.__getattribute__(attr)
      if isinstance(attr_val.__class__, Param):
        params.append(attr_val.tens)
    return params


class Param:
  def __init__(self, tens):
    self.tens = tens
  
  def __str__(self):
    return f'Param({self.tens.__str__()})'
  
  def __repr__(self):
    return f'Param({self.tens.__repr__()})'


class Sequential(Container):
  def __init__(self, *args):
    self.layers = args
  
  def forward(self, inputs):
    for layer in self.layers:
      output = layer.forward(inputs)
      input = output
    return output
  
  def __str__(self):
    return f'Sequential(\n{super().__str__()}\n)'
  
  def __repr__(self):
    return f'Sequential(\n{super().__repr__()}\n)'


class Linear(Layer):
  def __init__(self, num_in, num_out):
    self.num_in = num_in
    self.num_out = num_out
    self.weights = Param(tensor(np.random.randn(num_in, num_out), requires_grad=True))
    self.bias = Param(tensor(np.zeros((num_out, 1)), requires_grad=True))
  
  def forward(self, inputs):
    output = Dot(self.weights.T, inputs).forward() + self.bias
    return output
  
  def __repr__(self):
    return f'Linear({self.num_in}, {self.num_out})'
  
  def __str__(self):
    return f'Linear in:{self.num_in} out:{self.num_out}'