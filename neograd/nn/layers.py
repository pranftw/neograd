from ..autograd import tensor


class Container:
  def __call__(self, inputs):
    return self.forward(inputs)

  def get_params(self):
    params = []
    for layer in self.layers:
      params+=layer.get_params()
    return params


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
    return f'{self.tens}'
  
  def __repr__(self):
    return f'Param({self.tens})'


class Sequential(Container):
  def __init__(self, *args):
    self.layers = args
  
  def forward(self, inputs):
    for layer in self.layers:
      output = layer.forward(inputs)
      input = output
    return output


class Linear(Layer):
  def __init__(self, num_in, num_out):
    self.weights = Param(tensor(np.random.randn(num_in, num_out), requires_grad=True))
    self.bias = Param(tensor(np.zeros((num_out, 1)), requires_grad=True))
  
  def forward(self, inputs):
    output = Dot(self.weights.T, inputs).forward() + self.bias
    return output