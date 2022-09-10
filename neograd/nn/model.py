from .layers import Container, Layer


class Model:
  __slots__ = ['model', 'layer_types']

  def __init__(self, model):
    self.model = model
    self.layer_types = (Container, Layer)
  
  def __call__(self, inputs):
    return self.forward(inputs)

  def get_layers(self):
    layers = []
    for attr in dir(self.model):
      model_attr_val = self.model.__getattribute__(attr) 
      if issubclass(model_attr_val.__class__, self.layer_types):
        layers.append(model_attr_val)
    return layers
  
  def get_params(self):
    params = []
    layers = self.get_layers()
    for layer in layers:
      params+=layer.get_params()
    return params