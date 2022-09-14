from .layers import Container, Layer


class Model:
  __slots__ = ['model', 'layer_types']

  def __init__(self, model):
    self.model = model
    self.layer_types = (Container, Layer)
  
  def __call__(self, inputs):
    return self.forward(inputs)

  def get_layers(self):
    '''
      In the attributes of the model object, it searches if any of them are
        subclasses of Container/Layer, if so it considers it as a layer
    '''
    layers = []
    for attr in dir(self.model):
      model_attr_val = self.model.__getattribute__(attr) 
      if issubclass(model_attr_val.__class__, self.layer_types):
        layers.append(model_attr_val)
    return layers
  
  def get_params(self):
    '''
      Gathers the params of the whole model by iterating through all layers and getting their params
    '''
    params = []
    layers = self.get_layers()
    for layer in layers:
      params+=layer.get_params()
    return params
  
  def __repr__(self):
    return f'Model( {[str(layer) for layer in self.layers]} )'
  
  def __str__(self):
    return f'Model( {[str(layer) for layer in self.layers]} )'