from .layers import Container, Layer
from ..autograd.utils import get_graph


class Model:
  def __call__(self, inputs):
    return self.forward(inputs)
  
  def eval(self, no_track=True):
    return EvalMode(self, no_track)
  
  def get_layers(self):
    layers = []
    for attr in dir(self):
      val = self.__getattribute__(attr)
      if isinstance(val, (Container, Layer)):
        layers.append(val)
    return layers
  
  def get_params(self):
    '''
      Gathers the params of the whole model by iterating through all layers and getting their params
    '''
    params = []
    for layer in self.get_layers():
      params+=layer.get_params(params)
    return params
  
  def set_eval(self, eval):
    for layer in self.get_layers():
      layer.set_eval(eval)
  
  def __repr__(self):
    return f'Model( {[str(layer) for layer in self.layers]} )'
  
  def __str__(self):
    return f'Model( {[str(layer) for layer in self.layers]} )'


class EvalMode:
  '''
    Returns a ContextManager to run the model in eval mode, ie while testing the model.
    Use of this is that some layers like Dropout need to be turned off while
      testing
    
    Params:
      no_track:Bool - If True, then the backward graph is not created and the tensors
        aren't tracked
  '''
  def __init__(self, model, no_track):
    self.model = model
    self.no_track = no_track
    self.graph = get_graph()

  def __enter__(self):
    if self.no_track:
      self.graph.track = True
    self.model.set_eval(True)
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    if self.no_track:
      self.graph.track = False
    self.model.set_eval(True)