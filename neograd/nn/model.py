import hickle as hkl
from itertools import chain as list_flattener
from .layers import Container, Layer
from ..autograd.utils import get_graph


class Model:
  def __call__(self, inputs):
    return self.forward(inputs)
  
  def eval(self, no_track=True):
    return EvalMode(self, no_track)
  
  def get_layers(self):
    layers = {}
    for attr in dir(self):
      val = self.__getattribute__(attr)
      if isinstance(val, (Container, Layer)) and (val not in layers.values()):
        layers[attr] = val
    return layers
  
  def get_params(self, as_dict=False, return_frozen=True):
    '''
      Gathers the params of the whole model by iterating through all layers and getting their params
    '''
    params = {}
    for attr, layer in self.get_layers().items():
      if return_frozen or not(layer.frozen):
        params[attr] = layer.get_params(as_dict, return_frozen)
    return params if as_dict else list(list_flattener(*params.values()))
  
  def set_eval(self, eval):
    for layer in self.get_layers().values():
      layer.set_eval(eval)
  
  def save(self, fpath):
    '''
      Saves the params of the model
    '''
    params = self.get_params(as_dict=True)
    hkl.dump(params, fpath, mode='w')
    print(f"\nPARAMS SAVED at {fpath}\n")

  def load(self, fpath):
    '''
      Loads the params onto the model
    '''
    params = hkl.load(fpath)
    for attr, param in params.items():
      layer = self.__getattribute__(attr)
      layer.set_params(param)
    print(f"\nPARAMS LOADED from {fpath}\n")
  
  def __setattr__(self, attr, val):
    if isinstance(val, (Container, Layer)) and (attr in self.__dict__):
      raise AttributeError(f"Attribute {attr} has already been defined, it cannot be defined again for a Container/Layer")
    object.__setattr__(self, attr, val)
  
  def __repr__(self):
    return f'Model( {[str(layer) for layer in self.get_layers().values()]} )'
  
  def __str__(self):
    return f'Model( {[str(layer) for layer in self.get_layers().values()]} )'


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
      self.graph.track = False
    self.model.set_eval(True)
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    if self.no_track:
      self.graph.track = True
    self.model.set_eval(False)