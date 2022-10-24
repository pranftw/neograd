import dill
from itertools import chain as list_flattener
from .layers import Container, Layer
from ..autograd.utils import get_graph


class Model:
  def __call__(self, inputs):
    '''Abstracts the forward method

    Args:
      inputs (Tensor): Inputs to the model
    
    Returns:
      Tensor of the result
    '''
    return self.forward(inputs)
  
  def eval(self, no_track=True):
    '''Invokes EvalMode ContextManager

    Args:
      no_track (bool): If Tensors shouldn't be tracked, Defaults to False
    
    Returns:
      EvalMode ContextManager
    '''
    return EvalMode(self, no_track)
  
  def get_layers(self):
    '''Gathers all the layers in the Model

    Accomplishes by going through all its attributes and if their values are
    instances of Container/Layer it is taken as a layer

    Returns:
      Dict with attributes as key and their objects as value
    '''
    layers = {}
    for attr, val in self.__dict__.items():
      if isinstance(val, (Container, Layer)):
        layers[attr] = val
    return layers
  
  def parameters(self, as_dict=False):
    '''Gathers the params of the whole Model

    Accomplishes this by iterating through all layers and getting their params
    
    Args:
      as_dict (bool): Whether to return the params as a dict. Defaults to False
    '''
    params = {}
    for attr, layer in self.get_layers().items():
      params[attr] = layer.parameters(as_dict)
    return params if as_dict else list(list_flattener(*params.values()))
  
  def set_eval(self, eval):
    '''Sets eval

    Sets the eval of its layers to eval argument

    Args:
      eval (bool): Whether in eval mode or not
    '''
    for layer in self.get_layers().values():
      layer.set_eval(eval)
  
  def save(self, fpath):
    '''Saves the params of the model in the specified file path

    Args:
      fpath (str): File path
    '''
    params = self.parameters(as_dict=True)
    with open(fpath, 'wb') as fp:
      dill.dump(params, fp)
    print(f"\nPARAMS SAVED at {fpath}\n")

  def load(self, fpath):
    '''Loads the params from the filepath onto the model

    Args:
      fpath (str): File path
    '''
    with open(fpath, 'rb') as fp:
      params = dill.load(fp)
    for attr, param in params.items():
      layer = self.__getattribute__(attr)
      layer.set_params(param)
    print(f"\nPARAMS LOADED from {fpath}\n")
  
  def __setattr__(self, attr, val):
    '''Sets the attributes

    Doesn't allow redefining an attribute that's been previously defined if the
    value of the attribute is instance of Container/Layer

    Args:
      attr (str): Attribute to be set
      val (object): Value to be set
    '''
    if isinstance(val, (Container, Layer)) and (attr in self.__dict__):
      raise AttributeError(f"Attribute {attr} has already been defined, it cannot be defined again for a Container/Layer")
    object.__setattr__(self, attr, val)
  
  def __repr__(self):
    return f'Model( {[str(layer) for layer in self.get_layers().values()]} )'
  
  def __str__(self):
    return f'Model( {[str(layer) for layer in self.get_layers().values()]} )'


class EvalMode:
  '''ContextManager for handling eval

  A ContextManager to run the model in eval mode, ie while testing the model.
  Use of this is that some layers like Dropout need to be turned off while
  testing
    
  Args:
    model (Model): Model to be put into eval
    no_track (bool): If True, then the backward graph is not created and the tensors
      aren't tracked
    graph (Graph): Graph that's currently in use
  '''
  def __init__(self, model, no_track):
    self.model = model
    self.no_track = no_track
    self.graph = get_graph()

  def __enter__(self):
    '''
    If no_track, then sets graph track to False
    Puts model in eval mode, by setting it to True
    '''
    if self.no_track:
      self.graph.track = False
    self.model.set_eval(True)
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    '''
    If no_track, then sets back graph track to True
    Puts model out of eval mode, by setting it to False
    '''
    if self.no_track:
      self.graph.track = True
    self.model.set_eval(False)