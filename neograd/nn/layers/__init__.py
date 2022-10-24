from copy import deepcopy
from ...autograd.tensor import Tensor


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


class Param(Tensor):
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


# these imports should be done after defining Layer, Container, Param to avoid circular import
from .misc import Sequential, Linear, Dropout
from .conv import Conv2D, Conv3D, MaxPool2D, MaxPool3D