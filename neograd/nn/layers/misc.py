import numpy as np
from ..layers import Container, Layer, Param
from ...autograd import dot
from ...autograd.ops.operation import Operation


class Sequential(Container):
  '''Sequential Container
  
  Outputs of one layer are passed as inputs to the next layer, sequentially
  '''
  
  def __init__(self, *args):
    self.layers = args
  
  def forward(self, inputs):
    '''Forward pass of Sequential

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    for layer in self.layers:
      output = layer(inputs)
      inputs = output
    return output
  
  def __str__(self):
    return f'Sequential(\n{super().__str__()}\n)'
  
  def __repr__(self):
    return f'Sequential(\n{super().__repr__()}\n)'


class Linear(Layer):
  '''Implements a fully connected Layer

  Parameters:
    num_in (int): Number of inputs to the Layer
    num_out (int): Number of outputs from the Layer
    weights (Param): Weights of the Layer
    bias (Param): Bias of the Layer
  '''
  def __init__(self, num_in, num_out):
    self.num_in = num_in
    self.num_out = num_out
    self.weights = Param(np.random.randn(num_in, num_out), requires_grad=True)
    self.bias = Param(np.zeros((1, num_out)), requires_grad=True)
  
  def forward(self, inputs):
    '''Forward pass of Linear

    The inputs are dotted with weights and then bias is added

    Args:
      inputs (Tensor): Inputs to the Linear
    
    Returns:
      Tensor of the result
    '''
    return dot(inputs, self.weights) + self.bias
  
  def __repr__(self):
    return f'Linear({self.num_in}, {self.num_out})'
  
  def __str__(self):
    return f'Linear in:{self.num_in} out:{self.num_out}'


class Dropout(Layer, Operation):
  '''Dropout Layer
  
  https://youtu.be/D8PJAL-MZv8
  
  Parameters:
    prob (float): Probability with which to keep the inputs
      With probability=prob, the units are kept and with probability=1-prob,
      they are shut off
  '''
  def __init__(self, prob):
    Layer.__init__(self)
    assert prob>0 and prob<=1, 'Probability should be between 0 and 1'
    self.prob = prob
  
  def forward(self, inputs):
    '''Forward pass of Dropout

    The inputs are turned on with the given prob
    If in eval mode, then all inputs are always on

    Args:
      inputs (Tensor): Inputs to the Layer
    
    Returns:
      Tensor of the result
    '''
    if self.eval:
      filter = np.ones(inputs.shape) # dont discard anything, just dummy because if eval or not, backward needs to have filter arg
    else:
      filter = np.where(np.random.random(inputs.shape)<self.prob, 1, 0)
    inputs, filter = self.get_tensors(inputs, filter)
    if not(self.eval): # Deliberately repeated condition check for eval, because if in eval, it shouldnt be scaled by prob
      result = (inputs.data*filter.data)/self.prob
    else:
      result = inputs.data
    return self.get_result_tensor(inputs.data, inputs, filter)
  
  def backward(self, inputs, filter):
    '''Sets the grad_fn of inputs only because filter doesnt have requires_grad=True

    Just multiplies the upper gradient with the filter set during forward pass and scales
    it by the probability

    Args:
      inputs (Tensor): Inputs to the Layer
      filter (Tensor): The dropout filter that was applied during forward pass
    '''
    if not(self.eval):
      inputs.set_grad_fn(lambda ug:(ug*filter.data)/self.prob)
    inputs.set_grad_fn(lambda ug:ug)
  
  def __repr__(self):
    return f'Dropout(prob={self.prob})'
  
  def __str__(self):
    return f'Dropout(prob={self.prob})'