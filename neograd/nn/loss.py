import numpy as np
from ..autograd import sum as _sum, log
from ..autograd.ops.operation import Operation
from .activations import Softmax


class Loss:
  '''Base class of all loss functions
  '''
  def __call__(self, outputs, targets):
    '''Abstracts the forward method

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
    
    Returns:
      Tensor of the result
    '''
    return self.forward(outputs, targets)
  
  def get_num_examples(self, outputs_shape):
    '''Gathers the number of examples

    If dimensions of outputs_shape is 0, ie it is a scalar, then num_examples is 1
    Else the first dimension value of outputs_shape is taken as num_examples

    Args:
      outputs_shape (tuple of int): Shape of the outputs
    
    Returns:
      Number of examples
    '''
    if len(outputs_shape) in (0, 1):
      return 1
    else:
      return outputs_shape[0] 


# <------------MEANSQUAREDERROR------------>
class MSE(Loss):
  '''Mean Squared Error
  '''
  def forward(self, outputs, targets):
    '''Forward pass of MSE

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
    
    Returns:
      Tensor of the result
    '''
    num_examples = self.get_num_examples(outputs.shape)
    cost = (1/(2*num_examples))*_sum((outputs-targets)**2)
    return cost
  
  def __repr__(self):
    return f'MSE()'
  
  def __str__(self):
    return 'MeanSquaredError'


# <------------BINARYCROSSENTROPY------------>
class BCE(Loss):
  '''Binary Cross Entropy
  '''
  def forward(self, outputs, targets, epsilon=1e-9):
    '''Forward pass of BCE

    epsilon used  to prevent log0

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
      epsilon (float): For numerical stability of log Defaults to 1e-9
    
    Returns:
      Tensor of the result
    '''
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum((outputs*log(targets+epsilon)) + ((1-outputs)*(log(1-targets+epsilon))))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return f'BCE()'
  
  def __str__(self):
    return 'BinaryCrossEntropy'


# <------------CROSSENTROPY------------>
class CE(Loss):
  '''Cross Entropy
  '''
  def forward(self, outputs, targets, epsilon=1e-9):
    '''Forward pass of CE

    epsilon used  to prevent log0

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
      epsilon (float): For numerical stability of log Defaults to 1e-9
    
    Returns:
      Tensor of the result
    '''
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum(targets*log(outputs+epsilon))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return 'CE()'
  
  def __str__(self):
    return 'CrossEntropy'


# <------------SOFTMAXCROSSENTROPY------------>
class SoftmaxCE(Operation, Loss):
  '''Implements Softmax activation with CrossEntropyLoss

  Purpose of this is to eliminate costly Jacobian calculation involved
  with vanilla softmax activation. Since Softmax is most commonly used with
  Cross Entropy loss, if both are combined in one single Operation, then the derivative
  is a very minimal subtraction between the softmax output and the targets.
  So many intermediate backward calculations can be prevented with this.

  Parameters:
    axis (int or tuple of int): Axis along which to calculate the Softmax
      Defaults to None

  epsilon to prevent log0
  '''
  def __init__(self, axis):
    self.axis = axis

  def forward(self, outputs, targets, epsilon=1e-9):
    '''Calculates Softmax of inputs and the Cross Entropy loss

    Args:
      outputs (Tensor): Outputs of Layer/Container/Model/Operation
      targets (Tensor): Targets to be evaluated against
      epsilon (float): For numerical stability of log Defaults to 1e-9
    
    Returns:
      Tensor of the result
    '''
    num_examples = self.get_num_examples(outputs.shape)
    probs = Softmax.calc_softmax(outputs.data, axis=self.axis)
    entropy = np.sum(targets.data*np.log(probs+epsilon))
    cost = (-1/num_examples)*entropy
    return self.get_result_tensor(cost, outputs, targets)
  
  def backward(self, outputs, targets):
    '''Sets the grad_fn of outputs

    Args:
      outputs (Tensor): Tensor which is usually the outputs of the last layer
        of the network
      targets (Tensor): Targets to be evaluated against
    '''
    def sce_backward(ug):
      num_examples = self.get_num_examples(outputs.shape)
      probs = Softmax.calc_softmax(outputs.data, axis=self.axis)
      return (ug/num_examples)*(probs-targets.data) # ug is a scalar(1 by default), because loss calculated in forward is a scalar
    outputs.set_grad_fn(sce_backward)
    assert targets.requires_grad is False, 'Targets Tensor should have requires_grad=False'
