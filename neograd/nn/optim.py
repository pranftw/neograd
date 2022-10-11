import numpy as np
from ..autograd.utils import get_graph


class Optimizer:
  '''Base class for all optimizers

  Parameters:
    params (list of Param): Params that need to be updated
    lr (float): Learning rate
  '''
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr

  def zero_grad(self, all_members=False):
    '''Resets the grads of tensors

    By default, since after loss.backward, only Tensors in memory are the params, only their
    gradients are reset since everytime a new graph is dynamically created

    However if retain_graph=True in backward, then all the members in the graph, need to be
    zero_grad-ed to get the correct gradients, to prevent this all_members can be set to True

    Args:
      all_members (bool): If all the members in the graph should be zero_grad-ed.
        Defaults to False
    '''
    if all_members:
      graph = get_graph()
      graph.zero_grad()
    for param in self.params: # This is done for redundancy, if all_members=True on a graph that's been reset
        param.zero_grad()


class GD(Optimizer):
  '''Vanilla Gradient Descent
  '''
  def __init__(self, params, lr):
    super().__init__(params, lr)
  
  def step(self):
    '''Updates the params
    '''
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.grad)
  
  def __repr__(self):
    return f'GD(params={self.params}, lr={self.lr})'
  
  def __str__(self):
    return f'GD(params={self.params}, lr={self.lr})'


class Momentum(Optimizer):
  '''Gradient Descent with Momentum
    
  https://youtu.be/k8fTYJPd3_I

  Parameters:
    beta (float): Value of Beta
  '''
  def __init__(self, params, lr, beta=0.9):
    super().__init__(params, lr)
    self.beta = beta
    self.init_momentum_grads()

  def step(self):
    '''Updates the params

    Uses the momentum_grads to update the params
    '''
    self.update_momentum_grads()
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.momentum_grad)
  
  def init_momentum_grads(self):
    '''Initializes momentum grads

    For each param, it sets its momentum_grad to 0
    '''
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = 0
  
  def update_momentum_grads(self):
    '''Updates momentum grads
    '''
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = (self.beta*param.momentum_grad) + ((1-self.beta)*param.grad)
  
  def __repr__(self):
    return f'Momentum(params={self.params}, lr={self.lr}, beta={self.beta})'
  
  def __str__(self):
    return f'Momentum(params={self.params}, lr={self.lr}, beta={self.beta})'


class RMSProp(Optimizer):
  '''RMSProp
    
  https://youtu.be/_e-LFe_igno

  Parameters:
    beta (float): Value of Beta
    epsilon (float): Value of epsilon
  '''
  def __init__(self, params, lr, beta=0.9, epsilon=1e-8):
    super().__init__(params, lr)
    self.beta = beta
    self.epsilon = epsilon
    self.init_rms_grads()

  def step(self):
    '''Updates the params

    Uses the rms_grads to update the params
    '''
    self.update_rms_grads()
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*(param.grad/(np.sqrt(param.rms_grad) + self.epsilon)))
  
  def init_rms_grads(self):
    '''Initializes rms grads

    For each param, it sets its rms_grad to 0
    '''
    for param in self.params:
      if param.requires_grad:
        param.rms_grad = 0
  
  def update_rms_grads(self):
    '''Updates rms grads
    '''
    for param in self.params:
      if param.requires_grad:
        param.rms_grad = (self.beta*param.rms_grad) + ((1-self.beta)*np.square(param.grad))
  
  def __repr__(self):
    return f'RMSProp(params={self.params}, lr={self.lr}, beta={self.beta}, epsilon={self.epsilon})'
  
  def __str__(self):
    return f'RMSProp(params={self.params}, lr={self.lr}, beta={self.beta}, epsilon={self.epsilon})'


class Adam(Optimizer):
  '''Adam

  https://youtu.be/JXQT_vxqwIs

  Parameters:
    iter (int): The number of iterations that has occurred, used for bias correction
    beta1 (float): Value of beta1
    beta2 (float): Value of beta2
    epsilon (float): Value of epsilon
  '''
  def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    super().__init__(params, lr)
    self.iter = 0
    self.beta1, self.beta2 = beta1, beta2
    self.epsilon = epsilon
    self.init_adam_grads()
  
  def step(self):
    '''Updates the params

    Uses the rms_grads and momentum_grads to update the params
    '''
    self.iter+=1
    self.update_adam_grads()
    for param in self.params:
      if param.requires_grad:
        bias_corrected_momentum_grad = param.momentum_grad/(1-(self.beta1**self.iter))
        bias_corrected_rms_grad = param.rms_grad/(1-(self.beta2**self.iter))
        param.data -= (self.lr*(bias_corrected_momentum_grad/(np.sqrt(bias_corrected_rms_grad)+self.epsilon)))
  
  def init_adam_grads(self):
    '''Initializes rms grads and momentum grads

    For each param, it sets its rms_grad to 0, momentum_grad to 0
    '''
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = 0
        param.rms_grad = 0
  
  def update_adam_grads(self):
    '''Updates rms grads and momentum_grads
    '''
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = (self.beta1*param.momentum_grad) + ((1-self.beta1)*param.grad)
        param.rms_grad = (self.beta2*param.rms_grad) + ((1-self.beta2)*np.square(param.grad))

  def reset_iter(self):
    '''Resets iter to 0
    '''
    self.iter = 0
  
  def __repr__(self):
    return f'Adam(params={self.params}, lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})'
  
  def __str__(self):
    return f'Adam(params={self.params}, lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})'
    
  
