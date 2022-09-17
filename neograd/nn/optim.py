import numpy as np


class Optimizer:
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr

  def zero_grad(self):
    '''
      Since after loss.backward, only Tensors in memory are the params, only their
        gradients are reset since everytime a new graph is dynamically created
    '''
    for param in self.params:
      param.zero_grad()


class GD(Optimizer):
  '''
    Vanilla Gradient Descent
  '''
  def __init__(self, params, lr):
    super().__init__(params, lr)
  
  def step(self):
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.grad)
  
  def __repr__(self):
    return f'GD(params={self.params}, lr={self.lr})'
  
  def __str__(self):
    return f'GD(params={self.params}, lr={self.lr})'


class Momentum(Optimizer):
  '''
    Gradient Descent with Momentum
    https://youtu.be/k8fTYJPd3_I
  '''
  def __init__(self, params, lr, beta=0.9):
    super().__init__(params, lr)
    self.beta = beta
    self.init_momentum_grads()

  def step(self):
    self.update_momentum_grads()
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.momentum_grad)
  
  def init_momentum_grads(self):
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = np.zeros(param.shape)
  
  def update_momentum_grads(self):
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = (self.beta*param.momentum_grad) + ((1-self.beta)*param.grad)
  
  def __repr__(self):
    return f'Momentum(params={self.params}, lr={self.lr}, beta={self.beta})'
  
  def __str__(self):
    return f'Momentum(params={self.params}, lr={self.lr}, beta={self.beta})'


class RMSProp(Optimizer):
  '''
    RMSProp
    https://youtu.be/_e-LFe_igno
  '''
  def __init__(self, params, lr, beta=0.9, epsilon=1e-8):
    super().__init__(params, lr)
    self.beta = beta
    self.epsilon = epsilon
    self.init_rms_grads()

  def step(self):
    self.update_rms_grads()
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*(param.grad/(np.sqrt(param.rms_grad) + self.epsilon)))
  
  def init_rms_grads(self):
    for param in self.params:
      if param.requires_grad:
        param.rms_grad = np.zeros(param.shape)
  
  def update_rms_grads(self):
    for param in self.params:
      if param.requires_grad:
        param.rms_grad = (self.beta*param.rms_grad) + ((1-self.beta)*np.square(param.grad))
  
  def __repr__(self):
    return f'RMSProp(params={self.params}, lr={self.lr}, beta={self.beta}, epsilon={self.epsilon})'
  
  def __str__(self):
    return f'RMSProp(params={self.params}, lr={self.lr}, beta={self.beta}, epsilon={self.epsilon})'
  
