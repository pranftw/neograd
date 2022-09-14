class Optimizer:
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
    self.params = params
    self.lr = lr
  
  def step(self):
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.grad)
  
  def __repr__(self):
    return f'GD(params={self.params}, lr={self.lr})'
  
  def __str__(self):
    return f'GD(params={self.params}, lr={self.lr})'