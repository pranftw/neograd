class Optimizer:
  def zero_grad(self):
    for param in self.params:
      param.zero_grad()


class GD(Optimizer):
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr
  
  def step(self):
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.grad)