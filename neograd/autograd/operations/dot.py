from .operation import Operation
import numpy as np

class Dot(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, False, tens1, tens2)
    self.tens1, self.tens2 = self.tensors
  
  def forward(self):
    self.tens1.local_grad = self.tens2.data
    self.tens2.local_grad = self.tens1.data
    return self.get_result_tensor(np.dot(self.tens1.data, self.tens2.data))
  
  def backward_fns(self):
    def backward1(self, local_grad, upper_grad):
      return np.dot(upper_grad, local_grad.T)
    def backward2(self, local_grad, upper_grad):
      return np.dot(local_grad.T, upper_grad)
    return backward1, backward2

def dot(tens1, tens2):
  return Dot(tens1, tens2).forward()