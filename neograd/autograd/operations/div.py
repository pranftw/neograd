from .operation import Operation
import numpy as np

class Div(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors

  def forward(self):
    self.tens1.local_grad = np.diag(np.ndarray.flatten(np.broadcast_to(1/self.tens2.data, self.tens2.broadcasted_shape)))
    self.tens2.local_grad = np.diag(np.ndarray.flatten(np.broadcast_to((-1*self.tens1.data)/np.power(self.tens2.data, 2), self.tens1.broadcasted_shape)))
    return self.get_result_tensor(self.tens1.data/self.tens2.data)
  
  def backward(self, local_grad, upper_grad):
    return np.dot(local_grad, upper_grad)

def div(tens1, tens2):
  return Div(tens1, tens2).forward()