from .operation import Operation
from ..helpers import mul_shape_dims
import numpy as np

class Sub(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors
  
  def forward(self):
    self.tens1.local_grad = np.eye(mul_shape_dims(self.get_broadcast_shape()))
    self.tens2.local_grad = -np.eye(mul_shape_dims(self.get_broadcast_shape()))
    return self.get_result_tensor(self.tens1.data-self.tens2.data)
  
  def backward(self, local_grad, upper_grad):
    return np.dot(local_grad, upper_grad)

def sub(tens1, tens2):
  return Sub(tens1, tens2)