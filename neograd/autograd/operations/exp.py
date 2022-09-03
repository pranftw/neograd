from .operation import Operation
import numpy as np

class Exp(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.tensors[0]
  
  def forward(self):
    self.tens.local_grad = np.exp(self.tens.data)
    return self.get_result_tensor(self.tens.local_grad)
  
  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad)

def exp(tens):
  return Exp(tens).forward()