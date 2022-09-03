from .operation import Operation
import numpy as np

class Pow(Operation):
  def __init__(self, tens1, tens2):
    super().__init__(self, True, tens1, tens2)
    self.tens1, self.tens2 = self.tensors
  
  def forward(self):
    result = np.power(self.tens1.data, self.tens2.data)
    self.tens1.local_grad = (np.power(self.tens1.data, self.tens2.data-1) * self.tens2.data).flatten()
    self.tens2.local_grad = (result*np.log(self.tens1.data)).flatten()
    return self.get_result_tensor(result)
  
  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad)

def pow(tens1, tens2):
  return Pow(tens1, tens2).forward()