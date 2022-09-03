from .operation import Operation
import numpy as np

class Log(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.tensors[0]
  
  def forward(self):
    self.tens.local_grad = 1/self.tens.data
    return self.get_result_tensor(np.log(self.tens.data))
  
  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad)

def log(tens):
  return Log(tens).forward()