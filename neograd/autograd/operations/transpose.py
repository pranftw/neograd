from .operation import Operation

class Transpose(Operation):
  def __init__(self, tens):
    super().__init__(self, False, tens)
    self.tens = self.tensors[0]
  
  def forward(self):
    self.tens.local_grad = 1
    return self.get_result_tensor(self.tens.data.T)

  def backward(self, local_grad, upper_grad):
    return (local_grad*upper_grad).T

def transpose(tens):
  return Transpose(tens).forward()