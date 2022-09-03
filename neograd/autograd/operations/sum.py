from .operation import Operation
from ..helpers import mul_shape_dims
import numpy as np

class Sum(Operation):
  def __init__(self, tens, axis=0):
    super().__init__(self, True, tens)
    self.tens = self.tensors[0]
    self.axis = axis
  
  def forward(self):
    tens_shape = list(self.tens.shape)
    try:
      tens_shape[self.axis] = 1
    except IndexError:
      pass
    self.tens.local_grad = np.eye(mul_shape_dims(tuple(tens_shape)))
    return self.get_result_tensor(np.sum(self.tens.data, axis=self.axis))
  
  def backward(self, local_grad, upper_grad):
    split_arrays_grads = np.dot(local_grad, upper_grad)
    try:
      num_repeat = self.tens.shape[self.axis]
    except IndexError:
      num_repeat = 1
    split_arrays_grads = split_arrays_grads[np.newaxis]
    concatenated_grads = np.concatenate([split_arrays_grads]*num_repeat)
    return concatenated_grads

def sum(tens, axis=0):
  return Sum(tens, axis).forward()