import numpy as np
from .operation import Operation


# <------------RELU------------>

class ReLU(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.maximum(0, tens.data), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:np.where(tens.data>=0, 1, 0)*ug)

def relu(tens):
  return ReLU().forward(tens)


# <------------SIGMOID------------>

class Sigmoid(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(1/(1+np.exp(-tens.data)), tens)
  
  def backward(self, tens):
    result = 1/(1+np.exp(-tens.data))
    tens.set_grad_fn(lambda ug:(result*(1-result))*ug)

def sigmoid(tens):
  return Sigmoid().forward(tens)


# <------------TANH------------>

class Tanh(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.tanh(tens.data), tens)
  
  def backward(self, tens):
    result = np.tanh(tens.data)
    tens.set_grad_fn(lambda ug:(1-np.power(result,2))*ug)

def tanh(tens):
  return Tanh().forward(tens)


# <------------SOFTMAX------------>

class Softmax(Operation):
  def __init__(self, axis):
    self.axis = axis

  def forward(self, tens):
    tens = self.get_tensors(tens)
    result = np.apply_along_axis(self.calc_softmax, self.axis, tens.data)
    return self.get_result_tensor(result, tens)
  
  def backward(self, tens):
    def softmax_grad(arr, ug_slices): # arr will always be 1d array
      local_grad = -np.broadcast_to(arr, (arr.size, arr.size))
      np.fill_diagonal(local_grad, 1+np.diagonal(local_grad))
      local_grad = local_grad*arr.reshape(arr.size, 1)
      result = np.dot(local_grad, ug_slices.pop(0))
      return result
    
    def get_ug_slices(arr, ug_slices):
      ug_slices.append(arr)

    def grad_backward(ug):
      result = np.apply_along_axis(self.calc_softmax, self.axis, tens.data)
      ug_slices = []
      np.apply_along_axis(get_ug_slices, self.axis, ug, ug_slices)
      grads = np.apply_along_axis(softmax_grad, self.axis, result, ug_slices)
      return grads

    tens.set_grad_fn(grad_backward)

  def calc_softmax(self, arr):
    exponentiated = np.exp(arr-np.max(arr))
    sum_val = np.sum(exponentiated)
    return exponentiated/sum_val

def softmax(tens, axis):
  return Softmax(axis).forward(tens)