import math
import numpy as np
from .operation import Operation


class Conv:
  def generate_fragments(self, inputs_data, kernel_shape):
    inputs_data_slice = ()
    inputs_data_slice += ((slice(None)),)*(len(inputs_data.shape)-2)
    inputs_x_dim, inputs_y_dim = inputs_data.shape[-2:]
    kernel_x_dim, kernel_y_dim = kernel_shape[-2:]
    j = 0
    while(j+kernel_y_dim<=inputs_y_dim):
      i = 0
      while(i+kernel_x_dim<=inputs_x_dim):
        row_slice = slice(i, i+kernel_x_dim)
        col_slice = slice(j, j+kernel_y_dim)
        yield inputs_data[inputs_data_slice+(row_slice, col_slice)], row_slice, col_slice
        i+=self.stride
      j+=self.stride
  
  def get_result_shape(self, inputs_shape, kernel_shape):
    inputs_x_dim, inputs_y_dim = inputs_shape[-2:]
    kernel_x_dim, kernel_y_dim = kernel_shape[-2:]
    def result_dim(inputs_dim, kernel_dim):
      return math.floor(((inputs_dim + (2*self.padding) - kernel_dim)/self.stride) + 1)
    result_x_dim = result_dim(inputs_x_dim, kernel_x_dim)
    result_y_dim = result_dim(inputs_y_dim, kernel_y_dim)
    return result_x_dim, result_y_dim
  
  def pad(self, data):
    padding = () 
    padding+=((0,0),)*(len(data.shape)-2)
    padding+=((self.padding,self.padding),)*2
    return np.pad(data, padding)

  def unpad(self, padded_data):
    padded_x_dim, padded_y_dim = padded_data.shape[-2:]
    extractor_slice = ()
    extractor_slice+=((slice(None)),)*(len(padded_data.shape)-2)
    extractor_slice+=(slice(self.padding,padded_x_dim-self.padding), slice(self.padding,padded_y_dim-self.padding))
    return padded_data[extractor_slice]
  
  def fragment_iterator(self, padded_inputs, kernel, *args):
    return zip(self.generate_fragments(padded_inputs, kernel.shape), *args)


# <------------CONV2D------------>

class Conv2D(Operation, Conv):
  def __init__(self, padding=0, stride=1):
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs, kernel, bias):
    inputs, kernel, bias = self.get_tensors(inputs, kernel, bias)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], *self.get_result_shape(inputs.shape, kernel.shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment, _, _), idx in self.fragment_iterator(padded_inputs, kernel, np.ndindex(outputs.shape[-2:])):
      output = np.sum((fragment*kernel.data), axis=2)
      output = np.sum(output, axis=1) + bias.data
      outputs[:,idx[0],idx[1]] = output
    return self.get_result_tensor(outputs, inputs, kernel, bias)
  
  def backward(self, inputs, kernel, bias):
    from ..utils import unbroadcast_data
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grads = np.zeros(padded_inputs.shape)
      for (fragment, row_slice, col_slice), idx in self.fragment_iterator(padded_inputs, kernel, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,idx[0],idx[1]]
        sum_grad = np.ones(fragment.shape)*sliced_ug.reshape(sliced_ug.size,1,1)
        fragment_grad = kernel.data*sum_grad
        inputs_grads[:, row_slice, col_slice]+=fragment_grad
      unpadded_inputs_grads = self.unpad(inputs_grads)
      return unpadded_inputs_grads

    def kernel_backward(ug):
      kernel_grads = np.zeros(kernel.shape)
      for (fragment, _, _), idx in self.fragment_iterator(padded_inputs, kernel, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,idx[0],idx[1]]
        sum_grad = np.ones(fragment.shape)*sliced_ug.reshape(sliced_ug.size,1,1)
        kernel_grad = unbroadcast_data(fragment*sum_grad, kernel.shape, fragment.shape)
        kernel_grads+=kernel_grad
      return kernel_grads

    def bias_backward(ug):
      return np.sum(ug)
      
    inputs.set_grad_fn(inputs_backward)
    kernel.set_grad_fn(kernel_backward)
    bias.set_grad_fn(bias_backward)
  
  def validate_inputs(self, inputs):
    if len(inputs.shape)!=3: # The first dimension should be number of examples
      raise ValueError("Only 3D inputs, with 0th dim as number of examples are supported!")

def conv2d(inputs, kernel, bias, padding, stride):
  return Conv2D(padding, stride).forward(inputs, kernel, bias)


# <------------CONV3D------------>

class Conv3D(Operation, Conv):
  def __init__(self, padding=0, stride=1):
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs, kernel, bias):
    inputs, kernel, bias = self.get_tensors(inputs, kernel, bias)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], kernel.shape[0], *self.get_result_shape(inputs.shape, kernel.shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, kernel, np.ndindex(outputs.shape[-2:])):
      expanded_fragment = np.expand_dims(fragment,1)
      output = expanded_fragment*kernel.data
      output = np.sum(output, axis=4)
      output = np.sum(output, axis=3)
      output = np.sum(output, axis=2) + bias.data
      outputs[:,:,idx[0],idx[1]] = output
    return self.get_result_tensor(outputs, inputs, kernel, bias)
  
  def backward(self, inputs, kernel, bias):
    from ..utils import unbroadcast_data
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grads = np.zeros(padded_inputs.shape)
      for (fragment, row_slice, col_slice), idx in self.fragment_iterator(padded_inputs, kernel, np.ndindex(ug.shape[-2:])):
        expanded_fragment = np.expand_dims(fragment,1)
        sliced_ug = ug[:,:,idx[0],idx[1]]
        sliced_ug = sliced_ug.reshape(*sliced_ug.shape,1,1,1)
        sum_grad = np.ones(expanded_fragment.shape)*sliced_ug
        fragment_grad = np.sum(sum_grad*kernel.data, axis=1)
        inputs_grads[:,:,row_slice,col_slice]+=fragment_grad
      unpadded_inputs_grads = self.unpad(inputs_grads)
      return unpadded_inputs_grads
    
    def kernel_backward(ug):
      kernel_grads = np.zeros(kernel.shape)
      for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, kernel, np.ndindex(ug.shape[-2:])):
        expanded_fragment = np.expand_dims(fragment,1)
        sliced_ug = ug[:,:,idx[0],idx[1]]
        sliced_ug = sliced_ug.reshape(*sliced_ug.shape,1,1,1)
        sum_grad = np.ones(expanded_fragment.shape)*sliced_ug
        kernel_grad = sum_grad*expanded_fragment
        kernel_grad = unbroadcast_data(kernel_grad, kernel.shape, kernel_grad.shape)
        kernel_grads+=kernel_grad
      return kernel_grads
    
    def bias_backward(ug):
      grad = np.sum(ug, axis=0)
      grad = np.sum(grad, axis=2, keepdims=True)
      grad = np.sum(grad, axis=1, keepdims=True)
      return grad
    
    inputs.set_grad_fn(inputs_backward)
    kernel.set_grad_fn(kernel_backward)
    bias.set_grad_fn(bias_backward)
  
  def validate_inputs(self, inputs):
    if len(inputs.shape)!=4:
      raise ValueError("Only 4D inputs, with 0th dim as number of examples, 1st dim as number of channels are supported!")

def conv3d(inputs, kernel, bias, padding, stride):
  return Conv3D(padding, stride).forward(inputs, kernel, bias)
