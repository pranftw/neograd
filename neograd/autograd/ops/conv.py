import math
import numpy as np
from .operation import Operation


class Conv:
  '''Base class for Convolution and Pooling operations

  Parameters:
    padding (int): Padding value to be applied
    stride (int): Stride to be taken
  '''
  def generate_fragments(self, padded_data, kernel_shape):
    '''Generates fragments of data

    Takes the data and slices it to the shape of kernel_shape in x and y axis (the last two
    dims), including all the other dimensions, effectively taking a chunk/fragment of the data.

    Each time a chunk is taken, the x index and y index are incremented by the stride

    Args:
      padded_data (np.ndarray): Data that is already padded, to be convolved on
      kernel_shape (tuple): Shape of kernel to be convolved with
    
    Yields:
      sliced inputs_data(fragment), row slice(start and end indices of fragment among rows)
      and column slice(start and end indices of fragment among columns)

    Raises:
      AssertionError: if stride isn't greater than or equal to 1
      AssertionError: if padding isn't greater than or equal to 0
    '''
    assert self.stride>=1, 'Stride must be greater than or equal to 1'
    assert self.padding>=0, 'Padding must be greater than or equal to zero'
    padded_data_slice = ()
    padded_data_slice += ((slice(None)),)*(len(padded_data.shape)-2)
    inputs_x_dim, inputs_y_dim = padded_data.shape[-2:]
    kernel_x_dim, kernel_y_dim = kernel_shape[-2:]
    j = 0
    while(j+kernel_y_dim<=inputs_y_dim):
      i = 0
      while(i+kernel_x_dim<=inputs_x_dim):
        row_slice = slice(i, i+kernel_x_dim)
        col_slice = slice(j, j+kernel_y_dim)
        yield padded_data[padded_data_slice+(row_slice, col_slice)], row_slice, col_slice
        i+=self.stride
      j+=self.stride
  
  def get_result_shape(self, inputs_shape, kernel_shape):
    '''Calculates the x and y dimensions of result of convolution

    Takes the inputs_shape and kernel_shape and returns the x and y dims of the result
    of convolving inputs with the kernel as a 2D convolution

    Args:
      inputs_shape (tuple): Shape of inputs
      kernel_shape (tuple): Shape of kernel
    
    Returns:
      tuple of x and y dims of result
    '''
    inputs_x_dim, inputs_y_dim = inputs_shape[-2:]
    kernel_x_dim, kernel_y_dim = kernel_shape[-2:]
    def result_dim(inputs_dim, kernel_dim):
      return math.floor(((inputs_dim + (2*self.padding) - kernel_dim)/self.stride) + 1)
    result_x_dim = result_dim(inputs_x_dim, kernel_x_dim)
    result_y_dim = result_dim(inputs_y_dim, kernel_y_dim)
    return result_x_dim, result_y_dim
  
  def pad(self, data):
    '''Pads the data in x and y dimensions (last two dims)

    Only the last two dimensions are padded rest aren't padded(padded with 0, 
    has no effect)

    Args:
      data (np.ndarray): Data to be padded
    
    Returns:
      data that is padded
    '''
    padding = () 
    padding+=((0,0),)*(len(data.shape)-2)
    padding+=((self.padding,self.padding),)*2
    return np.pad(data, padding)

  def unpad(self, padded_data):
    '''Unpads the padded data

    Slices the padded_data in last two dimensions where it is padded, by rejecting
    the padding and only extracting the original data

    Args:
      padded_data (np.ndarray): Data that needs to be unpadded
    
    Returns:
      Unpadded data
    '''
    padded_x_dim, padded_y_dim = padded_data.shape[-2:]
    extractor_slice = ()
    extractor_slice+=((slice(None)),)*(len(padded_data.shape)-2)
    extractor_slice+=(slice(self.padding,padded_x_dim-self.padding), slice(self.padding,padded_y_dim-self.padding))
    return padded_data[extractor_slice]
  
  def fragment_iterator(self, padded_inputs, kernel_shape, *args):
    '''Zips the fragments with any other args

    Args:
      padded_inputs (np.ndarray): Inputs that are padded
      kernel_shape (tuple): Shape of the kernel
      *args: Any other objects that should be zipped together, this is usually the
        upper gradient chunks that are the result of the convolution
    
    Returns:
      zip of fragments and ony other objects in args
    '''
    return zip(self.generate_fragments(padded_inputs, kernel_shape), *args)


# <------------CONV2D------------>
class Conv2D(Operation, Conv):
  '''Implements 2D convolution

  2D convolution where the inputs has only 1 channel and its shape is of the form
  (num_examples, x_dim, y_dim) is convolved with a 2D kernel
  '''
  def __init__(self, padding, stride):
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs, kernel, bias):
    '''Implements the forward pass

    Each fragment of shape (num_examples, kernel_shape[0], kernel_shape[1]) is element wise
    multipled with the kernel and then summed along its x and y axis and then adds it with the bias

    Args:
      inputs (Tensor or int or float or list or np.ndarray): Tensor to be convolved on
      kernel (Tensor or int or float or list or np.ndarray): Tensor to be convolved with(weights)
      bias (Tensor or int or float or list or np.ndarray): bias value
    
    Returns:
      Tensor of the result that is convolved
    '''
    inputs, kernel, bias = self.get_tensors(inputs, kernel, bias)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], *self.get_result_shape(inputs.shape, kernel.shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment, _, _), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(outputs.shape[-2:])):
      output = np.sum((fragment*kernel.data), axis=(1,2)) + bias.data
      outputs[:,idx[0],idx[1]] = output
    return self.get_result_tensor(outputs, inputs, kernel, bias)
  
  def backward(self, inputs, kernel, bias):
    '''Sets the grad_fn of inputs, kernel and bias

    Since each convolution of a fragment with kernel results in a Tensor, the corresponding upper
    gradient values of all the examples are taken.
    
    inputs_grads are initialized to zero of shape of padded_inputs, then the corresponding gradient
    that is calculated is tucked into the inputs_grads based on the row slice and the column slice.
    The inputs_grads are then unpadded to reject the gradients of pads and only keeps the gradients
    of the original inputs

    kernel_grads are intitialized to zero and since they are used multiple times, its gradient is added
    each time

    Since bias is added to all outputs, its gradient is just
    the sum of the upper gradient

    Args:
      inputs (Tensor): Tensor that is convolved on
      kernel (Tensor): Tensor that is convolved with(weights)
      bias (Tensor): bias value
    '''
    from ..utils import unbroadcast_data
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grads = np.zeros(padded_inputs.shape)
      for (fragment, row_slice, col_slice), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,idx[0],idx[1]]
        sum_grad = np.ones(fragment.shape)*sliced_ug.reshape(sliced_ug.size,1,1)
        fragment_grad = kernel.data*sum_grad
        inputs_grads[:, row_slice, col_slice]+=fragment_grad
      unpadded_inputs_grads = self.unpad(inputs_grads)
      return unpadded_inputs_grads

    def kernel_backward(ug):
      kernel_grads = np.zeros(kernel.shape)
      for (fragment, _, _), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
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
    '''Validates the inputs

    Args:
      inputs (Tensor or np.ndarray): Data to be validated

    Raises:
      ValueError: If inputs aren't of format (num_examples, x_dim, y_dim)
    '''
    if len(inputs.shape)!=3: # The first dimension should be number of examples
      raise ValueError("Only 3D inputs, with 0th dim as number of examples are supported!")

def conv2d(inputs, kernel, bias, padding, stride):
  '''Abstraction of Conv2D.forward

  Args:
    inputs (Tensor or int or float or list or np.ndarray): Tensor to be convolved on
    kernel (Tensor or int or float or list or np.ndarray): Tensor to be convolved with(weights)
    bias (Tensor or int or float or list or np.ndarray): bias value
    padding (int): Padding value to be applied
    stride (int): Stride to be taken

  Returns:
    Tensor of the result
  '''
  return Conv2D(padding, stride).forward(inputs, kernel, bias)


# <------------CONV3D------------>
class Conv3D(Operation, Conv):
  '''Implements 3D convolution

  3D convolution over a colume where the inputs has multiple channels and
  its shape is of the form (num_examples, num_channels, x_dim, y_dim) is convolved
  with a 3D kernel of shape (num_channels, kernel_shape[0], kernel_shape[1])
  '''
  def __init__(self, padding, stride):
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs, kernel, bias):
    '''Implements the forward pass

    Each fragment of shape (num_examples, num_channels, kernel_shape[0], kernel_shape[1])
    is element wise multipled with the kernel and then summed along its x, y and z axis and
    then adds it with the bias

    Args:
      inputs (Tensor or int or float or list or np.ndarray): Tensor to be convolved on
      kernel (Tensor or int or float or list or np.ndarray): Tensor to be convolved with(weights)
      bias (Tensor or int or float or list or np.ndarray): bias value
    
    Returns:
      Tensor of the result that is convolved
    '''
    inputs, kernel, bias = self.get_tensors(inputs, kernel, bias)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], kernel.shape[0], *self.get_result_shape(inputs.shape, kernel.shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(outputs.shape[-2:])):
      expanded_fragment = np.expand_dims(fragment, axis=1)
      output = expanded_fragment*kernel.data
      output = np.sum(output, axis=(2,3,4)) + bias.data
      outputs[:,:,idx[0],idx[1]] = output
    return self.get_result_tensor(outputs, inputs, kernel, bias)
  
  def backward(self, inputs, kernel, bias):
    '''Sets the grad_fn of inputs, kernel and bias

    Since each convolution of a fragment with kernel results in a Tensor, the corresponding upper
    gradient values of all the examples, across all channels are taken.
    
    To calculate sum_grad which is the gradient of the sum operation during forward pass, the fragment
    needs to be expanded along first axis to allow for broadcasting of upper gradient slice.

    inputs_grads are initialized to zero of shape of padded_inputs, then the corresponding gradient
    that is calculated is tucked into the inputs_grads based on the row slice and the column slice.
    The inputs_grads are then unpadded to reject the gradients of pads and only keeps the gradients
    of the original inputs

    kernel_grads are intitialized to zero and since they are used multiple times, its gradient is added
    each time

    Since bias is a vector here and is not added to all the outputs, it is only summed across all the examples
    and the first and second axis

    Args:
      inputs (Tensor): Tensor that is convolved on
      kernel (Tensor): Tensor that is convolved with(weights)
      bias (Tensor): bias value
    '''
    from ..utils import unbroadcast_data
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grads = np.zeros(padded_inputs.shape)
      for (fragment, row_slice, col_slice), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
        expanded_fragment = np.expand_dims(fragment, axis=1)
        sliced_ug = ug[:,:,idx[0],idx[1]]
        sliced_ug = sliced_ug.reshape(*sliced_ug.shape,1,1,1)
        sum_grad = np.ones(expanded_fragment.shape)*sliced_ug
        fragment_grad = np.sum(sum_grad*kernel.data, axis=1)
        inputs_grads[:,:,row_slice,col_slice]+=fragment_grad
      unpadded_inputs_grads = self.unpad(inputs_grads)
      return unpadded_inputs_grads
    
    def kernel_backward(ug):
      kernel_grads = np.zeros(kernel.shape)
      for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
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
    '''Validates the inputs

    Args:
      inputs (Tensor or np.ndarray): Data to be validated

    Raises:
      ValueError: If inputs aren't of format (num_examples, num_channels, x_dim, y_dim)
    '''
    if len(inputs.shape)!=4:
      raise ValueError("Only 4D inputs, with 0th dim as number of examples, 1st dim as number of channels are supported!")

def conv3d(inputs, kernel, bias, padding, stride):
  '''Abstraction of Conv3D.forward

  Args:
    inputs (Tensor or int or float or list or np.ndarray): Tensor to be convolved on
    kernel (Tensor or int or float or list or np.ndarray): Tensor to be convolved with(weights)
    bias (Tensor or int or float or list or np.ndarray): bias value
    padding (int): Padding value to be applied
    stride (int): Stride to be taken

  Returns:
    Tensor of the result
  '''
  return Conv3D(padding, stride).forward(inputs, kernel, bias)


# <------------MAXPOOL2D------------>
class MaxPool2D(Operation, Conv):
  '''Implements 2D MaxPooling

  In MaxPooling, it is differentiable only if kernel_shape along with the stride
  covers the entire input if not it is not differentiable and fails gradient
  checking

  Parameters:
    kernel_shape (tuple): Shape of the kernel
  '''
  def __init__(self, kernel_shape, padding, stride):
    '''
    Raises:
      ValueError: if kernel shape dimensions aren't 2D
    '''
    if len(kernel_shape)==2:
      self.kernel_shape = kernel_shape
    else:
      raise ValueError("Only 2D kernels are allowed!")
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs):
    '''Forward pass of max pooling 2d

    The fragments are generated, for each of it, the maximum value in the x and y dims
    is returned for all examples

    Args:
      inputs (Tensor or int or float or list or np.ndarray): Data to be maxpooled
    
    Returns:
      Tensor of the result
    '''
    inputs = self.get_tensors(inputs)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], *self.get_result_shape(inputs.shape, self.kernel_shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(outputs.shape[-2:])):
      outputs[:,idx[0],idx[1]] = np.max(fragment, axis=(1,2))
    return self.get_result_tensor(outputs, inputs)
  
  def backward(self, inputs):
    '''Sets the grad_fn of inputs

    Since argmax operates only on one axis, the fragment is first flattened across x and
    y dims (last two dims)
    Then the one hot encoding of the max indices are taken that are the multiplied
    with the corresponding upper grad slice
    fragment_grad is reshaped to original fragment shape

    Args:
      inputs (Tensor): Tensor that is maxpooled
    '''
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grad = np.empty(padded_inputs.shape)
      for (fragment,row_slice,col_slice),idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,idx[0],idx[1]]
        fragment_shape = fragment.shape
        flattened_fragment = fragment.reshape(fragment_shape[0], fragment_shape[1]*fragment_shape[2])
        args = np.argmax(flattened_fragment, axis=-1)
        fragment_grad = np.eye(flattened_fragment.shape[-1])[args] # one hot encoding of args
        fragment_grad = fragment_grad.reshape(fragment_shape)
        inputs_grad[:,row_slice,col_slice] = fragment_grad*np.expand_dims(sliced_ug,axis=(1,2))
      unpadded_inputs_grads = self.unpad(inputs_grad)
      return unpadded_inputs_grads

    inputs.set_grad_fn(inputs_backward)
  
  def validate_inputs(self, inputs):
    '''Validates the inputs

    Args:
      inputs (Tensor or np.ndarray): Data to be validated

    Raises:
      ValueError: If inputs aren't of format (num_examples, x_dim, y_dim)
    '''
    if len(inputs.shape)!=3: # The first dimension should be number of examples
      raise ValueError("Only 3D inputs, with 0th dim as number of examples are supported!")

def maxpool2d(inputs, kernel_shape, padding, stride):
  '''Abstraction of MaxPool2D.forward

  Args:
    inputs (Tensor or int or float or list or np.ndarray): Tensor to be convolved on
    kernel_shape (tuple): Shape of the kernel
    padding (int): Padding value to be applied
    stride (int): Stride to be taken

  Returns:
    Tensor of the result
  '''
  return MaxPool2D(kernel_shape, padding, stride).forward(inputs)


# <------------MAXPOOL3D------------>
class MaxPool3D(Operation, Conv):
  '''Implements 3D MaxPooling

  In MaxPooling, it is differentiable only if kernel_shape along with the stride
  covers the entire input if not it is not differentiable and fails gradient
  checking

  Parameters:
    kernel_shape (tuple): Shape of the kernel
  '''
  def __init__(self, kernel_shape, padding, stride):
    '''
    Raises:
      ValueError: if kernel shape dimensions aren't 2D
    '''
    if len(kernel_shape)==2:
      self.kernel_shape = kernel_shape
    else:
      raise ValueError("Only 2D kernels are allowed!")
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs):
    '''Forward pass of max pooling 3d

    The fragments are generated, for each of it, the maximum value in the x and y dims
    is returned for all examples across all channels

    Args:
      inputs (Tensor or int or float or list or np.ndarray): Data to be maxpooled
    
    Returns:
      Tensor of the result
    '''
    inputs = self.get_tensors(inputs)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], inputs.shape[1], *self.get_result_shape(inputs.shape, self.kernel_shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(outputs.shape[-2:])):
      outputs[:,:,idx[0],idx[1]] = np.max(fragment, axis=(2,3))
    return self.get_result_tensor(outputs, inputs)
  
  def backward(self, inputs):
    '''Sets the grad_fn of inputs

    Since argmax operates only on one axis, the fragment is first flattened across x and
    y dims (last two dims)
    Then the one hot encoding of the max indices are taken that are the multiplied
    with the corresponding upper grad slice
    fragment_grad is reshaped to original fragment shape

    Args:
      inputs (Tensor): Tensor that is maxpooled
    '''
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grad = np.empty(padded_inputs.shape)
      for (fragment,row_slice,col_slice),idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,:,idx[0],idx[1]]
        fragment_shape = fragment.shape
        flattened_fragment = fragment.reshape(fragment_shape[0]*fragment_shape[1], fragment_shape[2]*fragment_shape[3])
        args = np.argmax(flattened_fragment, axis=-1)
        fragment_grad = np.eye(flattened_fragment.shape[-1])[args] # one hot encoding of args
        fragment_grad = fragment_grad.reshape(fragment_shape)
        inputs_grad[:,:,row_slice,col_slice] = fragment_grad*np.expand_dims(sliced_ug,axis=(2,3))
      unpadded_inputs_grads = self.unpad(inputs_grad)
      return unpadded_inputs_grads
    
    inputs.set_grad_fn(inputs_backward)
  
  def validate_inputs(self, inputs):
    '''Validates the inputs

    Args:
      inputs (Tensor or np.ndarray): Data to be validated

    Raises:
      ValueError: If inputs aren't of format (num_examples, num_channels, x_dim, y_dim)
    '''
    if len(inputs.shape)!=4:
      raise ValueError("Only 4D inputs, with 0th dim as number of examples, 1st dim as number of channels are supported!")

def maxpool3d(inputs, kernel_shape, padding, stride):
  '''Abstraction of MaxPool3D.forward

  Args:
    inputs (Tensor or int or float or list or np.ndarray): Tensor to be convolved on
    kernel_shape (tuple): Shape of the kernel
    padding (int): Padding value to be applied
    stride (int): Stride to be taken

  Returns:
    Tensor of the result
  '''
  return MaxPool3D(kernel_shape, padding, stride).forward(inputs)