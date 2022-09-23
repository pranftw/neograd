import numpy as np
import math
from .node import Node


class Operation:
  '''
    Used when some input is getting transformed into an output, for functions
      where gradient calculation is required with the forward pass and the backward
      pass defined
  '''

  graph = None
  
  def process_operands(self, operands):
    '''
      All operands are converted to Tensor

      Params:
        operands:(any class that is supported/Tensor)
    '''
    from .tensor import Tensor
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_tensors(self, *operands):
    '''
      Returns the processed operands as tuple of Tensors

      Params:
        operands:*args(any class that is supported/Tensor)
    '''
    tensors = self.process_operands(operands)
    if len(tensors)==0:
      return None
    elif len(tensors)==1:
      return tensors[0]
    else:
      return tensors
  
  def get_broadcast_shape(self, *tensors):
    '''
      If the tensors can be broadcasted, then the broadcasted
        shape is returned, else None
      
      Params:
        tensors:*args(Tensor)
    '''
    try:
      return np.broadcast_shapes(*(tens.data.shape for tens in tensors))
    except ValueError:
      return None
  
  def result_requires_grad(self, tensors):
    '''
      Checks if the result requires_grad given the operands of the Operation, if atleast
        one operand requires_grad, then result will also have requires_grad
      
      Params:
        tensors:(Tensor)
    '''
    for tens in tensors:
      if tens.requires_grad:
        return True
    return False
  
  def get_result_tensor(self, result, *tensors):
    '''
      Returns the result tensor of the Operation
      Creates a Node for the result_tensor with parent_broadcast_shape and parent_needs_broadcasting
      Adds the edges to the graph

      Params:
        result:np object - Result after performing a raw numpy operation
        tensors:*args(Tensor)
    '''
    from .tensor import Tensor
    from .utils import get_graph
    graph = get_graph()
    result = result.astype(np.ndarray)
    result_tensor = Tensor(result, self.result_requires_grad(tensors))
    if graph.track:
      result_node = Node(result_tensor)
      result_node.backward_fn = self.backward
      result_node.parent_broadcast_shape = self.get_broadcast_shape(*tensors)
      graph.add_edge(result_node, tensors)
    return result_tensor
  
  def backward(self, *args):
    raise NotImplementedError(f"Backward method not implemented for Operation {self}")


# <------------ADD------------>

class Add(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data+tens2.data, tens1, tens2)

  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:ug)

def add(tens1, tens2):
  return Add().forward(tens1, tens2)


# <------------SUB------------>

class Sub(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data-tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:-ug)

def sub(tens1, tens2):
  return Sub().forward(tens1, tens2)


# <------------MUL------------>

class Mul(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data*tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:tens2.data*ug)
    tens2.set_grad_fn(lambda ug:tens1.data*ug)

def mul(tens1, tens2):
  return Mul().forward(tens1, tens2)


# <------------DIV------------>

class Div(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data/tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:(1/tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:((-1*tens1.data)/np.power(tens2.data, 2))*ug)

def div(tens1, tens2):
  return Div().forward(tens1, tens2)


# <------------DOT------------>

class Dot(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.dot(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:np.dot(ug, tens2.data.T))
    tens2.set_grad_fn(lambda ug:np.dot(tens1.data.T, ug))

def dot(tens1, tens2):
  return Dot().forward(tens1, tens2)


# <------------EXP------------>

class Exp(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.exp(tens.data), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:np.exp(tens.data)*ug)

def exp(tens):
  return Exp().forward(tens)


# <------------LOG------------>

class Log(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.log(tens.data), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:(1/tens.data)*ug)

def log(tens):
  return Log().forward(tens)


# <------------POW------------>

class Pow(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.power(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    result = np.power(tens1.data, tens2.data)
    tens1.set_grad_fn(lambda ug:(np.power(tens1.data, tens2.data-1) * tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:(result*np.log(tens1.data))*ug)

def pow(tens1, tens2):
  return Pow().forward(tens1, tens2)


# <------------SUM------------>

class Sum(Operation):
  def __init__(self, axis=None):
    self.axis = axis
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.sum(tens.data, axis=self.axis), tens)
  
  def backward(self, tens):
    def grad_backward(ug):
      tens_shape = list(tens.shape)
      if self.axis is not None:
        try:
          tens_shape[self.axis] = 1
        except IndexError:
          pass
        lg = 1
      else:
        lg = np.ones(tens.shape)

      if self.axis is not None:
        grads = np.dot(lg,ug)
        try:
          num_repeat = tens.shape[self.axis]
        except IndexError:
          num_repeat = 1
        grads = grads[np.newaxis]
        grads = np.concatenate([grads]*num_repeat)
      else:
        grads = lg*ug
      return grads
    tens.set_grad_fn(grad_backward)

def sum(tens, axis=None):
  return Sum(axis).forward(tens)


# <------------TRANSPOSE------------>

class Transpose(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.T, tens)

  def backward(self, tens):
    tens.set_grad_fn(lambda ug:ug.T)

def transpose(tens):
  return Transpose().forward(tens)


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
    def softmax_backward(arr): # arr will always be 1d array
      grads = -np.broadcast_to(arr, (arr.size, arr.size))
      np.fill_diagonal(grads, 1+(np.diagonal(grads)))
      grads *= arr.reshape(arr.size, 1)
      return np.dot(grads, arr)

    def grad_backward(ug):
      result = np.apply_along_axis(self.calc_softmax, self.axis, tens.data)
      local_grads = np.apply_along_axis(softmax_backward, self.axis, result)
      return local_grads*ug

    tens.set_grad_fn(grad_backward)

  def calc_softmax(self, arr):
    exponentiated = np.exp(arr-np.max(arr))
    sum_val = np.sum(exponentiated)
    return exponentiated/sum_val

def softmax(tens, axis):
  return Softmax(axis).forward(tens)


# <------------FLATTEN------------>

class Flatten(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    flattened = tens.data.flatten()
    return self.get_result_tensor(flattened.reshape(flattened.shape[0],1), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def flatten(tens):
  return Flatten().forward(tens)


# <------------RESHAPE------------>

class Reshape(Operation):
  def forward(self, tens, new_shape):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.reshape(new_shape), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def reshape(tens, new_shape):
  return Reshape().forward(tens, new_shape)


# <------------CONV2D------------>

class Conv2D(Operation):
  def __init__(self, padding=0, stride=1):
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs, kernel, bias):
    inputs, kernel, bias = self.get_tensors(inputs, kernel, bias)
    self.validate_inputs(inputs)
    num_examples = inputs.shape[0] # first dim should be number of examples
    result_shape = self.get_result_shape(inputs.shape, kernel.shape)
    outputs = np.empty((num_examples, *result_shape))
    padded_inputs = np.pad(inputs.data, ((0,0),(self.padding,self.padding),(self.padding,self.padding)))
    for i, (fragment, _, _) in enumerate(self.generate_fragments(padded_inputs, kernel.shape)):
      output = np.sum((fragment*kernel.data), axis=2)
      output = np.sum(output, axis=1) + bias.data
      row = math.floor(i/result_shape[1])
      col = i%result_shape[1]
      outputs[:,row,col] = output
    return self.get_result_tensor(outputs, inputs, kernel, bias)
  
  def backward(self, inputs, kernel, bias):
    from .utils import unbroadcast_data
    padded_inputs = np.pad(inputs.data, ((0,0),(self.padding,self.padding),(self.padding,self.padding)))

    def inputs_backward(ug):
      inputs_grads = np.zeros(padded_inputs.shape)
      for i, (fragment, row_slice, col_slice) in enumerate(self.generate_fragments(padded_inputs, kernel.shape)):
        sliced_ug = ug[:,row_slice.start,col_slice.start]
        sum_grad = np.ones(fragment.shape)*sliced_ug.reshape(sliced_ug.size,1,1)
        fragment_grad = kernel.data*sum_grad
        inputs_grads[:, row_slice, col_slice]+=fragment_grad
      unpadded_inputs_grads = self.unpad(inputs_grads)
      return unpadded_inputs_grads

    def kernel_backward(ug):
      kernel_grads = np.zeros(kernel.shape)
      for i, (fragment, row_slice, col_slice) in enumerate(self.generate_fragments(padded_inputs, kernel.shape)):
        sliced_ug = ug[:,row_slice.start,col_slice.start]
        sum_grad = np.ones(fragment.shape)*sliced_ug.reshape(sliced_ug.size,1,1)
        kernel_grad = unbroadcast_data(fragment*sum_grad, kernel.shape, fragment.shape)
        kernel_grads+=kernel_grad
      return kernel_grads

    def bias_backward(ug):
      return np.sum(ug)
      
    inputs.set_grad_fn(inputs_backward)
    kernel.set_grad_fn(kernel_backward)
    bias.set_grad_fn(bias_backward)
  
  def generate_fragments(self, inputs_data, kernel_shape):
    inputs_x_dim, inputs_y_dim = inputs_data.shape[1:]
    kernel_x_dim, kernel_y_dim = kernel_shape
    j = 0
    while(j+kernel_y_dim<=inputs_y_dim):
      i = 0
      while(i+kernel_x_dim<=inputs_x_dim):
        row_slice = slice(i, i+kernel_x_dim)
        col_slice = slice(j, j+kernel_y_dim)
        yield inputs_data[:, row_slice, col_slice], row_slice, col_slice
        i+=self.stride
      j+=self.stride
  
  def get_result_shape(self, inputs_shape, kernel_shape):
    inputs_x_dim, inputs_y_dim = inputs_shape[1:]
    kernel_x_dim, kernel_y_dim = kernel_shape
    def result_dim(inputs_dim, kernel_dim):
      return math.floor(((inputs_dim + (2*self.padding) - kernel_dim)/self.stride) + 1)
    result_x_dim = result_dim(inputs_x_dim, kernel_x_dim)
    result_y_dim = result_dim(inputs_y_dim, kernel_y_dim)
    return result_x_dim, result_y_dim
  
  def unpad(self, padded_data):
    padded_x_dim, padded_y_dim = padded_data.shape[1:]
    return padded_data[:, self.padding:padded_x_dim-self.padding, self.padding:padded_y_dim-self.padding]
  
  def validate_inputs(self, inputs):
    if len(inputs.shape)!=3: # The first dimension should be number of examples
      raise ValueError("Only 3D inputs, with 0th dim as number of examples are supported!")

def conv2d(inputs, kernel, bias, padding, stride):
  return Conv2D(padding, stride).forward(inputs, kernel, bias)