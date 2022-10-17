from .utils import process_data, unbroadcast_data
from .ops import add, sub, mul, div, pow as _pow, transpose, sum as _sum, exp, dot, flatten, reshape


class Tensor:
  '''Wrapper around NumPy arrays
  
  Parameters:
    data (int or float or list or np.ndarray): The data to be stored or manipulated
    requires_grad (bool): Whether the Tensor requires gradient to be calculated or not
      Defaults to False
    requires_broadcasting (bool): Whether the Tensor needs to be broadcasted when some Operation
      is performed. Defaults to True. This attribute is present as there are some operations like
      Convolution for which the kernel shouldn't be broadcasted to inputs shape
    grad (np.ndarray): The gradient value of the Tensor. Defaults to 0 if requires_grad else None
    grad_fn: The function that is set in Operation.backward, that'll be executed during backward
      pass to set the gradient of the Tensor
  '''

  def __init__(self, data, requires_grad=False, requires_broadcasting=True):
    '''
    Args:
      data (int or float or list or np.ndarray): The data to be stored or manipulated
      requires_grad (bool): Whether the Tensor requires gradient to be calculated or not
        Defaults to False
      requires_broadcasting (bool): Whether the Tensor needs to be broadcasted when some Operation
        is performed. Defaults to True. This attribute is present as there are some operations like
        Convolution for which the kernel shouldn't be broadcasted to inputs shape
    
    Raises:
      TypeError: if data doesn't belong to (int or float or list or np.ndarray)
    '''
    self.data = data
    self.requires_grad = requires_grad
    self.requires_broadcasting = requires_broadcasting
    self.grad = 0. if requires_grad else None
    self.grad_fn = None
  
  def zero_grad(self):
    '''Resets the grad of the Tensor to the defaults
    '''
    self.grad = 0. if self.requires_grad else None
  
  def backward(self, upper_grad=1., retain_graph=False):
    '''Kicks off the backward pass to calculate gradients

    Starts the gradient calculation for the backward pass from the
    Tensor, by calling the backward method of its corresponding Node
      
    Args:
      upper_grad (int or float or list or np.ndarray): The gradient with which to start the
        gradient calculation. Shape of upper_grad and shape of Tensor must be the same.
        Defaults to 1 as usually backward is called on a loss Tensor that has a scalar value
      retain_graph (bool): If the graph should be retained after backward pass or should be reset.
        Auto-removal of Tensors from the graph, which happens when the gradients of all tensors of
        node's parents have been calculated will be turned off.
    
    Raises:
      ValueError: If called on a Tensor that doesn't have requires_grad
      ValueError: If shapes of upper_grad and Tensor doesn't match
    '''
    if not(self.requires_grad):
      raise ValueError("Only tensors who requires_grad can call backward")
    from .utils import get_graph
    graph = get_graph()
    upper_grad = process_data(upper_grad)
    if self.shape!=upper_grad.shape:
      raise ValueError("Shapes of grad and Tensor data must match!")
    self.accumulate_grad(upper_grad) # Setting the grad of the current Tensor by adding the upper_grad
    node = graph.get_node(self)
    node.backward(retain_graph)
    if not(retain_graph):
      graph.reset_graph() # tensors are auto-removed, this is just for redundancy / safety
  
  def _backward(self, node, retain_graph, calculate_grads=True):
    '''The essence of autograd, final gradient calculations for the Tensor is performed here

    The gradient of each child is taken as upper gradient, the backward_fn of the 
    Node of the Tensor is executed to set the grad_fn of Tensor.

    grad_fn is executed, the grad is then unbroadcasted, if Tensor has been broadcasted
    during the Operation. auto-removal of Tensor from the graph is performed when
    retain_graph is False

    Args:
      node (Node): The Node corresponding to the Tensor
      retain_graph (bool): Whether the graph needs to be retained or reset
      calculate_grads (bool): Whether gradients should be calculated or not,
        Defaults to True
    '''
    from .utils import get_graph
    graph = get_graph()
    for child in node.children:
      if self.requires_grad and calculate_grads:
        child.backward_fn(*[node.tens for node in child.parents])
        upper_grad = child.tens.grad
        grad = self.grad_fn(upper_grad)
        grad = unbroadcast_data(grad, self.shape, child.parent_broadcast_shape)
        grad = grad.reshape(self.shape)
        self.accumulate_grad(grad)
      if not(retain_graph) and child.are_parents_visited():
        graph.remove_tensor(child.tens)
    if not(retain_graph) and node.are_parents_visited():
      graph.remove_tensor(node.tens)
  
  def set_grad_fn(self, grad_fn):
    '''Sets the grad_fn for the Tensor

    If requires_grad is True, then Tensor.grad_fn is set to grad_fn else None

    Args:
      grad_fn: Function that is set during execution of Operation.backward
    '''
    self.grad_fn = grad_fn if self.requires_grad else None

  def __add__(self, other):
    '''Performs element wise addition of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be added with
    
    Returns:
      Tensor of the result
    '''
    return add(self, other)
  
  def __radd__(self, other):
    '''Performs element wise addition of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be added with
    
    Returns:
      Tensor of the result
    '''
    return add(other, self)
  
  def __sub__(self, other):
    '''Performs element wise subtraction of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be subtracted from
    
    Returns:
      Tensor of the result
    '''
    return sub(self, other)
  
  def __rsub__(self, other):
    '''Performs element wise subtraction of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be subtracted from
    
    Returns:
      Tensor of the result
    '''
    return sub(other, self)
  
  def __mul__(self, other):
    '''Performs element wise multiplication of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be multiplied with
    
    Returns:
      Tensor of the result
    '''
    return mul(self, other)
  
  def __rmul__(self, other):
    '''Performs element wise multiplication of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be multiplied with
    
    Returns:
      Tensor of the result
    '''
    return mul(other, self)
  
  def __truediv__(self, other):
    '''Performs element wise division of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be divided with
    
    Returns:
      Tensor of the result
    '''
    return div(self, other)
  
  def __rtruediv__(self, other):
    '''Performs element wise division of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be divided with
    
    Returns:
      Tensor of the result
    '''
    return div(other, self)
  
  def __pow__(self, other):
    '''Performs element wise power of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be raised to
    
    Returns:
      Tensor of the result
    '''
    return _pow(self, other)
  
  def __rpow__(self, other):
    '''Performs element wise power of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be raised to
    
    Returns:
      Tensor of the result
    '''
    return _pow(other, self)
  
  def __pos__(self):
    '''Performs unary plus on the Tensor
    
    Returns:
      Tensor of the result
    '''
    return (1*self)
  
  def __neg__(self):
    '''Performs unary minus on the Tensor
    
    Returns:
      Tensor of the result
    '''
    return (-1*self)
  
  def dot(self, other):
    '''Performs dot product of Tensor with another object

    Args:
      other (int or float or list or np.ndarray): The object that needs to be dotted with
    
    Returns:
      Tensor of the result
    '''
    return dot(self, other)
  
  def sum(self, axis=None):
    '''Performs sum of Tensor along an axis

    Args:
      axis (None or int): The axis along which it should be summed
    
    Returns:
      Tensor of the result
    '''
    return _sum(self, axis)
  
  def exp(self):
    '''Performs exponentiation on the Tensor

    Returns:
      Tensor of the result
    '''
    return exp(self)
  
  def flatten(self):
    '''Flattens the Tensor from any dimension to 1D

    Returns:
      Flattened Tensor
    '''
    return flatten(self)
  
  def reshape(self, new_shape):
    '''Reshapes the Tensor to the new shape

    Args:
      new_shape (tuple): The shape to which the Tensor should be reshaped to
    
    Returns:
      Reshaped Tensor
    '''
    return reshape(self, new_shape)
  
  def accumulate_grad(self, grad):
    '''Accumulates gradients for the Tensor

    Adds the gradient calculated with the overall gradient of the Tensor because
    if gradients are flowing into a Tensor from two different paths, they need to be
    summed up

    Args:
      grad (np.ndarray): The gradient to be added/accumulated
    '''
    self.grad+=grad
  
  @property
  def data(self):
    '''Returns the data present in the Tensor

    Returns:
      data (np.ndarray): Data in the Tensor
    '''
    return self._data
  
  @data.setter
  def data(self, data):
    '''Sets the data to the Tensor

    Args:
      data (int or float or list or np.ndarray): Data to be set
    
    Raises:
      TypeError: If data is not instance of (int or float or list or np.ndarray)
    '''
    self._data = process_data(data)

  @property
  def shape(self):
    '''Returns the shape of the Tensor

    Returns:
      Shape of data in the Tensor
    '''
    return self.data.shape
  
  @property
  def T(self):
    '''Performs the transpose of the Tensor

    Returns:
      Transpose of the Tensor
    '''
    return transpose(self)
  
  def __getitem__(self, *indices):
    '''Performs the slice of the data in the Tensor

    Args:
      index (tuple of int and/or slice): The indices to be sliced along or indexed
    
    Returns:
      Tensor with sliced data
    '''
    supported_types = (int, slice)
    for index in indices:
      if type(index) not in supported_types:
        raise TypeError(f"Expected index of {supported_types} instead got {type(index)}")
    return Tensor(self.data[indices], requires_grad=self.requires_grad)
  
  def __repr__(self):
    return f'Tensor({self.data}, requires_grad={self.requires_grad})'
  
  def __str__(self):
    return f'Tensor( {self.data},\n requires_grad={self.requires_grad},\n grad_fn={self.grad_fn},\n shape={self.shape} )\n'