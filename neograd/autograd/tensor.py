from .utils import process_data, unbroadcast_data
from .ops import add, sub, mul, div, pow as _pow, transpose, sum as _sum, exp, dot


class Tensor:
  __slots__ = ['_data', 'requires_grad', 'grad', 'grad_fn']
  '''
    The main computational element which abstracts the np.ndarray used to
      perform all the Operation
  '''

  def __init__(self, data, requires_grad=False):
    '''
      Params:
        data:int/float/np.ndarray/[] - Data to be stored in the tensor
        requires_grad:Bool - If a Tensor requires gradient to be calculated for it or not
      
      grad:np.ndarray - The gradient for the current Tensor
      grad_fn - takes the gradient from above, uses the local gradient to give the gradient
    '''
    self.data = data
    self.requires_grad = requires_grad
    self.grad = 0. if requires_grad else None
    self.grad_fn = None
  
  def zero_grad(self):
    '''
      Resets the grad of the Tensor
    '''
    self.grad = 0. if self.requires_grad else None
  
  def backward(self, upper_grad=1., retain_graph=False):
    '''
      Starts the gradient calculation for the backward pass from the current
        Tensor
      
      Params:
        upper_grad:int/float/np.ndarray/[] - The gradient with which to start the
          gradient calculation
        retain_graph:Bool - If the graph should be retained after backward pass or flushed
    '''
    if not(self.requires_grad):
      raise ValueError("Only tensors who requires_grad can call backward")
    from .utils import get_graph
    graph = get_graph()
    upper_grad = process_data(upper_grad)
    if self.shape!=upper_grad.shape:
      raise ValueError("Shapes of grad and Tensor data must match!")
    self.grad+=upper_grad
    node = graph.get_node(self)
    node.backward(retain_graph)
    if not(retain_graph):
      graph.reset_graph() # tensors are auto-removed, this is just for redundancy / safety
  
  def _backward(self, node, retain_graph):
    '''
      The essence of autograd, final gradient calculations for the Tensor is performed here
      For each child, its gradient is taken as upper gradient, the backward function of the 
        Node of the current Tensor is executed to set the grad_fn of current Tensor
      grad_fn is executed, the grad is then unbroadcasted, if Tensor has been broadcasted
        during the Operation
      auto-removal of Tensor from the graph is performed when retain_graph is False
    '''
    from .utils import get_graph
    graph = get_graph()
    for child in node.children:
      if self.requires_grad:
        child.backward_fn(*[node.tens for node in child.parents])
        upper_grad = child.tens.grad
        if child.parent_needs_broadcasting:
          upper_grad = upper_grad.flatten()
        grad = self.grad_fn(upper_grad)
        grad = unbroadcast_data(grad, self.shape, child.parent_broadcast_shape)
        grad = grad.reshape(self.shape)
        self.grad+=grad
      if not(retain_graph) and child.are_parents_visited():
        graph.remove_tensor(child.tens)
    if not(retain_graph) and node.are_parents_visited():
      graph.remove_tensor(node.tens)
  
  def set_grad_fn(self, grad_fn):
    '''
      Sets the grad_fn for the Tensor
      if it doesn't require_grad it is set to None
    '''
    if self.requires_grad:
      self.grad_fn = grad_fn
    else:
      self.grad_fn = None

  def __add__(self, other):
    return add(self, other)
  
  def __radd__(self, other):
    return add(other, self)
  
  def __sub__(self, other):
    return sub(self, other)
  
  def __rsub__(self, other):
    return sub(other, self)
  
  def __mul__(self, other):
    return mul(self, other)
  
  def __rmul__(self, other):
    return mul(other, self)
  
  def __truediv__(self, other):
    return div(self, other)
  
  def __rtruediv__(self, other):
    return div(other, self)
  
  def __pow__(self, other):
    return _pow(self, other)
  
  def __rpow__(self, other):
    return _pow(other, self)
  
  def __pos__(self):
    return self
  
  def __neg__(self):
    return (-1*self)
  
  def dot(self, other):
    return dot(self, other)
  
  def sum(self, axis=None):
    return _sum(self, axis)
  
  def exp(self):
    return exp(self)
  
  @property
  def data(self):
    return self._data
  
  @data.setter
  def data(self, data):
    self._data = process_data(data)

  @property
  def shape(self):
    return self.data.shape
  
  @property
  def T(self):
    return transpose(self)
  
  def __getitem__(self, index):
    supported_types = [int, slice]
    if type(index) not in supported_types:
      raise TypeError(f"Expected index of {supported_types} instead got {type(index)}")
    return Tensor(self.data[index], requires_grad=self.requires_grad)
  
  def __repr__(self):
    return f'Tensor({self.data}, requires_grad={self.requires_grad})'
  
  def __str__(self):
    return f'Tensor( {self.data},\n requires_grad={self.requires_grad},\n grad_fn={self.grad_fn},\n shape={self.shape} )\n'