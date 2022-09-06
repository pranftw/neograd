from .utils import process_data, unflatten_data
from .ops import add, sub, mul, div, pow as _pow, transpose, sum as _sum, exp, dot


class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = process_data(data)
    self.requires_grad = requires_grad
    self.node = None
    self.grad_fn = None
    self.grad = 0.0 if requires_grad else None
  
  def zero_grad(self):
    self.grad = 0.0
  
  def backward(self, upper_grad=1.0):
    self.grad = process_data(upper_grad)
    self.node.backward()
  
  def _backward(self, upper_grad, broadcast_shape):
    grad = self.grad_fn(upper_grad)
    grad = unflatten_data(grad, self.shape, broadcast_shape)
    grad = grad.reshape(self.shape)
    self.grad+=grad
  
  def set_grad_fn(self, grad_fn):
    if self.requires_grad:
      self.grad_fn = grad_fn
  
  @property
  def shape(self):
    return self.data.shape
  
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
    return f'Tensor( {self.data},\n requires_grad={self.requires_grad},\n grad_fn={self.grad_fn} )'