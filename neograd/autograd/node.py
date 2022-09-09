from .utils import unflatten_data


class Node:
  __slots__ = ['operands', 'children', 'needs_broadcasting', 'visited']

  def __init__(self):
    self.operands = [] # can this be weakref
    self.children = []
    self.needs_broadcasting = True
    self.visited = False
  
  def top_sort(self):
    sorted_tensors = []
    if self.are_children_visited(): # All children are resolved
      self.visited = True
      sorted_tensors.append(self)
      for operand in self.operands:
        if not(operand.visited):
          sorted_tensors+=operand.top_sort()
    else:
      for child in self.children: # Resolve children first
        if not(child.visited):
          sorted_tensors+=child.top_sort()
      self.visited = False
      sorted_tensors.append(self)
    return sorted_tensors
  
  def add_child(self, child):
    self.children.append(child)
  
  def add_operand(self, operand):
    self.operands.append(operand)
  
  def node_backward(self):
    self.reset_visited()
    self.visit_all_children()
    sorted_tensors = self.top_sort()
    self.reset_visited()
    self.visit_all_children()
    for tens in sorted_tensors:
      tens.visited = True
      if tens.requires_grad:
        upper_grad = tens.grad
        tens._backward(upper_grad)
  
  def _backward(self, upper_grad):
    if len(self.operands)!=0:
      if self.needs_broadcasting:
        upper_grad = upper_grad.flatten()
      self.backward_fn()
      for operand in self.operands:
        if operand.requires_grad and operand.are_children_visited():
          operand.visited = True
          grad = operand.grad_fn(upper_grad)
          grad = unflatten_data(grad, operand.shape, self.operand_broadcast_shape)
          grad = grad.reshape(operand.shape)
          operand.grad+=grad
  
  def visit_all_children(self):
    for child in self.children:
      child.visited = True
  
  def are_children_visited(self):
    for child in self.children:
      if not(child.visited):
        return False
    return True

  def reset_visited(self):
    self.visited = False
    for operand in self.operands:
      operand.visited = False
  
  def __str__(self):
    return f"Node(\ninputs:{', '.join([tens.__str__() for tens in self.operation.tensors])}\noutputs:{self.operation.result_tensor}\n)"
  
  def __repr__(self):
    return f"Node({self.operation})"