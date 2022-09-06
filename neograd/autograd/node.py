from .utils import unflatten_data


class Node:
  def __init__(self, operation):
    self.operation = operation
    self.children = []
    self.is_visited = False
  
  def add_child(self, child):
    self.children.append(child)
  
  def top_sort(self, is_start=False):
    sorted_nodes = []
    if len(self.children)==0 or self.check_if_all_children_visited() or is_start: # children are all resolved, add the current node and go to its operands or this is the starting node
      self.is_visited = True        
      sorted_nodes.append(self)
      for tens in self.operation.tensors:
        if tens.node is not None:
          sorted_nodes+=tens.node.top_sort()
    else: # First resolve the children, then add current node
      for child in self.children:
        if not(child.is_visited):
          sorted_nodes+=child.top_sort()
      self.is_visited = True
      sorted_nodes.append(self)
    return sorted_nodes
  
  def reset_is_visited(self):
    self.is_visited = False
    for tens in self.operation.tensors:
      if tens.node is not None:
        tens.node.reset_is_visited()
  
  def check_if_all_children_visited(self):
    for child in self.children:
      if not(child.is_visited):
        return False
    return True
  
  def backward(self):
    self.reset_is_visited()
    sorted_nodes = self.top_sort(True)
    self.reset_is_visited()
    for i,node in enumerate(sorted_nodes):
      is_start = True if i==0 else False
      if node.operation.result_tensor.requires_grad:
        node._backward(is_start) # This is where the damn issue is..... WTF
  
  def _backward(self, is_start=False):
    if is_start or self.check_if_all_children_visited():
      self.operation.backward()
      upper_grad = self.operation.result_tensor.grad
      if self.operation.needs_broadcasting:
        upper_grad = upper_grad.flatten()
      broadcast_shape = self.operation.broadcast_shape
      for tens in self.operation.tensors:
        if tens.requires_grad:
          tens._backward(upper_grad, broadcast_shape)
      self.is_visited = True
  
  def __str__(self):
    return f"Node(\ninputs:{', '.join([tens.__str__() for tens in self.operation.tensors])}\noutputs:{self.operation.result_tensor}\n)"
  
  def __repr__(self):
    return f"Node({self.operation})"