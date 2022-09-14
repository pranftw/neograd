class Node:
  __slots__ = ['tens', 'children', 'parents', 'parent_broadcast_shape', 'needs_broadcasting',
              'backward_fn', 'visited']

  def __init__(self, tens):
    self.tens = tens
    self.children = []
    self.parents = []
    self.parent_broadcast_shape = None
    self.needs_broadcasting = None
    self.backward_fn = None
    self.visited = False
  
  def top_sort(self):
    sorted_tensors = []
    if self.are_children_visited():
      self.visited = True
      sorted_tensors.append(self.tens)
      for parent in self.parents:
        if not(parent.visited):
          sorted_tensors+=parent.top_sort()
    else:
      for child in self.children:
        if not(child.visited):
          sorted_tensors+=child.top_sort()
    return sorted_tensors
  
  def backward(self, retain_graph):
    from .. import _NG_GRAPH
    _NG_GRAPH.reset_visited()
    sorted_tensors = self.top_sort()
    _NG_GRAPH.reset_visited()
    for tens in sorted_tensors:
      node = _NG_GRAPH.get_node(tens)
      node.visited = True
      tens._backward(node, retain_graph)

  def visit_all_children(self):
    for child in self.children:
      child.visited = True

  def are_children_visited(self):
    for child in self.children:
      if not(child.visited):
        return False
    return True
  
  def are_parents_visited(self):
    for parent in self.parents:
      if not(parent.visited):
        return False
    return True
  
  def add_child(self, other):
    self.children.append(other)
  
  def add_parent(self, other):
    self.parents.append(other)
  
  def __repr__(self):
    return f'Node({self.tens})'
  
  def __str__(self):
    return f'Node( \n{self.tens}\nbackward_fn: {self.backward_fn}\nvisited: {self.visited}\n )'