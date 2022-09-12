class Node:
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
      self.visited = True
      sorted_tensors.append(self.tens)
    return sorted_tensors
  
  def backward(self):
    from .. import _NG_GRAPH
    _NG_GRAPH.reset_visited()
    sorted_tensors = self.top_sort()
    assert len(sorted_tensors)==len(_NG_GRAPH.nodes_dict.keys()) # Here these two arent the same, hence resulting in error while trying to remove tensors
    _NG_GRAPH.reset_visited()
    for tens in sorted_tensors:
      if tens.requires_grad:
        tens._backward()

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