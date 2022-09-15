class Node:
  '''
    Used as an abstraction to connect the graph together and hold relationships
  '''
  __slots__ = ['tens', 'children', 'parents', 'parent_broadcast_shape', 'parent_needs_broadcasting',
              'backward_fn', 'visited']

  def __init__(self, tens):
    '''
      Params:
        tens:Tensor - The Tensor corresponding to the Node
      
      children:[Node] - List of all Nodes which uses the current Node as an operand in an Operation
      parents:[Node] - List of all Nodes(operands) that has resulted in the creation of current Node
      parent_broadcast_shape:np.shape - If the parent needs to be broadcasted from one shape to
        another, then the final broadcasted shape of the parent is stored here
      parent_needs_broadcasting:Bool - Some Operation demands that broadcasting be done, ex add, but
        some like dot must not perform broadcasting
      backward_fn:Operation.backward - Sets the grad_fn of Tensor(operand) involved in the Operation
      visited:Bool - If Node is visited or not
    '''
    self.tens = tens
    self.children = []
    self.parents = []
    self.parent_broadcast_shape = None
    self.parent_needs_broadcasting = None
    self.backward_fn = None
    self.visited = False
  
  def top_sort(self):
    '''
      Sorts the graph topologically, to perform backward pass efficiently, so that all the children's
        is calculated before the current node's gradient is calculated.
      Sorting is done by first checking if all the children are visited, if they are then the current
        node is added to sorted_tensors if not then the top_sort is performed on children
    '''
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
    '''
      Iterated through all sorted_tensors and performs their backward calculation

      Params:
        retain_graph:Bool - If the graph should be retained after backward pass or flushed
    '''
    from .utils import get_graph
    graph = get_graph()
    graph.reset_visited()
    sorted_tensors = self.top_sort()
    graph.reset_visited()
    for tens in sorted_tensors:
      node = graph.get_node(tens)
      node.visited = True
      tens._backward(node, retain_graph)

  def visit_all_children(self):
    '''
      All children are visited
      This will be used when Tensor.backward is calculated for a Tensor that is not the leaf TODO
    '''
    for child in self.children:
      child.visited = True

  def are_children_visited(self):
    '''
      Checks if all children are visited
    '''
    for child in self.children:
      if not(child.visited):
        return False
    return True
  
  def are_parents_visited(self):
    '''
      Checks if all parents are visited
    '''
    for parent in self.parents:
      if not(parent.visited):
        return False
    return True
  
  def add_child(self, other):
    '''
      Adds a child to the Node
    '''
    self.children.append(other)
  
  def add_parent(self, other):
    '''
      Adds a parent to the Node
    '''
    self.parents.append(other)
  
  def __repr__(self):
    return f'Node({self.tens})'
  
  def __str__(self):
    return f'Node( \n{self.tens}\nbackward_fn: {self.backward_fn}\nvisited: {self.visited}\n )'