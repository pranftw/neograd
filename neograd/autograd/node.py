class Node:
  '''Used as an abstraction to connect the tensors together and hold relationships

  Each Tensor is assigned a Node and this Node monitors all the incoming edges(parents)
  and the outgoing edges(children)

  Parameters:
    children (list of Node): List of all Nodes which uses the current Node
      as an operand in an Operation
    parents (list of Node): List of all Nodes(operands) that has resulted in the creation
      of current Node
    parent_broadcast_shape (tuple or None): If the parent needs to be broadcasted from one shape to
      another, then the final broadcasted shape of the parent is stored here.
      If they cannot be broadcasted, then it is None
    backward_fn (Operation.backward): Sets the grad_fn of Tensor(operand) involved in the Operation
    visited (bool) - If Node is visited or not
  '''

  def __init__(self, tens):
    '''
      Args:
        tens (Tensor) - The Tensor corresponding to the Node
    '''
    self.tens = tens
    self.children = []
    self.parents = []
    self.parent_broadcast_shape = None
    self.backward_fn = None
    self.visited = False
  
  def top_sort(self):
    '''Performs topological sort of all Nodes starting from current Node

    Sorts the graph topologically, to perform backward pass efficiently, so that all the children's
    is calculated before the current node's gradient is calculated.

    Sorting is done by first checking if all the children are visited, if they are, then the current
    node is added to sorted_tensors if not, then topological sort is performed on children
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
    '''Initiates backward pass starting from current Node

    This first visits all the children to make sure that they aren't included in
    sorted_tensors as they aren't required as backward pass is being initiated from the current
    node.

    Then it pops its corresponding Tensor from sorted_tensors (it is the first tensor) so that
    _backward can be called on it with calculate_grads=False, so that grads arent calculated for
    it, but allows flushing of all Tensors

    Next it topologically sorts all Tensors starting from current Node then the Node
    corresponding to the Tensor is retreived, which is marked as visited and the Tensor's
    backward pass is initiated.

    Args:
      retain_graph (bool): If the graph should be retained after backward pass or flushed
        after backward calculation
    '''
    from .utils import get_graph
    graph = get_graph()
    graph.reset_visited()
    self.visit_all_children() # this allows for gradient calculation from any intermediate node in the graph
    sorted_tensors = self.top_sort()
    graph.reset_visited()

    sorted_tensors.pop(0) # Remove the Tensor corresponding to the current node
    self.visited = True
    self.tens._backward(self, retain_graph, calculate_grads=False)

    for tens in sorted_tensors:
      node = graph.get_node(tens)
      node.visited = True
      tens._backward(node, retain_graph)

  def visit_all_children(self):
    '''Marks all children as visited
    '''
    for child in self.children:
      child.visited = True

  def are_children_visited(self):
    '''Checks if all children are visited

    Returns:
      True if all children are visited else False
    '''
    for child in self.children:
      if not(child.visited):
        return False
    return True
  
  def are_parents_visited(self):
    '''Checks if all parents are visited

    Returns:
      True if all parents are visited else False
    '''
    for parent in self.parents:
      if not(parent.visited):
        return False
    return True
  
  def add_child(self, other):
    '''Adds a child to the Node

    Args:
      other (Node): The child Node
    '''
    self.children.append(other)
  
  def add_parent(self, other):
    '''Adds a parent to the Node

    Args:
      other (Node): The parent Node
    '''
    self.parents.append(other)
  
  def __repr__(self):
    return f'Node({self.tens})'
  
  def __str__(self):
    return f'Node( \n{self.tens}\nbackward_fn: {self.backward_fn}\nvisited: {self.visited}\n )'