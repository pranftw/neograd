from .node import Node


class Graph:
  '''Used to keep track of nodes and tensors

  The graph is constructed during the forward pass, and used by the backward
  pass to calculate gradients through automatic differentiation

  Parameters:
    graph (Graph or None): Graph object that's currently in use. If None, then the global
      _NG_GRAPH is used, else a specific graph object is used. Defaults to None
    nodes_dict (dict): Stores key-value pairs of tensors and their corresponding
      nodes in the graph
    track (bool): Whether the graph must track the tensor operations or not, ie if True, when any
      operation happens and a new result tensor is created, then the operands of the operation
      are added as parents to the result tensor and the result tensor is added as child to the
      operands, if False, none of these happens. Defaults to True
  '''

  graph = None
  
  def __init__(self):
    '''Initializes the nodes_dict to empty dict, track to True
    '''
    self.nodes_dict = {}
    self.track = True
  
  def add_edge(self, result_node, operands):
    '''Creates an edge between two nodes

    Adds edges between the result_node, which is created during an Operation, and the
    operands that produced the result. This means the result_node is added as a child of
    each of the operands and the result_node adds all operands as its parents

    Args:
      result_node (Node): node that is created in Operation.get_result_tensor
      operands (list of Tensor): All the operands for an Operation
    '''
    self.add_node(result_node)
    for operand in operands:
      if self.get_node(operand) is None:
        self.add_tensor(operand)
      operand_node = self.get_node(operand)
      result_node.add_parent(operand_node)
      operand_node.add_child(result_node)
  
  def add_node(self, node):
    '''Adds a Node to the graph

    Creates an key-value pair in nodes_dict with the specified node as the value
    and its tens attribute as the key

    Args:
      node (Node): Node to be added to the graph
    '''
    self.nodes_dict[node.tens] = node

  def get_node(self, tens):
    '''Returns the Node corresponding to the Tensor

    Args:
      tens (Tensor): Tensor whose node is to be fetched
    
    Returns:
      Node if found, else None
    '''
    return self.nodes_dict.get(tens)
  
  def add_tensor(self, tens):
    '''Adds a Tensor to the graph
    
    A new node is created for the Tensor and corresponding entry is made
    in nodes_dict

    Args:
      tens (Tensor): Tensor to be added
    '''
    self.nodes_dict[tens] = Node(tens)
  
  def remove_tensor(self, tens):
    '''Removes a Tensor from the graph

    Pops the Tensor from nodes_dict

    Args:
      tens (Tensor): Tensor to be removed
    '''
    self.nodes_dict.pop(tens)
  
  def reset_visited(self):
    '''Sets visited=False for each Node in the graph
    '''
    for node in self.nodes_dict.values():
      node.visited = False
  
  def reset_graph(self):
    '''Resets the whole graph

    This is accomplished by setting nodes_dict to an empty dictionary
    Doing so, removes all the Tensors and their Nodes from the graph
    '''
    self.nodes_dict = {}
  
  def zero_grad(self):
    '''Performs zero_grad on all the tensors in the graph

    Iterates through nodes_dict and performs zero_grad on the tensors
    '''
    for tensor in self.nodes_dict.keys():
      tensor.zero_grad()
  
  def __repr__(self):
    return 'Graph()'
  
  def __str__(self):
    return f'Graph( {self.nodes_dict} )'