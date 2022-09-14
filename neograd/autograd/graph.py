from .node import Node


class Graph:
  '''
    The graph that is constructed during the forward pass, and used by the backward
      pass to calculate gradients through automatic differentiation
  '''
  __slots__ = ['nodes_dict']
  
  def __init__(self):
    '''
      nodes_dict contains a Tensor as the key and its respective Node as the value
    '''
    self.nodes_dict = {}
  
  def add_edge(self, result_node, operands):
    '''
      Adds edges between the result_node, which is created during an Operation, and the
      operands that produced the result.

      Params:
        result_node:Node - node that is created in Operation.get_result_tensor
        operands:[Tensor] - All the operands for an Operation
    '''
    self.add_node(result_node)
    for operand in operands:
      if self.get_node(operand) is None:
        self.add_tensor(operand)
      operand_node = self.get_node(operand)
      result_node.add_parent(operand_node)
      operand_node.add_child(result_node)
  
  def add_node(self, node):
    '''
      Adds a Node to the graph

      Params:
        node:Node
    '''
    self.nodes_dict[node.tens] = node

  def get_node(self, tens):
    '''
      Returns the Node corresponding to the Tensor

      Params:
        tens:Tensor
    '''
    return self.nodes_dict.get(tens)
  
  def add_tensor(self, tens):
    '''
      Adds a Tensor to the graph. A new node is created for the Tensor

      Params:
        tens:Tensor
    '''
    self.nodes_dict[tens] = Node(tens)
  
  def remove_tensor(self, tens):
    '''
      Removes a Tensor from the graph

      Params:
        tens:Tensor
    '''
    self.nodes_dict.pop(tens)
  
  def reset_visited(self):
    '''
      Resets Node.visited for each Node in the graph
    '''
    for node in self.nodes_dict.values():
      node.visited = False
  
  def reset_graph(self):
    '''
      Resets the whole graph clears the nodes_dict
    '''
    self.nodes_dict = {}
  
  def __repr__(self):
    return 'Graph()'
  
  def __str__(self):
    return f'Graph( {self.nodes_dict} )'