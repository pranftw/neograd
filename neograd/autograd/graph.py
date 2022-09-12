from .node import Node


class Graph:
  def __init__(self):
    self.nodes_dict = {} # Tensor is the key, its Node is the value
  
  def add_edge(self, result_node, operands):
    self.add_node(result_node)
    for operand in operands:
      if self.get_node(operand) is None:
        self.add_tensor(operand)
      operand_node = self.get_node(operand)
      result_node.add_parent(operand_node)
      operand_node.add_child(result_node)
  
  def add_node(self, node):
    self.nodes_dict[node.tens] = node

  def get_node(self, tens):
    return self.nodes_dict.get(tens)
  
  def add_tensor(self, tens):
    self.nodes_dict[tens] = Node(tens)
  
  def remove_tensor(self, tens):
    from ..nn.layers import Param
    node = self.nodes_dict.get(tens)
    del node
    self.nodes_dict.pop(tens)
    if not(isinstance(tens, Param)):
      del tens
  
  def zero_grad(self):
    for tens in self.nodes_dict:
      tens.zero_grad()
  
  def reset_visited(self):
    for node in self.nodes_dict.values():
      node.visited = False
  
  def reset_graph(self):
    self.nodes_dict = {}