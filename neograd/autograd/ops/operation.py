import numpy as np
from ..node import Node


class Operation:
  '''
    Used when some input is getting transformed into an output, for functions
      where gradient calculation is required with the forward pass and the backward
      pass defined
  '''

  graph = None
  
  def process_operands(self, operands):
    '''
      All operands are converted to Tensor

      Params:
        operands:(any class that is supported/Tensor)
    '''
    from ..tensor import Tensor
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_tensors(self, *operands):
    '''
      Returns the processed operands as tuple of Tensors

      Params:
        operands:*args(any class that is supported/Tensor)
    '''
    tensors = self.process_operands(operands)
    if len(tensors)==0:
      return None
    elif len(tensors)==1:
      return tensors[0]
    else:
      return tensors
  
  def get_broadcast_shape(self, *tensors):
    '''
      If the tensors can be broadcasted, then the broadcasted
        shape is returned, else None
      
      Params:
        tensors:*args(Tensor)
    '''
    for tens in tensors:
      if not(tens.requires_broadcasting):
        return None
    try:
      return np.broadcast_shapes(*(tens.data.shape for tens in tensors))
    except ValueError:
      return None
  
  def result_requires_grad(self, tensors):
    '''
      Checks if the result requires_grad given the operands of the Operation, if atleast
        one operand requires_grad, then result will also have requires_grad
      
      Params:
        tensors:(Tensor)
    '''
    for tens in tensors:
      if tens.requires_grad:
        return True
    return False
  
  def get_result_tensor(self, result, *tensors):
    '''
      Returns the result tensor of the Operation
      Creates a Node for the result_tensor with parent_broadcast_shape and parent_needs_broadcasting
      Adds the edges to the graph

      Params:
        result:np object - Result after performing a raw numpy operation
        tensors:*args(Tensor)
    '''
    from ..tensor import Tensor
    from ..utils import get_graph
    graph = get_graph()
    result = result.astype(np.ndarray)
    result_tensor = Tensor(result, self.result_requires_grad(tensors))
    if graph.track:
      result_node = Node(result_tensor)
      result_node.backward_fn = self.backward
      result_node.parent_broadcast_shape = self.get_broadcast_shape(*tensors)
      graph.add_edge(result_node, tensors)
    return result_tensor
  
  def backward(self, *args):
    raise NotImplementedError(f"Backward method not implemented for Operation {self}")