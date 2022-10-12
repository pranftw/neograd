import numpy as np
from ..node import Node


class Operation:
  '''Transforms Tensors by applying some function

  Used when some input is getting transformed into an output, for functions
  where gradient calculation is required with the forward pass and the backward
  pass defined
  '''
  
  def process_operands(self, operands):
    '''All operands are converted to Tensors

    Args:
      operands (Tensor or int or float or list or np.ndarray): Operands of the Operation
    
    Returns:
      tuple of Tensors
    '''
    from ..tensor import Tensor
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
  def get_tensors(self, *operands):
    '''Returns the processed operands as tuple of Tensors

    Args:
      *operands (Tensor or int or float or list or np.ndarray): Operands of the Operation
    
    Returns:
      tuple of Tensors if len(tuple)>1 else returns the first Tensor
    '''
    tensors = self.process_operands(operands)
    if len(tensors)==0:
      return None
    elif len(tensors)==1:
      return tensors[0]
    else:
      return tensors
  
  def get_broadcast_shape(self, *tensors):
    '''Return broadcasted shape of Tensors

    If the tensors can be broadcasted, then the broadcasted shape is returned
    , else None.

    Args:
      *tensors (Tensor): Tensors that should be broadcasted

    Returns:
      Broadcasted shape if it can be broadcasted, if not None
      Also even if atleast one of the Tensors has requires_broadcasting set to False,
      it returns None
    '''
    for tens in tensors:
      if not(tens.requires_broadcasting):
        return None
    try:
      return np.broadcast_shapes(*(tens.data.shape for tens in tensors))
    except ValueError:
      return None
  
  def result_requires_grad(self, tensors):
    '''Checks if the result requires grad

    Checks if the result requires gradient to be calculated given the operands of the
    Operation, if atleast one operand requires_grad to True, then result will also have
    requires_grad to True

    Args:
      tensors (Tensor): Tensors that are operated on
    '''
    for tens in tensors:
      if tens.requires_grad:
        return True
    return False
  
  def get_result_tensor(self, result, *tensors):
    '''Returns the result tensor of the Operation
    
    If tracking is enabled, then, it creates a Node for the result_tensor
    with parent_broadcast_shape and adds edges to the graph
    
    If tracking is disabled, then no Node creation and edge addition
    occurs

    Args:
      result (np object): Result after performing a raw numpy operation
      *tensors (Tensor): Operands of the operation

    Returns:
      Tensor of the result
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
    '''Abstract backward method

    Raises:
      NotImplementedError: If backward method isn't overridden
    '''
    raise NotImplementedError(f"Backward method not implemented for Operation {self}")