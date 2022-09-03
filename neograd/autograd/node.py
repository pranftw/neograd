class Node:
  def __init__(self, operation):
    self.operation = operation

  def backward(self, upper_grad):
    if self.operation.needs_broadcasting:
      upper_grad = upper_grad.flatten()
    tensors = self.operation.tensors
    parent_upper_grads = {}
    for tens in tensors:
      if tens.requires_grad:
        grad = tens._backward(self.operation, upper_grad)
        if tens.node is not None:
          if tens.node not in parent_upper_grads.keys():
            parent_upper_grads[tens.node] = 0.0
          parent_upper_grads[tens.node]+=grad #Accumulate grads
    for tens in tensors:
      if tens.node is not None and tens.requires_grad:
        tens.node.backward(parent_upper_grads[tens.node])