import _setup
import numpy as np
import torch
import neograd as ng
from neograd.autograd.utils import process_data


def to_tensors(tens_cls, operands):
  '''
    Converts all operands of type numpy to tens_cls which can be torch.tensor or ng.tensor

    Params:
      tens_cls:torch.tensor/ng.tensor - The Tensor type to which the operands should be converted to
      operands:(np.ndarray)
    
    Returns:
      tensors:[Tensor]
  '''
  tensors = []
  for operand in operands:
    operand = operand.astype(float) # Torch only accepts floating point data for requires_grad=True
    tensors.append(tens_cls(operand, requires_grad=True))
  return tensors

def execute_neograd(fn, *operands, **kwargs):
  '''
    Executes the given function on the operands in neograd

    Params:
      fn:ng_operation
      operands:*args(np.ndarray)
      **kwargs - any keyword args to fn
    
    Returns:
      rounded_result:np.ndarray - Result rounded to 2 places
      grads:[np.ndarray] - Each array rounded to 2 places
  '''
  tensors = to_tensors(ng.tensor, operands)
  result = fn(*tensors, **kwargs)
  result.backward(np.ones(result.shape).astype(float))
  result_np = process_data(result.data)
  rounded_result = np.around(result_np, decimals=2)
  grads = []
  for tens in tensors:
    grads.append(np.around(tens.grad, decimals=2))
  return rounded_result, grads

def execute_torch(fn, *operands, **kwargs):
  '''
    Executes the given function on the operands in torch

    Params:
      fn:torch_operation
      operands:*args(np.ndarray)
      **kwargs - any keyword args to fn
    
    Returns:
      rounded_result:np.ndarray - Result rounded to 2 places
      grads:[np.ndarray] - Each array rounded to 2 places
  '''
  tensors = to_tensors(torch.tensor, operands)
  result = fn(*tensors, **kwargs)
  result.backward(torch.tensor(np.ones(result.shape).astype(float)))
  result_torch = process_data(result.detach().numpy())
  rounded_result = np.around(result_torch, decimals=2)
  grads = []
  for tens in tensors:
    grads.append(np.around(tens.grad.numpy(), decimals=2))
  return rounded_result, grads

def compare_outputs(ng_output, torch_output):
  '''
    Compares the outputs after execute_neograd and execute_torch
    Checks equality of respective results
    Checks equality of grads length
    Checks the equality of each grad in grads
  '''
  result_ng, grads_ng = ng_output
  result_torch, grads_torch = torch_output
  assert np.array_equal(result_ng, result_torch)
  assert len(grads_ng)==len(grads_torch)
  for grad_ng, grad_torch in zip(grads_ng, grads_torch):
    assert np.array_equal(grad_ng, grad_torch)


a = np.array(3)
b = np.array([1,2,3])
c = np.array([[3,4,5],[6,7,8]])
d = np.array([[[9,8,7],[6,5,4]], [[1,2,3],[4,5,6]]])
e = np.array([[1,2],[3,4]])


# <------------ADD------------>
def test_add():
  compare_outputs(execute_neograd(ng.add, a, b), execute_torch(torch.add, a, b))
  compare_outputs(execute_neograd(ng.add, a, d), execute_torch(torch.add, a, d))


# <------------SUB------------>
def test_sub():
  compare_outputs(execute_neograd(ng.sub, a, b), execute_torch(torch.sub, a, b))
  compare_outputs(execute_neograd(ng.sub, a, d), execute_torch(torch.sub, a, d))


# <------------MUL------------>
def test_mul():
  compare_outputs(execute_neograd(ng.mul, b, c), execute_torch(torch.mul, b, c))
  compare_outputs(execute_neograd(ng.mul, c, d), execute_torch(torch.mul, c, d))