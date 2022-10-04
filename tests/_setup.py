import sys
sys.path.append('..')
import neograd as ng
from neograd.autograd.utils import process_data, fn_grad_check


def to_tensors(operands):
  '''Converts all operands of type numpy to ng.tensor

  Args:
    operands (list of np.ndarray): Operands to be converted
  
  Returns:
    tuple of Tensors
  '''
  tensors = []
  for operand in operands:
    operand = operand.astype(float)
    tensors.append(ng.tensor(operand, requires_grad=True))
  return tensors

def execute(fn, operands, params=None, epsilon=1e-7, tolerance=1e-7, **kwargs):
  '''Performs the grad_check on the fn using fn_grad_check

  Args:
    fn: Function to be gradient checked
    operands (list of np.ndarray): Operands to the function
    params (list of Param): Any other params involved with the function that needs to be checked
      Defaults to None. If not None, then it is combined with the tensors
    epsilon (float): value of epsilon
    tolerance (float): Lower tolerance acceptable Defaults to 1e-7 tolerance
      is required in cases like Linear layer where the backprop is calculated by
      autograd, without any explicit gradient fns, so there might be some floating point
      truncations that could result in slight deviation that is acceptable
    **kwargs: kwargs to be passed to fn
  
  Raises:
    AssertionError: If distance isn't less than tolerance, then it has failed
  '''
  tensors = to_tensors(operands)
  params_to_be_tested = tensors if params is None else (params+tensors)
  dist = fn_grad_check(fn, tensors, params_to_be_tested, epsilon=epsilon, print_vals=False, **kwargs)
  assert dist<tolerance