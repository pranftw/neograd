import numpy as np
from .ops.operation import Operation
from .graph import Graph
from itertools import zip_longest


def process_data(data):
  '''
    Processes the data that is stored in Tensor
    Supprted types for data - [int, float, list, np.ndarray]
    Elements in data should be float or be typecastable to float
  '''
  supported_types = [int, float, list, np.ndarray]
  if type(data) in supported_types:
    if not isinstance(data, np.ndarray):
      data = np.array(data)
    try:
      data = data.astype(float)
    except ValueError:
      raise TypeError("Elements of data should be of type float or be typecastable to float")
  else:
    raise TypeError(f"Expected data of types {supported_types} instead got {type(data)}")
  return data

def unbroadcast_data(data, orig_data_shape, broadcasted_shape):
  '''
    if data(a np object) is broadcasted during an operation, then it is unbroadcasted here
      where all dimensions where it was broadcasted are summed along that dimension to
      give the original shape of the data
  '''
  if broadcasted_shape is not None:
    dims_to_be_summed = get_dims_to_be_summed(orig_data_shape, broadcasted_shape)
    unbroadcasted_data = data.reshape(broadcasted_shape)
    for i,dim in reversed(list(enumerate(dims_to_be_summed))):
      if dim:
        unbroadcasted_data = np.sum(unbroadcasted_data, axis=i)
  else:
    unbroadcasted_data = data
  return unbroadcasted_data

def get_dims_to_be_summed(orig_data_shape, broadcasted_shape):
  '''
    True is given if it has been broadcasted along that dimension, False if not
    https://numpy.org/doc/stable/user/basics.broadcasting.html
  '''
  dims_to_be_summed = []
  zipped = zip_longest(tuple(reversed(broadcasted_shape)), tuple(reversed(orig_data_shape)), fillvalue=None)
  for dim_broadcasted, dim_orig in reversed(list(zipped)):
    if dim_broadcasted!=dim_orig:
      dims_to_be_summed.append(True)
    else:
      dims_to_be_summed.append(False)
  return dims_to_be_summed

def get_graph():
  '''
    Returns graph present in Operation.graph, if it is None, then the global graph _NG_GRAPH
      is used
  '''
  if Operation.graph is None:
    from .. import _NG_GRAPH
    graph = _NG_GRAPH
  else:
    graph = Operation.graph
  return graph


class new_graph:
  '''
    Context Manager to create a new graph if required within an operation or
      anywhere and shouldn't interfere with the global _NG_GRAPH
  '''
  def __enter__(self):
    Operation.graph = Graph()
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    Operation.graph = None


class no_track:
  '''
    Context Manager to prevent creation of a backward graph, when gradient calculation
      is not required, for ex when testing a model after training it, you don't need
      any backward pass
  '''
  def __init__(self):
    self.graph = get_graph()

  def __enter__(self):
    self.graph.track = False
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.graph.track = True


def _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals):
  dist = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
  if print_vals:
    print("Gradient Check Distance:", dist)
    if dist<epsilon:
      print("Gradient Check PASSED")
    else:
      print("Gradient Check FAILED")
  return dist


def _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon):
  for param in params:
    if param.requires_grad:
      if not(isinstance(param.grad, np.ndarray)):
        param.grad = np.array(param.grad)
      for idx in np.ndindex(param.shape):
        with no_track():
          param.data[idx]+=epsilon # PLUS
          loss1 = get_loss()
          param.data[idx]-=(2*epsilon) # MINUS
          loss2 = get_loss()
          param.data[idx]+=epsilon # ORIGINAL
        calculated_grads.append(param.grad[idx])
        analytical_grads.append((loss1.data-loss2.data)/(2*epsilon))
    param.zero_grad()


def grad_check(model, inputs, targets, loss_fn, epsilon=1e-7, print_vals=True):
  '''
    Implements Gradient Check, to make sure that backprop is calculating
      the right gradients.
    If distance between backprop gradients and numerical gradients is less
      than epsilon, then the gradients are proper, if not there is
      an issue
    
    Params:
      model:Model - The Neural Network to be evaluated
      inputs:Tensor - Input data(No need for complete data, only sample enough)
      targets:Tensor - Targets
      loss_fn:Loss - Loss Function
      epsilon:float
  '''
  params = model.get_params()
  analytical_grads = []
  calculated_grads = []

  def get_loss():
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    return loss

  with new_graph():
    loss = get_loss()
    loss.backward()
    _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon)

  analytical_grads = np.array(analytical_grads)
  calculated_grads = np.array(calculated_grads)
  return _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals)


def fn_grad_check(fn, inputs, params, targets=None, loss_fn=None, epsilon=1e-7, print_vals=True, **kwargs):
  '''
    Implements Gradient Check for a function instead of a complete model
    Any params that are required to be gradient checked can be specified
    targets default is ones and loss_fn default is MSE

    Params:
      fn - Function to be gradient checked
      inputs:tuple(Tensor) - inputs to the function
      params:tuple(Tensor) - the params whose data can be wiggled to get the gradients
      targets:Tensor - targets of the function
      loss_fn:Loss - loss_fn to evaluate the function
      epsilon:float
  '''
  if loss_fn is None:
    from ..nn.loss import MSE
    loss_fn = MSE()
  analytical_grads = []
  calculated_grads = []

  def get_loss(targets=targets):
    outputs = fn(*inputs, **kwargs)
    if targets is None:
      from .tensor import Tensor as tensor
      targets = tensor(np.ones(outputs.shape))
    loss = loss_fn(outputs, targets)
    return loss
  
  with new_graph():
    loss = get_loss()
    loss.backward()
    _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon)

  analytical_grads = np.array(analytical_grads)
  calculated_grads = np.array(calculated_grads)
  return _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals)
