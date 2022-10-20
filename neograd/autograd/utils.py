import numpy as np
from .graph import Graph
from itertools import zip_longest


def process_data(data):
  '''Checks and processes the data for storage in Tensor

  Supported types for data - [int, float, list, np.ndarray]
  Elements in data should be float or be typecastable to float

  Args:
    data (int or float or list or np.ndarray): Data to be processed
  
  Returns:
    Processed data
  
  Raises:
    TypeError: If data or its elements aren't typecastable to float
    TypeError: If data is not instance of supported types
  '''
  supported_types = (int, float, list, np.ndarray)
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
  ''' Unbroadcasts the data to its original shape

  If data(a np object) is broadcasted during an operation, then it is unbroadcasted here,
  where all axes where it was broadcasted are summed along those axes to
  give the original shape of the data. If broadcasted_shape is None, then the data is
  returned as is.

  Args:
    data (np.ndarray): Data to be unbroadcasted
    orig_data_shape (tuple): Original shape of data before broadcasting
    broadcasted_shape (tuple): Shape to which data has been broadcasted to
  
  Returns:
    Data that is unbroadcasted
  '''

  def get_axes_to_be_summed(orig_data_shape, broadcasted_shape):
    '''Returns the axes along which data has been broadcasted

    Given the original data shape and its broadcasted shape, it returns True along
    an axis if their dimensions don't match, else returns False if they match,
    meaning there has been no broadcasting along that axis.
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    Args:
      orig_data_shape (tuple): Original shape of data before broadcasting
      broadcasted_shape (tuple): Shape to which data has been broadcasted to
    
    Returns:
      tuple of axes on which there's been broadcasting
    '''
    axes_to_be_summed = []
    zipped = list(zip_longest(tuple(reversed(broadcasted_shape)), tuple(reversed(orig_data_shape)), fillvalue=None))
    for dim, (dim_broadcasted, dim_orig) in enumerate(reversed(zipped)):
      if dim_broadcasted!=dim_orig:
        axes_to_be_summed.append(dim)
    return tuple(axes_to_be_summed)

  if broadcasted_shape is not None:
    axes_to_be_summed = get_axes_to_be_summed(orig_data_shape, broadcasted_shape)
    unbroadcasted_data = np.sum(data, axis=axes_to_be_summed)
  else:
    unbroadcasted_data = data
  return unbroadcasted_data

def get_graph():
  '''Returns graph that is in use and present in Graph.graph

  If Graph.graph is None, then the global graph _NG_GRAPH is used

  Returns:
    Graph object that is currently used
  '''
  if Graph.graph is None:
    from .. import _NG_GRAPH
    graph = _NG_GRAPH
  else:
    graph = Graph.graph
  return graph


class new_graph:
  '''Creates a Graph object

  Context Manager to create a new graph if required anywhere and under the
  circumstances where it shouldn't interfere with the global _NG_GRAPH

  After entering, Graph object created is set in Graph.graph. After exiting
  the Graph.graph is set back to None which implies that global _NG_GRAPH will
  be used
  '''
  def __enter__(self):
    Graph.graph = Graph()
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    Graph.graph = None


class no_track:
  '''Prevents tracking of Tensors

  Context Manager to prevent creation of a backward graph, when gradient calculation
  is not required, for ex when testing a model after training it, you don't need
  any backward pass

  On entering, graph.track is set to False to indicate no tracking and on exiting, it is
  set back to True

  Parameters:
    graph (Graph): The current graph in use
  '''
  def __init__(self):
    self.graph = get_graph()

  def __enter__(self):
    self.graph.track = False
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.graph.track = True


def _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals):
  '''Evaluates the gradient check and indicates whether it has passed or not

  Calculates the distance between the analytical and calculated gradients and if it is
  less than epsilon, then it has passed else failed

  Args:
    analytical_grads (list of int or float): Gradients that are calculated analytically
      by wiggling the parameters
    calculated_grads (list of int or float): Gradients that are calulated through
      backpropagation
    epsilon (float): The amount by which params need to be wiggled
    print_vals (bool): True if distance and verdict needs to be printed
  
  Returns:
    Distance between analytical and calculated gradients
  '''
  dist = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
  if print_vals:
    print("Gradient Check Distance:", dist)
    if dist<epsilon:
      print("Gradient Check PASSED")
    else:
      print("Gradient Check FAILED")
  return dist


def _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon):
  '''Changes the params value by epsilon and calculates the analytical gradient

  First to each element in params.data epsilon is added and loss is calculated, similarly
  2*epsilon is subtracted to get another loss and using these two analytical gradient is calculated
  and is appended to analytical_grads and the gradient in param is appended to calculated_grads

  Args:
    analytical_grads (list of int or float): Gradients that are calculated analytically
      by wiggling the parameters
    calculated_grads (list of int or float): Gradients that are calulated through
      backpropagation
    params (list of Tensor): All params that need to be wiggled
    get_loss: function that is used to calculate the loss
    epsilon (float): The amount by which params need to be wiggled
  '''
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
    param.zero_grad() # to prevent any side effects


def grad_check(model, inputs, targets, loss_fn, epsilon=1e-7, print_vals=True):
  '''Performs Gradient Check

  Implements Gradient Check, to make sure that backprop is calculating
  the right gradients. All the parameters in the model are checked.

  If distance between backprop gradients and numerical gradients is less
  than epsilon, then the gradients are proper, if not there is an issue
    
  Args:
    model (Model): The Neural Network to be evaluated
    inputs (Tensor): Input data(No need for complete data, only sample enough)
    targets (Tensor): Targets
    loss_fn (Loss): Loss Function
    epsilon (float): The amount by which params need to be wiggled Defaults to 1e-7
    print_vals (bool): True if distance and verdict needs to be printed
  
  Returns:
    Distance between analytical and calculated gradients
  '''
  params = model.parameters()
  analytical_grads = []
  calculated_grads = []

  for param in params:
    param.zero_grad()

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
  '''Performs Gradient Check for a function

  Implements Gradient Check for a function instead of a complete model
  Any params that are required to be gradient checked can be specified

  Args:
    fn: Function to be gradient checked
    inputs (list of Tensor): inputs to the function
    params (list of Tensor): the params whose data can be wiggled to get the gradients
    targets (Tensor): targets of the function
    loss_fn (Loss): loss_fn to evaluate the function
    epsilon (float): The amount by which params need to be wiggled Defaults to 1e-7
    print_vals (bool): True if distance and verdict needs to be printed
    **kwargs: Any kwargs to be passed to fn
  
  Returns:
    Distance between analytical and calculated gradients
  '''
  if loss_fn is None:
    from ..nn.loss import MSE
    loss_fn = MSE()
  analytical_grads = []
  calculated_grads = []

  for param in params:
    param.zero_grad()

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
