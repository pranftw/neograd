import numpy as np
from .ops import Operation
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


class NewGraph:
  '''
    Context Manager to create a new graph if required within an operation or
      anywhere and shouldn't interfere with the global _NG_GRAPH
  '''
  def __enter__(self):
    Operation.graph = Graph()
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    Operation.graph = None