import numpy as np


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
  '''
  dims_to_be_summed = []
  for i,broadcasted_shape_dim in enumerate(broadcasted_shape):
    try:
      orig_data_shape_dim = orig_data_shape[i]
    except IndexError:
      dims_to_be_summed.append(True)
      continue
    if broadcasted_shape_dim==orig_data_shape_dim:
      dims_to_be_summed.append(False)
    else:
      dims_to_be_summed.append(True)
  return dims_to_be_summed