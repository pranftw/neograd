import numpy as np


def process_data(data):
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

def unflatten_data(data, orig_data_shape, broadcasted_shape):
    if broadcasted_shape is not None:
      dims_to_be_summed = get_dims_to_be_summed(orig_data_shape, broadcasted_shape)
      unflattened_data = data.reshape(broadcasted_shape)
      for i,dim in reversed(list(enumerate(dims_to_be_summed))):
        if dim==1:
          unflattened_data = np.sum(unflattened_data, axis=i)
    else:
      unflattened_data = data
    return unflattened_data

def get_dims_to_be_summed(orig_data_shape, broadcasted_shape):
    dims_to_be_summed = []
    for i,broadcasted_shape_dim in enumerate(broadcasted_shape):
      try:
        orig_data_shape_dim = orig_data_shape[i]
      except IndexError:
        dims_to_be_summed.append(1)
        continue
      if broadcasted_shape_dim==orig_data_shape_dim:
        dims_to_be_summed.append(0)
      else:
        dims_to_be_summed.append(1)
    return dims_to_be_summed