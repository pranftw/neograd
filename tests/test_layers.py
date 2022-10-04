import _setup
from _setup import execute
import numpy as np
import neograd as ng


# <------------LINEAR------------>
def test_linear():
  '''
  in cases like Linear layer where the backprop is calculated by
  autograd, without any explicit gradient fns, so there might be some floating point
  truncations that could result in slight deviation that is acceptable
  '''
  input_data = np.random.randn(10,5)
  fn = ng.nn.Linear(5, 3)
  execute(fn, [input_data], fn.get_params(as_dict=False,return_frozen=False), tolerance=3e-7)


# <------------CONV2D------------>
def test_conv2d():
  input_data = np.random.randn(2,12,15)
  fn1 = ng.nn.Conv2D((3,3))
  fn2 = ng.nn.Conv2D((2,2), padding=2)
  fn3 = ng.nn.Conv2D((4,4), stride=2)
  fn4 = ng.nn.Conv2D((3,3), padding=1, stride=1)
  execute(fn1, [input_data], fn1.get_params(as_dict=False,return_frozen=False))
  execute(fn2, [input_data], fn2.get_params(as_dict=False,return_frozen=False))
  execute(fn3, [input_data], fn3.get_params(as_dict=False,return_frozen=False))
  execute(fn4, [input_data], fn4.get_params(as_dict=False,return_frozen=False))


# <------------CONV3D------------>
def test_conv3d():
  input_data = np.random.randn(3,2,12,15)
  fn1 = ng.nn.Conv3D(2,5,(3,3))
  fn2 = ng.nn.Conv3D(2,5,(2,2), padding=2)
  fn3 = ng.nn.Conv3D(2,5,(4,4), stride=2)
  fn4 = ng.nn.Conv3D(2,5,(3,3), padding=1, stride=1)
  execute(fn1, [input_data], fn1.get_params(as_dict=False,return_frozen=False))
  execute(fn2, [input_data], fn2.get_params(as_dict=False,return_frozen=False))
  execute(fn3, [input_data], fn3.get_params(as_dict=False,return_frozen=False))
  execute(fn4, [input_data], fn4.get_params(as_dict=False,return_frozen=False))


# <------------MAXPOOL2D------------>
def test_maxpool2d():
  input_data = np.random.randn(2,12,15)
  fn1 = ng.nn.MaxPool2D((3,3), stride=3)
  execute(fn1, [input_data], fn1.get_params(as_dict=False,return_frozen=False))


# <------------MAXPOOL3D------------>
def test_maxpool3d():
  input_data = np.random.randn(2,3,12,15)
  fn1 = ng.nn.MaxPool3D((3,3), stride=3)
  execute(fn1, [input_data], fn1.get_params(as_dict=False,return_frozen=False))