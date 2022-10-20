import _setup
from _setup import execute
import numpy as np
import neograd as ng
from neograd import nn


# <------------DROPOUT------------>
def test_dropout():
  input_data = np.random.randn(7,5)
  fn1 = nn.Dropout(0.5)
  fn2 = nn.Dropout(0.3)
  fn3 = nn.Dropout(0.7)
  fn4 = nn.Dropout(1)
  execute(fn1, [input_data])
  execute(fn2, [input_data])
  execute(fn3, [input_data])
  execute(fn4, [input_data])


# <------------LINEAR------------>
def test_linear():
  '''
  in cases like Linear layer where the backprop is calculated by
  autograd, without any explicit gradient fns, so there might be some floating point
  truncations that could result in slight deviation that is acceptable
  '''
  input_data = np.random.randn(2,5)
  fn = nn.Linear(5, 3)
  execute(fn, [input_data], fn.parameters(), tolerance=3e-7)


# <------------CONV2D------------>
def test_conv2d():
  input_data = np.random.randn(2,12,15)
  fn1 = nn.Conv2D((3,3))
  fn2 = nn.Conv2D((2,2), padding=2)
  fn3 = nn.Conv2D((4,4), stride=2)
  fn4 = nn.Conv2D((3,3), padding=1, stride=1)
  execute(fn1, [input_data], fn1.parameters())
  execute(fn2, [input_data], fn2.parameters())
  execute(fn3, [input_data], fn3.parameters())
  execute(fn4, [input_data], fn4.parameters())


# <------------CONV3D------------>
def test_conv3d():
  input_data = np.random.randn(2,2,12,13)
  fn1 = nn.Conv3D(2,5,(3,3))
  fn2 = nn.Conv3D(2,5,(2,2), padding=2)
  fn3 = nn.Conv3D(2,5,(4,4), stride=2)
  fn4 = nn.Conv3D(2,5,(3,3), padding=1, stride=1)
  execute(fn1, [input_data], fn1.parameters())
  execute(fn2, [input_data], fn2.parameters())
  execute(fn3, [input_data], fn3.parameters())
  execute(fn4, [input_data], fn4.parameters())


# <------------MAXPOOL2D------------>
def test_maxpool2d():
  input_data = np.random.randn(2,12,15)
  fn1 = nn.MaxPool2D((3,3), stride=3)
  execute(fn1, [input_data], fn1.parameters())


# <------------MAXPOOL3D------------>
def test_maxpool3d():
  input_data = np.random.randn(2,3,12,15)
  fn1 = nn.MaxPool3D((3,3), stride=3)
  execute(fn1, [input_data], fn1.parameters())