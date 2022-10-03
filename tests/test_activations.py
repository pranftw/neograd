import _setup
from _setup import execute
import numpy as np
import neograd as ng

from neograd.nn.activations import relu, sigmoid, tanh, softmax


a = np.array(3)
b = np.array([1,2,3])
c = np.array([[3,4,5], [6,7,8]])
d = np.array([[[9,8,7], [6,5,4]], [[1,2,3], [4,5,6]]])
e = np.array([[1,2], [3,4]])
f = np.array([[0.5, -2, 1], [-1, -0.4, 20]])


# <------------RELU------------>
def test_relu():
  execute(relu, (f,))


# <------------SIGMOID------------>
def test_sigmoid():
  execute(sigmoid, (c,))


# <------------TANH------------>
def test_tanh():
  execute(tanh, (f,))


# <------------SOFTMAX------------>
def test_softmax():
  execute(softmax, (d,), axis=0)
  execute(softmax, (d,), axis=1)
  execute(softmax, (d,), axis=2)