import _setup
from _setup import execute
import numpy as np
import neograd as ng

from neograd.nn.activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU


a = np.array(3)
b = np.array([1,2,3])
c = np.array([[3,4,5], [6,7,8]])
d = np.array([[[9,8,7], [6,5,4]], [[1,2,3], [4,5,6]]])
e = np.array([[1,2], [3,4]])
f = np.array([[0.5, -2, 1], [-1, -0.4, 20]])


# <------------RELU------------>
def test_relu():
  relu = ReLU()
  execute(relu, [f])


# <------------SIGMOID------------>
def test_sigmoid():
  sigmoid = Sigmoid()
  execute(sigmoid, [c])


# <------------TANH------------>
def test_tanh():
  tanh = Tanh()
  execute(tanh, [f])


# <------------SOFTMAX------------>
def test_softmax():
  softmax1 = Softmax(axis=0)
  softmax2 = Softmax(axis=1)
  softmax3 = Softmax(axis=2)
  execute(softmax1, [d])
  execute(softmax2, [d])
  execute(softmax3, [d])


# <------------LEAKYRELU------------>
def test_leaky_relu():
  leaky_relu = LeakyReLU()
  execute(leaky_relu, [c])