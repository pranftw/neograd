import _setup
from _setup import execute
import numpy as np
import neograd as ng
from neograd.nn.loss import BCE, CE, MSE, SoftmaxCE
from neograd.nn.activations import Softmax


inputs = np.random.randn(10,5)
def fn(inputs):
  # Returns inputs themselves because, we're only testing the loss function
  return inputs


# <------------BCE------------>
def test_bce():
  execute(fn, [inputs], loss_fn=BCE())


# <------------CE------------>
def test_ce():
  execute(fn, [Softmax.calc_softmax(inputs, axis=1)], targets=ng.tensor(np.eye(5)[np.random.randint(low=0,high=4)]), loss_fn=CE())


# <------------MSE------------>
def test_mse():
  execute(fn, [inputs], loss_fn=MSE())


# <------------SoftmaxCE------------>
def test_softmaxce():
  execute(fn, [inputs], targets=ng.tensor(np.eye(5)[np.random.randint(low=0,high=4)]), loss_fn=SoftmaxCE(axis=1))