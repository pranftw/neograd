import _setup
from _setup import execute
import numpy as np
import neograd as ng


a = np.array(3)
b = np.array([1,2,3])
c = np.array([[3,4,5], [6,7,8]])
d = np.array([[[9,8,7], [6,5,4]], [[1,2,3], [4,5,6]]])
e = np.array([[1,2], [3,4]])
f = np.array([[0.5, -2, 1], [-1, -0.4, 20]])
g = np.array([1,2,3,4,5,6])


# <------------ADD------------>
def test_add():
  execute(ng.add, [a, b])
  execute(ng.add, [a, d])


# <------------SUB------------>
def test_sub():
  execute(ng.sub, [a, b])
  execute(ng.sub, [a, d])


# <------------MUL------------>
def test_mul():
  execute(ng.mul, [b, c])
  execute(ng.mul, [c, d])


# <------------DIV------------>
def test_div():
  execute(ng.div, [b, c])


# <------------DOT------------>
def test_dot():
  execute(ng.dot, [e, c])


# <------------EXP------------>
def test_exp():
  execute(ng.exp, [e])


# <------------LOG------------>
def test_log():
  execute(ng.log, [d])


# <------------POW------------>
def test_pow():
  execute(ng.pow, [c, b])


# <------------SUM------------>
def test_sum():
  execute(ng.sum, [d])
  execute(ng.sum, [d], axis=0)
  execute(ng.sum, [d], axis=1)


# <------------TRANSPOSE------------>
def test_transpose():
  execute(ng.transpose, [c])


# <------------FLATTEN------------>
def test_flatten():
  execute(ng.flatten, [d])


# <------------RESHAPE------------>
def test_reshape():
  execute(ng.reshape, [g], new_shape=(2,3))