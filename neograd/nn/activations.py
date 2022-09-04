from .layers import Layer
from ..autograd.ops import relu, sigmoid, tanh


class ReLU(Layer):
  def forward(self, inputs):
    return relu(inputs)


class Sigmoid(Layer):
  def forward(self, inputs):
    return sigmoid(inputs)


class Tanh(Layer):
  def forward(self, inputs):
    return tanh(inputs)
