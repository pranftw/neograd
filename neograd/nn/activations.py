from .layers import Layer
from ..autograd.ops import relu, sigmoid, tanh


class ReLU(Layer):
  def forward(self, inputs):
    return relu(inputs)

  def __repr__(self):
    return 'ReLU()'
  
  def __str__(self):
    return 'ReLU'


class Sigmoid(Layer):
  def forward(self, inputs):
    return sigmoid(inputs)

  def __repr__(self):
    return 'Sigmoid()'
  
  def __str__(self):
    return 'Sigmoid'


class Tanh(Layer):
  def forward(self, inputs):
    return tanh(inputs)
  
  def __repr__(self):
    return 'Tanh()'
  
  def __str__(self):
    return 'Tanh'
