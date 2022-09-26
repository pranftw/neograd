import numpy as np
from .operation import Operation


# <------------ADD------------>

class Add(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data+tens2.data, tens1, tens2)

  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:ug)

def add(tens1, tens2):
  return Add().forward(tens1, tens2)


# <------------SUB------------>

class Sub(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data-tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:-ug)

def sub(tens1, tens2):
  return Sub().forward(tens1, tens2)


# <------------MUL------------>

class Mul(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data*tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:tens2.data*ug)
    tens2.set_grad_fn(lambda ug:tens1.data*ug)

def mul(tens1, tens2):
  return Mul().forward(tens1, tens2)


# <------------DIV------------>

class Div(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data/tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:(1/tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:((-1*tens1.data)/np.power(tens2.data, 2))*ug)

def div(tens1, tens2):
  return Div().forward(tens1, tens2)


# <------------DOT------------>

class Dot(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.dot(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    tens1.set_grad_fn(lambda ug:np.dot(ug, tens2.data.T))
    tens2.set_grad_fn(lambda ug:np.dot(tens1.data.T, ug))

def dot(tens1, tens2):
  return Dot().forward(tens1, tens2)


# <------------EXP------------>

class Exp(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.exp(tens.data), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:np.exp(tens.data)*ug)

def exp(tens):
  return Exp().forward(tens)


# <------------LOG------------>

class Log(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.log(tens.data), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:(1/tens.data)*ug)

def log(tens):
  return Log().forward(tens)


# <------------POW------------>

class Pow(Operation):
  def forward(self, tens1, tens2):
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.power(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    result = np.power(tens1.data, tens2.data)
    tens1.set_grad_fn(lambda ug:(np.power(tens1.data, tens2.data-1) * tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:(result*np.log(tens1.data))*ug)

def pow(tens1, tens2):
  return Pow().forward(tens1, tens2)


# <------------SUM------------>

class Sum(Operation):
  def __init__(self, axis=None):
    self.axis = axis
  
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.sum(tens.data, axis=self.axis), tens)
  
  def backward(self, tens):
    def grad_backward(ug):
      tens_shape = list(tens.shape)
      if self.axis is not None:
        try:
          tens_shape[self.axis] = 1
        except IndexError:
          pass
        lg = 1
      else:
        lg = np.ones(tens.shape)

      if self.axis is not None:
        grads = np.dot(lg,ug)
        try:
          num_repeat = tens.shape[self.axis]
        except IndexError:
          num_repeat = 1
        grads = grads[np.newaxis]
        grads = np.concatenate([grads]*num_repeat)
      else:
        grads = lg*ug
      return grads
    tens.set_grad_fn(grad_backward)

def sum(tens, axis=None):
  return Sum(axis).forward(tens)


# <------------TRANSPOSE------------>

class Transpose(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.T, tens)

  def backward(self, tens):
    tens.set_grad_fn(lambda ug:ug.T)

def transpose(tens):
  return Transpose().forward(tens)


# <------------FLATTEN------------>

class Flatten(Operation):
  def forward(self, tens):
    tens = self.get_tensors(tens)
    flattened = tens.data.flatten()
    return self.get_result_tensor(flattened.reshape(flattened.shape[0],1), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def flatten(tens):
  return Flatten().forward(tens)


# <------------RESHAPE------------>

class Reshape(Operation):
  def forward(self, tens, new_shape):
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.reshape(new_shape), tens)
  
  def backward(self, tens):
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def reshape(tens, new_shape):
  return Reshape().forward(tens, new_shape)