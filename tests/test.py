import sys
sys.path.append('..')
from autograd import tensor

a = tensor([1,2,3], requires_grad=True)
b = tensor(3, requires_grad=True)

d = a.T
print(d)