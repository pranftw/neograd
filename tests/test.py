import tests_setup
import neograd as ng

a = ng.tensor([1,2,3], requires_grad=True)
b = ng.tensor(3, requires_grad=True)

d = a.T
print(d)