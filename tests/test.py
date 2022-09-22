import _setup
import neograd as ng
import numpy as np
# from neograd.nn.loss import BCE
# from neograd.nn.optim import Adam
# from sklearn.datasets import make_circles
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
# X_train, X_test, y_train, y_test = train_test_split(X,y)

# X_train, X_test = ng.tensor(X_train), ng.tensor(X_test)
# y_train, y_test = ng.tensor(y_train.reshape(750,1)), ng.tensor(y_test.reshape(250,1))

# num_train = 750
# num_test = 250
# num_iter = 50

# class NN(ng.nn.Model):
#   def __init__(self):
#     super().__init__(self)
#     self.stack = ng.nn.Sequential(
#       ng.nn.Linear(2,100),
#       ng.nn.ReLU(),
#       ng.nn.Linear(100,1),
#       ng.nn.Sigmoid()
#     )
  
#   def forward(self, inputs):
#     return self.stack(inputs)

# model = NN()
# loss_fn = BCE()
# optim = Adam(model.get_params(), 0.05)

# for iter in range(num_iter):
#   optim.zero_grad()
#   outputs = model(X_train)
#   loss = loss_fn(outputs, y_train)
#   loss.backward()
#   optim.step()
#   print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")

# with ng.no_track():
#   test_outputs = model(X_test)
#   preds = np.where(test_outputs.data>=0.5, 1, 0)

# print(classification_report(y_test.data.astype(int).flatten(), preds.flatten()))
# print(accuracy_score(y_test.data.astype(int).flatten(), preds.flatten()))

from neograd.autograd.ops import softmax
a = ng.tensor([1,2,3], requires_grad=True)
b = softmax(a, 0)
b.backward([1,1,1])
print(a.grad)