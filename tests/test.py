import _setup
import neograd as ng
import numpy as np
from neograd import nn
from neograd.nn.loss import BCE
from neograd.nn.optim import Adam
from neograd.autograd.utils import grad_check
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X,y)

num_train = 750
num_test = 250
num_iter = 50

X_train, X_test = ng.tensor(X_train[:num_train,:]), ng.tensor(X_test[:num_test,:])
y_train, y_test = ng.tensor(y_train[:num_train].reshape(num_train,1)), ng.tensor(y_test[:num_test].reshape(num_test,1))

class NN(nn.Model):
  def __init__(self):
    self.stack = nn.Sequential(
      nn.Linear(2,100),
      nn.ReLU(),
      nn.Linear(100,1),
      nn.Sigmoid()
    )
  
  def forward(self, inputs):
    return self.stack(inputs)

model = NN()
loss_fn = BCE()
optim = Adam(model.parameters(), 0.05)

for iter in range(num_iter):
  optim.zero_grad()
  outputs = model(X_train)
  loss = loss_fn(outputs, y_train)
  loss.backward()
  optim.step()
  print(f"iter {iter+1}/{num_iter}\nloss: {loss}\n")

with model.eval():
  test_outputs = model(X_test)
  preds = np.where(test_outputs.data>=0.5, 1, 0)

print(classification_report(y_test.data.astype(int).flatten(), preds.flatten()))
print(accuracy_score(y_test.data.astype(int).flatten(), preds.flatten()))

# grad_check(model, X_train, y_train, loss_fn)

# # save the model
# ng.save('model.pkl', model)

# # save the parameters of the model
# model.save('model_params.pkl')

# # load the model
# model = ng.load('model.pkl')

# # load the parameters of the model
# model.load('model_params.pkl')