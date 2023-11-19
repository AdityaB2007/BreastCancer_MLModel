import torch as th
from bioinfokit.analys import get_data

df = get_data('CancerData.csv').data

# convert independent and dependent variables to PyTorch tensor
X = th.tensor(df[['Age', 'BMI',	'Glucose', 'Insulin',	'HOMA',	'Leptin',	'Adiponectin', 'Resistin',	'MCP.1']].values, dtype = th.float32)
y = th.tensor(df[['Classification']].values, dtype = th.float32)

# create regression model
in_features = 1
out_features = 1
reg_model = th.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

# defining loss function; measures mean squared error of regression model
mse_loss = th.nn.MSELoss()

'''
Gradient descent optimizer.
Optimization aims at finding regression coefficients (or parameters) in such a way that the loss function achieves a minimal value.
This error minimization algorithm is known as the "backpropagation of errors"
The learning rate, lr, describes the change in weight that will aim to minimize the loss fucntion; for this reason, a small lr value is used.
'''
optimizer = th.optim.SGD(reg_model.parameters(), lr = 0.002)

# model training
'''
Neural networks use iterative solutions to estimate the regression parameters.
Reiterate the model multiple times to update the regression parameters until the loss is minimized (it should converge to a stable minimum or at the best, zero).
This process is known as gradient descent
'''
n_epoch = 6000
for i in range(n_epoch):
  # forward pass (feed data to model)
  y_pred = reg_model(X)
  # calculate loss function
  step_loss = mse_loss(y_pred, y)
  # Backward to find the derivatives of the loss function with respect to regression parameters
  # make any stored gradients to zero
  # backward pass (go back and update the regression parameters to minimize the loss)
  optimizer.zero_grad()
  step_loss.backward()
  # update with current step parameters
  optimizer.step()
  print('epoch [{}], Loss: {:.2f}'.format(i, step_loss.item()))

# estimate regression parameters
reg_model.bias.item()

# weight (W)
reg_model.weight.item()

# regression plot
from bioinfokit.analys import stat
y_pred = reg_model(X).detach()
df['Classification'] = y_pred.numpy()
stat.regplot(df=df, X = '', y = 'Classification', yhat = 'yhat')

# use r^2 value to evaluate model performance (~1 indicates very high accuracy)
from sklearn.metrics import r2_score
r2_score(y_true=y, y_pred=y_pred.detach().numpy())
