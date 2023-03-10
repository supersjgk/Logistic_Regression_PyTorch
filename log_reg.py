import torch
import torch.nn as nn
import numpy  as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#binary classification data for logistic regression
dataset=datasets.load_breast_cancer()
x,y=dataset.data,dataset.target
n_samples,n_features=x.shape
#print(n_samples,n_features)

#training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)

#scaling data to tranform data to have mean=0, and standard variance=1
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#initially values loaded are double
x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))
#print(x_train)
#print(y_train)

#reshaping y from row to column vector
y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

#model => f=wx+b => sigmoid function

class Log(nn.Module):
	def __init__(self,inp_features):
		super(Log,self).__init__()
		self.linear=nn.Linear(inp_features,1)
	
	def forward(self,x):
		y_pred=torch.sigmoid(self.linear(x))
		return y_pred

model=Log(n_features)

#learning rate
lrate=0.01

#loss-binary cross entropy loss
loss=nn.BCELoss()

#optimizer-Stochastic gradient descent
optimizer=torch.optim.SGD(model.parameters(),lr=lrate)

#training
epochs=100

for epoch in range(epochs):
	#forward pass
	y_pred=model(x_train)
	#loss
	l=loss(y_pred,y_train)
	#backward pass-calculate gradients
	l.backward()
	#update weights
	optimizer.step()
	#empty gradients
	optimizer.zero_grad()
	
	if (epoch+1)%10==0:
		print(f'epoch={epoch+1}, loss={l.item():.4f}')

#evaluating the model
with torch.no_grad():
	y_pred=model(x_test)
	y_pred_class=y_pred.round()
	acc=y_pred_class.eq(y_test).sum()/float(y_test.shape[0])
	print(f'accuracy={acc:.4f}')
