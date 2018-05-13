
from sklearn.model_selection import train_test_split


from skorch.regressor import Regressor
from skorch.torch_env import *
import numpy as np
import os

X=np.linspace(-20,20,20000).reshape(-1,1)
Y=np.sin(X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

net=nn.Sequential(
    nn.Linear(1,50),
    nn.Sigmoid(),
    nn.Linear(50,1),
    # nn.Sigmoid(),
    # nn.Linear(10,1)
)

sknet=Regressor(net,loss_func=F.l1_loss)
weight_path='housing.regressor.pkl'
if os.path.exists(weight_path):
    sknet.load_weight(weight_path)

optim=optimizer.Adadelta(net.parameters(),lr=0.01)
sknet.fit(X_train,y_train,epoches=20,early_stop=2,optim=optim)
sknet.save_weight('housing.regressor.pkl')
import matplotlib.pylab as plt
ypred=sknet.predict(X)
plt.plot(X,Y,'b',X,ypred,'r')
plt.show()
print('\n')
