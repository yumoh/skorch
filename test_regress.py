
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
housing=fetch_california_housing(download_if_missing=True)

from skorch.regressor import Regressor
from skorch.torch_env import *

# housing.data.shape
# Out[4]: (20640, 8)
# housing.target.shape
# Out[5]: (20640,)

X_train,X_test,y_train,y_test=train_test_split(housing.data,housing.target,test_size=0.2)

net=nn.Sequential(
    nn.BatchNorm1d(8),
    nn.Linear(8,16),
    nn.ReLU(),
    nn.Linear(16,32),
    nn.Linear(32,8),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(8,1)
)

sknet=Regressor(net,loss_func=F.l1_loss)
sknet.fit(X_train,y_train,epoches=5)
sknet.save_weight('housing.regressor.pkl')
s=sknet.score(X_test,y_test)
print(s)