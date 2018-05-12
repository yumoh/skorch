from sklearn.datasets import load_digits

digits = load_digits()

from my_net import Net

net = Net()

from skorch.classify import Classifier

from skorch.torch_env import *
sknet=Classifier(net,loss_func=F.nll_loss)
sknet.load_weight('sklearn.dataset.digits.pkl')
X=digits.data.reshape(-1,1,8,8)
Y=digits.target
# sknet.fit(X,Y)
# sknet.save_weight('sklearn.dataset.digits.pkl')
s=sknet.score(X,Y)
print(s)
