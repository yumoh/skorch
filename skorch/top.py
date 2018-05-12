from .torch_env import *
from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold

import torch.utils.data
import os
import numpy as np


class BaseClassifier(ClassifierMixin):
    def __init__(self, net: nn.Module, weight=None, device=None, loss_func=None, optim=None, worker_num=-1):
        super().__init__()

        self.net = net

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.loss_func = loss_func or F.nll_loss
        self.optimizer_func = optim or optimizer.SGD

        if weight:
            self.load_weight(weight)
        else:
            self.net.to(self.device)

        if worker_num < 0:
            self.worker_num = 1
        else:
            self.worker_num = int(worker_num)

    def fit(self, X, Y=None, lr=0.01, cv=5, batch_size=32, shuffle=True, epoches=10):
        """
        训练
        :param X:
        :param Y:
        :return:
        """
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if Y.dtype != np.int64:
            Y = Y.astype(np.int64)

        if not hasattr(self, 'optimizer'):
            self.optimizer = self.optimizer_func(self.net.parameters(), lr=lr)

        kfold = KFold(n_splits=cv)
        cv_loop = 0

        history = []
        for train_indexes, test_indexes in kfold.split(X, Y):
            cv_loop += 1
            train_set_data = X[train_indexes]
            train_set_target = Y[train_indexes]
            test_set_data = X[test_indexes]
            test_set_target = Y[test_indexes]

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(train_set_data),
                                               torch.from_numpy(train_set_target)),
                batch_size=batch_size, shuffle=shuffle
            )

            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(test_set_data),
                                               torch.from_numpy(test_set_target)),
                batch_size=batch_size, shuffle=shuffle
            )



            for epoch in range(1, epoches + 1):
                self.net.train()
                train_loss = 0
                train_correct = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.net(data)
                    loss = self.loss_func(output, target)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    if len(output.size()) != 2:
                        output = torch.squeeze(output)
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_history=[train_loss / len(train_set_data), train_correct / len(train_set_data)]
                self.net.eval()
                test_loss = 0
                test_correct = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.net(data)
                        loss = self.loss_func(output, target)
                        self.optimizer.step()
                        test_loss += loss.item()
                        if len(output.size()) != 2:
                            output = torch.squeeze(output)
                        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_history=[test_loss / len(test_set_data), test_correct / len(test_set_data)]
                stat_one_loop = train_history + test_history
                print('cv {} epoch {} train loss:{:.2f} train acc:{:.2f} '
                      ' test loss:{:.2f} test acc:{:.2f}'.format(cv_loop,epoch, *stat_one_loop))

                history.append(stat_one_loop)
        return history

    def predict(self, X):
        """
        预测
        :param X:
        :return:
        """
        output = self._predict_proba(X)
        pred = output.max(1)[1]
        return pred.cpu().numpy()

    def _predict_proba(self, X):
        self.net.eval()

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)

        with torch.no_grad():
            output = self.net(X.to(self.device))
            if len(output.size()) != 2:
                output = torch.squeeze(output)
            return output

    def predict_proba(self, X):
        """
        预测概率分量
        :param X:
        :return:
        """
        ypred = self._predict_proba(X)
        return ypred.cpu().numpy()

    def score(self, X, Y=None, weight=None):
        """
        模型评估
        :param X:
        :param Y:
        :param weight:
        :return:
        """
        if Y.dtype != np.int64:
            Y = Y.astype(np.int64)
        return (self.predict(X) == Y).sum() / len(Y)

    def load_weight(self, path):
        """
        torch.load_state_dict
        :param path:
        :return:
        """
        if isinstance(path, str):
            weight = torch.load(path)
        else:
            weight = path
        self.net.load_state_dict(weight)
        self.net.to(self.device)

    def save_weight(self, path):
        """
        torch.save(self.net.state_dict(),path)
        :param path:
        :return:
        """
        torch.save(self.net.state_dict(), path)
