from .torch_env import *
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold

import torch.utils.data
import os
import numpy as np
from sklearn.metrics import r2_score


class BaseNet(ClassifierMixin):
    def __init__(self, net: nn.Module, weight=None, device=None, loss_func=None, optim=None, worker_num=-1, **kwargs):
        super().__init__()

        self.net = net

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.loss_func = loss_func
        self.optimizer = optim

        if weight:
            self.load_weight(weight)
        else:
            self.net.to(self.device)

        if worker_num < 0:
            self.worker_num = 1
        else:
            self.worker_num = int(worker_num)

        self.early_times = 0
        self.history = []

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


class BaseClassifier(BaseNet):
    def __init__(self, net: nn.Module, loss_func=F.nll_loss, **kwargs):
        kwargs['loss_func'] = loss_func
        super().__init__(net=net, **kwargs)

    def fit(self, X, Y=None, lr=0.01, cv=5, batch_size=32, shuffle=True, epoches=10, early_stop=None, optim=None):
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
        if early_stop:
            self.early_times = 0
        if self.optimizer is None:
            if optim is None:
                self.optimizer = optimizer.SGD(self.net.parameters(), lr)
            else:
                self.optimizer = optim
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

                train_history = [train_loss / len(train_set_data), train_correct / len(train_set_data)]
                self.net.eval()
                test_loss = 0
                test_correct = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.net(data)
                        loss = self.loss_func(output, target)
                        test_loss += loss.item()
                        if len(output.size()) != 2:
                            output = torch.squeeze(output)
                        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_history = [test_loss / len(test_set_data), test_correct / len(test_set_data)]
                stat_one_loop = train_history + test_history
                print('\rcv {} epoch {} train loss:{:.4f} train acc:{:.4f} '
                      ' test loss:{:.4f} test acc:{:.4f}'.format(cv_loop, epoch, *stat_one_loop), end='')

                history.append(stat_one_loop)

                if early_stop:
                    if isinstance(early_stop, str):
                        i = ['train_loss', 'train_acc', 'test_loss', 'test_acc'].index(early_stop)
                        if len(history) > 2:
                            if early_stop in {'train_acc', 'test_acc'} and history[-1][i] <= history[-2][i] and \
                                    history[-1][i] <= history[-3][i]:
                                self.early_times += 1
                            elif history[-1][i] >= history[-2][i] and history[-1][i] >= history[-3][i]:
                                self.early_times += 1
                            else:
                                self.early_times = 0
                    elif str(type(early_stop)) == 'function' and early_stop(*stat_one_loop):
                        self.early_times += 1

                    elif isinstance(early_stop, int):

                        if len(history) > early_stop and all(
                                [history[-1][1] <= history[-2 - i][1] for i in range(early_stop)]):
                            self.early_times += 1
                        else:
                            self.early_times = 0

                    if self.early_times > 2:
                        self.history += history
                        return

        self.history += history
        return

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


class BaseRegressor(BaseNet):
    def __init__(self, net: nn.Module, loss_func=F.l1_loss, **kwargs):
        kwargs['loss_func'] = loss_func
        super().__init__(net=net, **kwargs)

    def fit(self, X, Y=None, lr=0.01, cv=5, batch_size=32, shuffle=True, epoches=10, early_stop=None, optim=None):
        """
        训练
        :param X:
        :param Y:
        :return:
        """
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if Y.dtype != np.float32:
            Y = Y.astype(np.float32)
        if early_stop:
            self.early_times = 0

        if self.optimizer is None:
            if optim is None:
                self.optimizer = optimizer.SGD(self.net.parameters(), lr)
            else:
                self.optimizer = optim

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
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.net(data)
                    loss = self.loss_func(output.view_as(target), target)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                # train_history = [train_loss / len(train_set_data)]
                train_history = [train_loss]
                self.net.eval()
                test_loss = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.net(data)
                        loss = self.loss_func(output.view_as(target), target)
                        test_loss += loss.item()
                # test_history = [test_loss / len(test_set_data)]
                test_history = [test_loss]
                stat_one_loop = train_history + test_history

                print('\rcv {} epoch {} train loss:{:.4f} '
                      'test loss:{:.4f}'.format(cv_loop, epoch, *stat_one_loop), end='')

                history.append(stat_one_loop)
                if str(type(early_stop)) == 'function' and early_stop(*stat_one_loop):
                    self.early_times += 1
                elif isinstance(early_stop, str):
                    i = ['train_loss', 'test_loss'].index(early_stop)
                    if len(history) > 2:
                        if history[-1][i] >= history[-2][i] and history[-1][i] >= history[-3][i]:
                            self.early_times += 1
                        else:
                            self.early_times = 0
                elif isinstance(early_stop, int) and len(history) > early_stop:
                    if all([history[-1][0] >= history[-2 - i][0] for i in range(early_stop)]):
                        self.early_times += 1
                    else:
                        self.early_times = 0
                else:
                    self.early_times = 0

                if self.early_times > 2:
                    self.history += history
                    return
        self.history += history
        return

    def predict(self, X):
        """
        预测
        :param X:
        :return:
        """
        output = self._predict_proba(X)
        return output.cpu().numpy()

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

    def score(self, X, Y, weight=None, metrics_func=r2_score):
        """
        模型评估
        :param X:
        :param Y:
        :param weight:
        :return:
        """
        if Y.dtype != np.float32:
            Y = Y.astype(np.float32)
        pred = self.predict(X)

        return metrics_func(pred, Y)
