
from .top import BaseRegressor

class Regressor(BaseRegressor):
    def __init__(self,net,*args,weight=None,device=None,loss_func=None,optim=None,worker_num=-1,**kwargs):
        super().__init__(net,weight,device,loss_func,optim,worker_num)