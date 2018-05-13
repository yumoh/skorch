
from .top import BaseRegressor

class Regressor(BaseRegressor):
    def __init__(self,net,**kwargs):
        super().__init__(net,**kwargs)