
from .torch_env import *
from .top import BaseClassifier
class Classifier(BaseClassifier):
    def __init__(self,net,**kwargs):
        super().__init__(net,**kwargs)
