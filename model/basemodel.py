import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel , self).__init__()

    def forward(self , x):
        raise NotImplementedError("Subclasses  must implement forward method")
    
    def get_config(self):
        return {}
    
    def set_config(self, config):
        pass


