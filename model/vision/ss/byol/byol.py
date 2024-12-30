import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.v2 as trans
from torch.optim import Optimizer
import torchvision
from torch.utils.data import DataLoader
# from .....optimizers.lars.LARS import LARS

#helper function
import copy
# from torch.utils.tensorboard import SummaryWriter
# from pathlib import Path
# from typing import Dict, Optional

class Augment:
    """
    Image Augmentation class , inspired from SimCLR
    args:
        img_size
    
    """
    
    def __init__(self , img_size=224):
        self.augment = trans.Compose(
            [
                trans.RandomResizedCrop(img_size),
                trans.RandomHorizontalFlip(p=0.5),
                trans.ColorJitter(0.4, 0.4, 0.4, 0.1),
                trans.RandomGrayscale(p=0.2),
                trans.GaussianBlur(kernel_size=3),
                trans.ToTensor(),
            ]
        )
    def __call__(self , img):
        return self.augment(img) , self.augment(img)
        


class BYOL(nn.Module):

    def __init__(self , momentum_decay_rate,encoder='resnet50' , hidden_dim=4096 , out_dim=256):
        '''
        arg:
        encoder
        decay_rate

        '''
        super(BYOL , self).__init__()
        self.encoder = models.__dict__[encoder](weights=False)
        self.encoder.conv1 = nn.LazyConv2d(out_channels=64 , kernel_size=7 , stride=2 , padding=3 , bias=False)
        self.encoder_inf = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.momentum_decay_rate = momentum_decay_rate

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.encoder_inf, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim)  
        )

        # online
        self.online_encoder = self.encoder
        self.online_project = self.projection
        self.online_pred = nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=hidden_dim),  
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim)  
        )

        # target
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_project = copy.deepcopy(self.projection)
        self.update_targetNetwork(self.momentum_decay_rate)
    
    @torch.no_grad()
    def update_targetNetwork(self , decay):
        #update target encoder params
        for online_params , target_params in zip(self.online_encoder.parameters() , self.target_encoder.parameters()):
            # target_params = decay * target_params + (1 - decay) * online_params
            target_params.data.copy_(decay * target_params.data + (1 - decay) * online_params.data)
        
        #update target projectins
        for online_param , target_param in zip(self.online_project.parameters() , self.target_project.parameters()):
            # target_param = decay * target_param + (1 - decay) * online_param
            target_params.data.copy_(decay * target_params.data + (1 - decay) * online_params.data)

    
    def forward(self , x1 , x2):
        pred_online_x1 = self.online_pred(self.online_project(self.online_encoder(x1)))
        pred_online_x2 = self.online_pred(self.online_project(self.online_encoder(x2)))


        with torch.no_grad():
            proj_target_x1 = self.target_project(self.target_encoder(x1))
            proj_target_x2 = self.target_project(self.target_encoder(x2))

        pred_online_x1 = F.normalize(pred_online_x1)
        pred_online_x2 = F.normalize(pred_online_x2)
        proj_target_x1 = F.normalize(proj_target_x1)
        proj_target_x2 = F.normalize(proj_target_x2)

        return pred_online_x1 , pred_online_x2 , proj_target_x1 , proj_target_x2

def byol_loss(online_pred1 , online_pred2 , target_proj1 , target_proj2):

    loss = 2 - 2 * (F.cosine_similarity(online_pred1 , target_proj1.detach()).mean() +
                    F.cosine_similarity(online_pred2 , target_proj2.detach()).mean())
    return loss



def train_byol(data_loader , optimizer , num_epoch):
    model.train()

    for ep in range(num_epoch):

        epoch_loss = 0.0

        for imgs , _ in data_loader:
            img1 , img2 = imgs
            img1 , img2 = img1.cuda() , img2.cuda()

            pred1 , pred2 , target1 , target2 = model(img1 , img2)
            loss = byol_loss(pred1 , pred2 , target1 , target2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_targetNetwork(model.momentum_decay_rate)

            epoch_loss += loss.item()

        print(f"Epoch [{ep+1} / {num_epoch}] , loss : {epoch_loss / len(data_loader):.4f}")

    
if __name__ == '__main__':
    lars_decay_rate = 1.5e-6
    momentum_decay_rate = 0.996
   
    batch_size = 32
    epochs = 100

    #lr = 0.2 x batch_size/256 --> according to the paper
    lr = 0.2 * (batch_size/256)

    transform = Augment(img_size=28)
    train_dataset = torchvision.datasets.MNIST(
        root='testing/playground/mnist/data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    model = BYOL(momentum_decay_rate=momentum_decay_rate).cuda()
    
    

    # optimizer = LARS(model.parameters() , lr=lr , weight_decay=lars_decay_rate)
    optimizer = torch.optim.AdamW(model.parameters() , lr=lr)
    train_byol(data_loader=train_loader , optimizer=optimizer , num_epoch=epochs)



    

        



        