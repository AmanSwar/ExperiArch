import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class BYOLAugmentation:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)
    


class BYOL(nn.Module):

    def __init__(self , backbone='resnet18' , out_dim=256 , hidden_dim=4096  , moving_avg_decay=0.99):
        super(BYOL ,self).__init__()
        
        self.online_encoder = models.__dict__[backbone](pretrained=False)
        feature_dim = self.online_encoder.fc.in_feature
        self.online_encoder.fc = nn.Identity()

        self.online_projection = nn.Sequential(
            nn.Linear(feature_dim , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim , out_dim)

        )

        self.predictor = nn.Sequential(
            nn.Linear(out_dim , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim , out_dim)
        )

        #target network
        self.target_encoder = models.__dict__[backbone](pretrained=False)
        self.target_encoder.fc = nn.Identity()
        self.target_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.moving_avg_decay = moving_avg_decay
        self._update_target_network(0)

    @torch.no_grad()
    def _update_target_network(self , decay):
        for online_params , target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = decay * target_params.data + (1-decay) * online_params.data
        for online_params, target_params in zip(self.online_projection.parameters(), self.target_projection.parameters()):
            target_params.data = decay * target_params.data + (1 - decay) * online_params.data

    
    def forward(self , x1 , x2):
        online_proj1 = self.online_projection(self.online_encoder(x1))
        online_proj2 = self.online_projection(self.online_encoder(x2))
        online_pred1 = self.predictor(online_proj1)
        online_pred2 = self.predictor(online_proj2)

        with torch.no_grad():
            target_proj1 = self.target_projection(self.target_encoder(x1))
            target_proj2 = self.target_projection(self.target_encoder(x2))

        return online_pred1, online_pred2, target_proj1, target_proj2



def byol_loss(pred1 , pred2 , target1, target2):
    loss = 2 - 2 * (F.cosine_similarity(pred1, target2.detach(), dim=-1).mean() +
                    F.cosine_similarity(pred2, target1.detach(), dim=-1).mean())
    return loss