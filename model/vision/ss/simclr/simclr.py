import os
import glob
import time
from skimage import io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision


from torch.utils.data import Dataset , DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomApply,
    Compose,
    GaussianBlur,
    ToTensor,
)
import torchvision.models as models

DEVICE  = torch.device('cuda')


def get_complete_transform(out_shape , kernel_size , s=1.0):
    rnd_crop = RandomResizedCrop(out_shape)
    rnd_flip = RandomHorizontalFlip(p=0.5)
    color_jitter = ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)
    
    rnd_gray = RandomGrayscale(p=0.2)
    gaussian_blur = GaussianBlur(kernel_size=kernel_size)
    rnd_gaussian_blur = RandomApply([gaussian_blur], p=0.5)
    to_tensor = ToTensor()
    image_transform = Compose([
        to_tensor,
        rnd_crop,
        rnd_flip,
        rnd_color_jitter,
        rnd_gray,
        rnd_gaussian_blur,
    ])
    return image_transform


class ContrastiveLearningViewGen(object):

    def __init__(self , base_trans , n_views=2):
        self.base_trans = base_trans
        self.n_views = n_views
    
    def __call__(self , x):
        views = [self.base_trans(x) for i in range(self.n_views)]
        return views
    


class CustomDataset(Dataset):

    def __init__(self , list_images , trans=None):
        self.list_images = list_images
        self.trans = trans


    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.list_images[idx]
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image
    

# The size of the images
output_shape = [224,224]
kernel_size = [21,21] # 10% of the output_shape

# The custom transform
base_transforms = get_complete_transform(output_shape=output_shape, kernel_size=kernel_size, s=1.0)
custom_transform = ContrastiveLearningViewGen(base_transform=base_transforms)

ds = CustomDataset(
    list_images=None,
    transform=custom_transform
)

BATCH_SIZE = 32

# Building the data loader
train_dl = torch.utils.data.DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


class Identity(nn.Module):

    def __init__(self):
        super(Identity , self).__init__()
    def forward(self , x):
        return x
    
class SimClr(nn.Module):

    def __init__(self , linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        resnet18 = models.resnet18(pretrained=False)
        resnet18.fc = Identity()
        self.encoder = resnet18
        self.projection = nn.Sequential(
            nn.Linear(512 , 256),
            nn.ReLU(),
            nn.Linear(256 , 256)
        )

    def forward(self , x):
        if not self.linear_eval:
            x = torch.cat(x , dim=0)
        
        encoding = self.encoder(x)
        projection = self.projection(encoding)
        return projection
    

# wtf does this does
LABELS = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
LABELS = LABELS.to(DEVICE) #128,128

def cont_loss(features, temp):
    """
    The NTxent Loss.
    
    Args:
        z1: The projection of the first branch
        z2: The projeciton of the second branch
    
    Returns:
        the NTxent loss
    """
    similarity_matrix = torch.matmul(features, features.T) # 128, 128
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(DEVICE)
    # ~mask is the negative of the mask
    # the view is required to bring the matrix back to shape
    labels = LABELS[~mask].view(LABELS.shape[0], -1) # 128, 127
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 128, 127

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # 128, 1

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # 128, 126

    logits = torch.cat([positives, negatives], dim=1) # 128, 127
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / temp
    return logits, labels
    

simclr_model = SimClr().to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(simclr_model.parameters())

epoch = 10

for ep in range(epoch):
    t0 = time.time()
    runn_loss = 0.0
    
    for i , views in enumerate(train_dl):
        proj = simclr_model([views.to(DEVICE)])
        logits , labels  = cont_loss(proj , temp=2)
        loss = criterion(logits ,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
