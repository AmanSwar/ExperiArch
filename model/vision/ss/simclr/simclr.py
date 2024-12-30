
#main libs
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as trans
from torch.utils.data import Dataset , DataLoader
import torch.nn.functional as F

#helper function
def same_device(x  ,y):

    x = x.to(y.device)
    return x

class DataAug:
    '''
    params:
    - img_size
    callable object
    - image

    returns:
    - 2 augmented views of same image
    
    '''

    def __init__(self , img_size):
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
    

class ContrastiveLoss(nn.Module):

    def __init__(self , batch_size , temp=0.5):
        super().__init__()

        self.batch_size = batch_size
        self.temp = temp
        self.mask = (~torch.eye(batch_size * 2 , batch_size * 2 , dtype=bool)).float()

    def similarity_per_batch(self , x, y):
        repr = torch.cat([x ,y] , dim=0)
       
        return F.cosine_similarity(repr.unsqueeze(1) , repr.unsqueeze(0) , dim=2)
    
    def forward(self, batch_view1 , batch_view2):
        
        b_size = batch_view1.shape[0]
        #[32, 128] -> z_i , z_j
        z_i = F.normalize(batch_view1 , p=2 , dim=1)
        z_j = F.normalize(batch_view2 , p=2 , dim=1)
        
        #calculate similarity
        sim_matrix = self.similarity_per_batch(z_i , z_j)
        #
        sim_ij = torch.diag(sim_matrix , b_size)
        sim_ji = torch.diag(sim_matrix , b_size)

        pos = torch.cat([sim_ij , sim_ji] , dim=0)
        #
        
        #nominator of the loss function
        nom = torch.exp(pos / self.temp)
        denom  = same_device(self.mask , sim_matrix) * torch.exp(sim_matrix / self.temp)
        all_loss = -torch.log(nom / torch.sum(denom , dim=1))
        loss = torch.sum(all_loss) / (2 * self.batch_size)

        return loss
    

import torchvision.models as models    
class SimCLR(nn.Module):

    def __init__(self  , model , out_dim):
        super().__init__()
        self.model = model
        self.encoder = models.__dict__[model](pretrained=False)
        self.encoder.conv1 = nn.LazyConv2d(64 , kernel_size=7 , stride=2 , padding=3 , bias=False)
        self.encoder_inf = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_inf , 512),
            nn.ReLU(),
            nn.Linear(512 , out_dim)
        )

    def forward(self , x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h , z
    
def train_simCLR(model , train_loader , optimizer , loss_fn , device , epochs):
    '''
    function to train
    param:
    '''
    model.train()
    total_loss = 0

    for batch_idx , (images , _)in enumerate(train_loader):
        # ([32, 1, 28, 28])
        img_1 = images[0].to(device)
        img_2 = images[1].to(device)
        
        print("img1 shape : " , img_1.shape)
        
        #[32, 128]
        h1 , z1 = model(img_1)
        h2 , z2 = model(img_2)
       
        print(z1.shape)

        loss = loss_fn(z1 , z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        return total_loss / len(train_loader)
        
if __name__ == "__main__":
    
    def main():
        batch_size = 32
        epochs = 100
        learning_rate = 3e-4
        temperature = 0.5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = DataAug(img_size=28)
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
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


        model = SimCLR('resnet18', out_dim=128).to(device)
        criterion = ContrastiveLoss(batch_size=batch_size, temp=temperature).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
        for epoch in range(epochs):
            train_loss = train_simCLR(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=criterion,
                device=device,
                epochs=epochs
            )
            print(f'Epoch: {epoch}, Average Loss: {train_loss:.4f}')
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, f'checkpoint_epoch_{epoch}.pt')

    main()

        

