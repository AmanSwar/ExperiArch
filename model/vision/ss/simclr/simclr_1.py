import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset



class SimClrAugmentation:

    def __init__(self , image_size):
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __call__(self , x):
        return self.transforms(x) , self.transforms(x)
    


class SimClr(nn.Module):
    def __init__(self , base_model='resnet18' , out_dim=128):
        super(SimClr , self).__init__()

        # backbone , dynamics architecture
        self.encoder = models.__dict__[base_model](pretrained=False)
        in_f = self.encoder.fc.in_features

        #making fully connected layer  redundant 
        self.encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(in_f , 512),
            nn.ReLU(),
            nn.Linear(512 , out_dim)

        )

    def forward(self , x):
        h = self.encoder(x)
        z = self.projection_head(h)

        return h ,z 
    

# loss function

def nt_xent_loss(z_i , z_j , temp=0.5):

    batch_size = z_i.size(0)

    # normalize embeddings
    z_i = F.normalize(z_i , dim=1)
    z_j = F.normalize(z_j , dim=1)

    # cosine similarity between z_i , z_j
    positive_sim = torch.exp(torch.sum(z_i * z_j , dim=1) / temp)

    #negative
    representation = torch.cat([z_i , z_j] , dim=0)
    similarity_mat = torch.exp(torch.mm(representation , representation.t()) / temp)

    mask = ~torch.eye(2* batch_size , device=z_i.device).bool()
    negative_sim = similarity_mat[mask].view(2*batch_size , -1).sum(dim=1)


    loss = -torch.log(positive_sim / negative_sim[:batch_size])
    loss = loss.mean()

    return loss

def train_simclr(model , dataloader , optimizer , temp=0.5 , epochs=10):
    model.train()

    for ep in range(epochs):
        epoch_loss = 0.0

        for images , _ in  dataloader:
            images = torch.cat(images, dim=0)
            images = images.cuda()
            # Forward pass
            h, z = model(images)
            z_i, z_j = z.chunk(2)

            # Compute loss
            loss = nt_xent_loss(z_i, z_j, temp)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{ep+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples=1000, image_size=224):
            self.data = torch.randn(num_samples, 3, image_size, image_size)
            self.labels = torch.zeros(num_samples)  # Dummy labels
            self.augment = SimClrAugmentation(image_size)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.augment(self.data[idx]), self.labels[idx]

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize model, optimizer
    model = SimClr(base_model='resnet18', out_dim=128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Train model
    train_simclr(model, dataloader, optimizer, temperature=0.5, epochs=10)