'''
MAIN AIM -
    create a function that can be called in for testing various archs which uses MNIST dataset for testing
    params{
        *necessary*
        model
        batch_size
        loss_fn
        optim
        epochs
        save_model

        *default*
        trans

    }

    output{
        test and valid accuracy
        test accuracy
        train , valid graph
        loss graph

    }

'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

def get_mnist_dataloaders(batch_size=64, train_shuffle=True, val_shuffle=False, test_shuffle=False, validation_split=0.2):
    """
    prepare MNIST dataset with transformations and dataloaders, including validation set.
    
    Args:
    - batch_size (int)
    - train_shuffle (bool)
    - val_shuffle (bool)
    - test_shuffle (bool)
    - validation_split (float)
    
    Returns:
    - train_loader (DataLoader)
    - val_loader (DataLoader)
    - test_loader (DataLoader)
    """
    # transformation
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    
    # Download 
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',  
        train=True,     
        download=True,  
        transform=transform  
    )
    
    # Calculate validation set size
    train_size = int((1 - validation_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # Split training dataset into train and validation sets
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(6969)
    )
    
    # Download test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,    # use test set
        download=True, 
        transform=transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=train_shuffle,
        num_workers=2  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=val_shuffle,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=test_shuffle,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader




def play_mnist(model , batch_size , loss_fn , optimizer , num_epochs , save_model=False):
    train_dl , valid_dl , test_dl = get_mnist_dataloaders(batch_size=batch_size)
    DEVICE = torch.device('cuda')

    model.to(DEVICE)

    start_time = time.time()
    ema_val_acc = 0

    print("epoch | train_loss | val_loss | train_acc | val_acc | ema_val_acc | total_time_seconds")

    for epoch in tqdm(range(num_epochs)):

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for input , label in tqdm(train_dl , desc=f"Epoch {epoch+1} - Train"):

            label = torch.tensor(label)
            input , label = input.to(DEVICE) , label.to(DEVICE)
            # init optim
            optimizer.zero_grad()

            # model output
            output = model(input)

            # calculate the loss
            loss = loss_fn(output , label)
            loss.backward()

            # stepping up the optimizer
            optimizer.step()

            train_loss += loss.item()
            _ , predicted = output.max(1)
            train_total += label.size(0)
            train_correct += predicted.eq(label).sum().item()

        train_loss /= len(train_dl)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0


        with torch.no_grad():

            for input, label in valid_dl:
                label = torch.tensor(label)
                input, label = input.to(DEVICE).float(), label.to(DEVICE)
                out = model(input)

                loss = loss_fn(out , label)

                val_loss += loss.item()
                _ , predicted = out.max(1)

                val_total += label.size(0)
                val_correct += predicted.eq(label).sum().item()

            val_loss /= len(valid_dl)
            val_acc = val_correct / val_total

        total_time = time.time() - start_time

        print(f"{epoch:5d} | {train_loss:.4f} | {val_loss:.4f} | {train_acc:.4f} | {val_acc:.4f}| {total_time:.4f}")

    if save_model:
        PATH = f"{model}_weights_{batch_size}_{lr}_{loss_fn}_{optimizer}_{num_epochs}.pth"

        torch.save(model.state_dict() , PATH)




