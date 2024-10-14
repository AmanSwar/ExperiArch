import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader , Subset
from tqdm import tqdm

BATCH_SIZE = 32
DEVICE = torch.device("cuda")
N_EPOCH = 100



trans = transforms.Compose([
    transforms.ToTensor(),
])


train_d = MNIST(root='./data' , train=True , download=True , transform=trans)
test_d = MNIST(root='./data' , train=False , download=True , transform=trans)

mnist_valid_dataset = Subset(train_d , torch.arange(10000))

mnist_train_dataset = Subset(train_d , torch.arange(10000 , len(train_d)))

BATCH_SIZE = 32


train_dl = DataLoader(mnist_train_dataset , batch_size=BATCH_SIZE , shuffle=True , pin_memory=True)
valid_dl = DataLoader(mnist_valid_dataset , batch_size=BATCH_SIZE , shuffle=False , pin_memory=True)


model = None
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters() , lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()


loss_hist_train = [0] * N_EPOCH
loss_hist_valid = [0] * N_EPOCH
acc_hist_valid = [0] * N_EPOCH
acc_hist_train = [0] * N_EPOCH

for epoch in tqdm(range(N_EPOCH)):
    model.train()

    for x , y in train_dl:
        x , y = x.to(DEVICE) , y.to(DEVICE)

        output = model(y)
        loss = loss_fn(output , y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train[epoch] += loss.item() * y.size(0)
        is_correct = (torch.argmax(output , dim=1) == y)
        acc_hist_train[epoch] += is_correct.sum()

    loss_hist_train[epoch] /= len(train_dl.dataset)
    acc_hist_train[epoch] /= len(train_dl.dataset)


    model.eval()

    with torch.no_grad():
        for x, y in valid_dl:
            x , y = x.to(DEVICE) , y.to(DEVICE)

            output = model(x)
            loss = loss_fn(output , y)
            loss_hist_valid[epoch] += loss.item() * y.size(0)
            is_correct = (torch.argmax(output, dim=1) == y).float()
            acc_hist_valid[epoch] += is_correct.sum()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        acc_hist_valid[epoch] /= len(valid_dl.dataset)


    print(f"epoch {epoch+1} | accuracy {acc_hist_train[epoch]:.2f} | val_acc {acc_hist_valid[epoch]:.2f}")

    
