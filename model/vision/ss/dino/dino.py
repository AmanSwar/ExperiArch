import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms


class MultiCrop(nn.Module):
    """
    
    """

    def __init__(self , encoder , head):
        super(MultiCrop , self).__init__()
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.head = head

    def forward(self , x):
        crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True

        )[1] , 0)

        start_index = 0

        for end_index in crops:
            _out = self.backbone(torch.cat(x[start_index : end_index]))
            if start_index == 0:
                output = _out
            
            else:
                output =  torch.cat((output , _out))

            start_index = end_index

        output = self.head(output)
        return output
    
class DINOHead(nn.Module):

    def __init__(self , in_dim , out_dim , hidden_dim=2048 , bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim , bottleneck_dim),

        )

        self.last_later = nn.Linear(bottleneck_dim , out_dim)
        self.apply(self._init_weights)

    def _init_weights(self , m):
        if isinstance(m , nn.Linear):
            nn.init.trunc_normal_(m.weight , std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias , 0)

    def forward(self , x):
        x = self.mlp(x)
        x = F.normalize(x , dim=-1)
        x = self.last_layer(x)
        return x
    

class DINO_loss(nn.Module):

    def __init__(self, out_dim , teach_temp=0.04 , student_temp=0.1 , center_mom=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teach_temp = teach_temp
        self.center_mom = center_mom

        self.register_buffer("center" , torch.zeros(1, out_dim))

    def forward(self , student_out , teacher_out):
        student_out = student_out / self.student_temp
        teacher_out = teacher_out / self.teach_temp

        student_probs = F.softmax(student_out ,dim=-1)
        teacher_probs = F.softmax((teacher_out - self.center) ,dim=-1).detach()

        loss = torch.sum(-teacher_probs * torch.log_softmax(student_out , dim=-1) , dim=-1).mean()

        self.update_center(teacher_out)

        return loss

    @torch.no_grad()
    def update_center(self , teacher_output):

        batch_center = torch.sum(teacher_output , dim=0 , keepdim=True)
        batch_center = batch_center / len(teacher_output)

        self.center = self.center * self.center_mom + batch_center * (1 - self.center_mom)

class DINO(nn.Module):

    def __init__(self , student , teacher , out_dim , teacher_temp=0.04 , student_temp=0.1 , center_mom=0.9):
        super().__init__()

        self.student = MultiCrop(
            student , 
            DINOHead(student.num_features , out_dim)
        )
        self.teacher = MultiCrop(
            teacher,
            DINOHead(teacher.num_features , out_dim)
        )

        # no gradients for teacher 
        for p in self.teacher.parameters():
            p.requires_grad = False


        self.loss_fn = DINO_loss(
            out_dim=out_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_mom=center_mom,
        )

    @torch.no_grad()
    def update_teacher(self , m=0.996):
        for param_q , param_k in zip(self.student.parameters() , self.teacher.parameters()):
            param_k.data.mul_(m).add_((1-m) * param_q.detach().data)

    def forward(self , x):

        student_output = self.student(x)
        teacher_output = self.teacher(x[:2]) # only the larger global views
        loss = self.loss_fn(student_output , teacher_output)

        return loss



def dino_transforms():
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
    ])
    
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
    ])
    
    return global_transform, local_transform


def train_one_epoch(model , data_loader , optimizer , device , epoch):

    model.train()
    total_loss= 0
    for batch_idx , img in enumerate(data_loader):

        img = [im.to(device) for im in img]

        loss = model(img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.update_teacher()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"epoch : {epoch} , batch : {batch_idx} , loss : {loss.item():.2f}")

    return total_loss / len(data_loader)


            

    

         
        
    




        

