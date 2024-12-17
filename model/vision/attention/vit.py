import torch
import torch.nn as nn

'''
img => (#batch , #channels , height , width)
patch => (#batch , #pathches , patch_dim)
#patches = patch_size **2
patch_dim = (height * width * channels)/#patches
'''

class PatchEmbedding(nn.Module):
    def __init__(self, img_size , patch_size , in_channels , embed_dim):
        super(PatchEmbedding , self).__init__()
        self.patch_size = patch_size
        self.num_pathces = (img_size//patch_size) ** 2
        self.proj = nn.Conv2d(in_channels , embed_dim , kernel_size=patch_size , stride=patch_size)

    def forward(self , x):
        x = self.proj(x)
        x = x.flatten(2) # (batch size , embed_dim , n_patches)
        x = x.transpose(1,2) # (batch size , n_patches , embded dim)
        return x


class Attention(nn.Module):
    def __init__(self , embed_dim , num_heads):
        super(Attention , self).__init__()
        self.num_heads = num_heads
        self.scale = embed_dim ** -0.5
        self.qkv = nn.Linear(embed_dim , embed_dim * 3 , bias=False)
        self.attention_dropout  =nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim , embed_dim)


    def forward(self , x):

        B , N ,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads , C // self.num_heads).permute(2,0,3,1,4)
        q  , k , v = qkv[0] , qkv[1] , qkv[2]

        attn = (q @ k.transpose(-2 , -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out
    
class MLP(nn.Module):
    def __init__(self , embed_dim , mlp_dim , dropout=0.1):
        super(MLP , self).__init__()
        self.fc1 = nn.Linear(embed_dim , mlp_dim)
        self.fc2 = nn.Linear(mlp_dim , embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self , x):
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self,  embed_dim , num_heads , mlp_dim , dropout=0.1):
        super(TransformerEncoderLayer , self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(embed_dim=embed_dim , num_heads= num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim , mlp_dim=mlp_dim , dropout=dropout)

    def forward(self ,x):
        x = x + self.attention(self.norm1(x))  # Add & Norm 1
        x = x + self.mlp(self.norm2(x))  # Add & Norm 2
        return x
    

class vit(nn.Module):
    def __init__(self , img_size=224 , patch_size = 16 , in_channels=3 , num_classes=1000 , embed_dim=768 , depth=12 , num_heads=12 , mlp_dim=3072):

        super(vit , self).__init__()
        self.patch_embed = PatchEmbedding(img_size , patch_size , in_channels , embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1 + self.patch_embed.num_pathces , embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim , num_heads , mlp_dim)
                for _ in range(depth)

            ]
        )


        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim , num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed , std=0.02)
        nn.init.trunc_normal_(self.cls_token , std=0.02)

        for p in self.parameters():
            if p.dim() >1:
                nn.init.xavier_uniform_(p)
        

    def forward(self ,x ):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B , -1 , -1)
        x = torch.cat((cls_token , x) , dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)


        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)

        cls_output = x[: , 0]
        out = self.fc(cls_output)
        return out



if __name__ == '__main__':
    image = torch.randn(10 , 3 , 225 ,225)
    model = vit(image , patch_size=15)
    print(model(image).shape)


