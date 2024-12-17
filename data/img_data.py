'''
load img in format 

class A 
    - img 1
    - img 2
class B
    - img 1 
    - img 2


'''



import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


DATA_DIR = ""


class ImageDataset(Dataset):

    def __init__(self , dir , transform=None):
        DATA_DIR = dir
        self.class_list = []
        self.img_list = []
        self.img_class_list = []
        self.transform = transform

        print("READING DATA...")
        for classes in tqdm(os.listdir(dir)):
            self.class_list.append(classes)
            img_path = os.path.join(classes , dir)

            for img in tqdm(os.listdir(img_path)):

                self.img_list.append(img)
                self.img_class_list.append(classes)

    def __getitem__(self , index):
        

    
