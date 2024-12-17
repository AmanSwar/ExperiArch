import torch
from torch.utils.data import DataLoader , Dataset ,Subset

'''
universal dataloader 
    images , text data , csv file

    

'''


'''
Images support
    Dataread('csv' , 'level' , image , )
    - csv 

    - folder
'''

IMG_DIR = ""
LABELS = []

class Dataread:
    '''
    

    output =    
    
    '''
    def __init__(self , format ,data_dir , target_variable , *features):
        self.dir = data_dir
        if format == 'csv':
            self.format = 'csv'
        elif format == 'folder':
            self.format = 'folder'
        else:
            raise Exception("format not supported")
        

        self.target = target_variable
        self.features = features

    def data_csv(self):
        import pandas as pd
        df = pd.read_csv(self.dir)

        for feat in self.features:


        




def get_labels_csv(DATA_DIR):
    import pandas as pd

    data = pd.read_csv(DATA_DIR)


def get_labels_folder():
    pass


label_format = ""

if label_format == 'csv':
    get_labels_csv()

elif label_format == 'folder':
    get_labels_folder()









