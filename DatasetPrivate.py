import torch

from torch.utils.data import Dataset

#TODO CHECK x and y types
class MyDataset(Dataset):
 
  def __init__(self,x,y):

    self.input_dimension=len(x.columns)
    self.x_train=torch.tensor(x.values,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

