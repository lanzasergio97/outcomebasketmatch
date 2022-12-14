
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch
import torch.nn.functional as F # All functions that don't have any parameters
import numpy as np
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.

class Autoencoder(nn.Module):
    
    def __init__(self,input_size,code_size,hidden_first_level_size,hidden_second_level_size=0,learning_rate=0.01):
        super().__init__()
        if(hidden_second_level_size==0):
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_size , hidden_first_level_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_first_level_size, code_size),
                torch.nn.ReLU(),
            
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(code_size, hidden_first_level_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_first_level_size, input_size),
                torch.nn.Sigmoid()
            )
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_size , hidden_first_level_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_first_level_size, hidden_second_level_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_second_level_size, code_size),
                torch.nn.ReLU(),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(code_size, hidden_second_level_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_second_level_size,hidden_first_level_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_first_level_size,input_size ),
                torch.nn.Sigmoid()
            )
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        self.optimizer=optimizer
    def forward(self, x):
        out_en = self.encoder(x)

        out = self.decoder(out_en)
        return out, out_en


    def MyTrain (self,train_LL,device):
        epochs = 20
       
        self.train()
        distance   = nn.MSELoss()
        for _ in range(epochs):
            for (image, label) in train_LL:
           
                
                # Output of Autoencoder
                reconstructed,out_enc = self(image)

                # Calculating the loss function
                loss = distance(reconstructed, image)
                
              
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
         

    def test_check(self,test_LL,device):
         
        result=[]
        labels=[]

        self.eval()
        
        
        with torch.no_grad():
            for image, label in test_LL:
                image = image
                reconstructed,out_enc = self(image)
            
                result.extend(out_enc.detach().numpy())
                labels.extend(label.detach().numpy())

            
            
        
        
        return np.array(result),np.array(labels)
