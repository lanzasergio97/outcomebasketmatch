import torch
import torch.nn as nn
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
import numpy as np
from  boxScore  import boxScore
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader # data management 

from NeuralNetwork import NN
from DatasetPrivate import MyDataset





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

years="2018-19"
stats="traditional"
# stats='advance'
box_score=boxScore(years,stats)
activation=['relu','sigmoid']
number_neurons=[10,30,50,100,150,200]

possible_learning_rate=[0.0001,0.001,0.01]
num_epochs = 500
batch_size=150
x_train, x_test, y_train, y_test=box_score.separation()
y_train=np.array(y_train)
y_test=np.array(y_test)

input_size=len(x_train.columns)
train_LL=DataLoader(MyDataset(x_train,y_train),batch_size=batch_size,shuffle=False)
test_LL=DataLoader(MyDataset(x_test,y_test),batch_size=batch_size,shuffle=False)
kf = KFold(n_splits=10, random_state=1, shuffle=True)

for num in number_neurons:
    best_config_tmp={'val_acc':0}
    best_history=[]
    for el in possible_learning_rate:
        for act in activation:
                # Initialize model
                model = NN(input_size=input_size,size_hidden_level=num,acti="relu").to(device)

                
                # Instantiate an optimizer to train the model.
                optimizer = optim.SGD(model.parameters(), lr=el)
                config={
                    "learning_rate":el,
                    'num_neurons':num,
                    'acti_fun':act,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "architecture": "OneHiddenLayer",
                    "dataset":years+"_"+stats
                }
                
                config,history=model.MyTrain(kf=kf,config=config,train_LL=train_LL,test_LL=test_LL,optimizer=optimizer,device=device)
               
                #Saved the best model based on: accuracy, number of neurons of first hidden level and type of stats
                if(config['val_acc']>best_config_tmp['val_acc']):
                    best_history=history
                    best_config_tmp=config
                    
    print("Best model",best_config_tmp)

# model.MyTrain(config,train_LL,optimizer,device)



"""
"""