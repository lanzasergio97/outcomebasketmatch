
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch
import torch.nn.functional as F # All functions that don't have any parameters
import numpy as np
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.

class NN(nn.Module):
    
    def __init__(self, input_size, size_hidden_level,act,learning_rate,pl2=0):
        
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, size_hidden_level)
        self.fc2 = nn.Linear(size_hidden_level, 2)
        self.acti=act
        self.loss=F.binary_cross_entropy_with_logits
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=pl2)
        self.optimizer=optimizer
    def forward(self, x):
        if( self.acti=="relu"):
            x = F.relu(self.fc1(x))
        if(self.acti=="sigmoid"):
            x = torch.sigmoid(self.fc1(x))

        x =F.softmax(input=self.fc2(x),dim=1) 
        return x 
    def reset_weights(self):
        if isinstance(self, nn.Conv2d) or isinstance(self, nn.Linear):
            self.reset_parameters()


    def MyTrain (self,kf,config,best_config_tmp,best_history_tmp,train_LL,test_LL,device):
        
        loss_array_fold=torch.tensor([]).to(device=device)
        acc_array_fold=torch.tensor([]).to(device=device)
        loss_val_array_fold=torch.tensor([]).to(device=device)
        acc_val_array_fold=torch.tensor([]).to(device=device)
        
        for train_idx,_ in kf.split(train_LL.dataset):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            train_loader_fold = torch.utils.data.DataLoader(
                      train_LL.dataset, 
                      batch_size=config["batch_size"], sampler=train_subsampler)
            total_elements=len(train_loader_fold.dataset)
            # Reset the weights after a run of folds
            self.reset_weights()
            
            # Arrays for loss and accuracy 
            loss_array_epoch=torch.tensor([]).to(device=device)
            acc_array_epoch=torch.tensor([]).to(device=device)
            loss_val_array_epoch=torch.tensor([]).to(device=device)
            acc_val_array_epoch=torch.tensor([]).to(device=device) 

            for _ in range(config["epochs"]):
                running_loss=0
                running_accuracy=0
                
                for data, targets in train_loader_fold:
                    self.train()
                    # Get data to cuda if possible
                    data = data.to(torch.float32)
                    data=data.to(device=device)
                    targets = targets.to(device=device)
                    
                    # forward propagation
                    scores = self(data)
                    loss = self.loss(scores, targets)
                    # zero previous gradients
                    self.optimizer.zero_grad()
                    
                    # back-propagation
                    loss.backward()

                    # gradient descent or adam step
                    self.optimizer.step()
                    # Accuracy and loss values       
                    running_accuracy +=torch.sum(torch.argmax(scores, dim=1)==torch.argmax(targets, dim=1))
                    running_loss += loss.item()

                # Accuracy and loss values also for validation set
                loss_val,acc_val=self.test_check(test_LL=test_LL,device=device)
                
                # Values for a single Epoch of loss and accuracy both for training and validation
                
                loss_val_array_epoch=torch.cat((loss_val_array_epoch, loss_val), 1)
                acc_val_array_epoch=torch.cat((acc_val_array_epoch, acc_val), 1)
                loss_array_epoch=torch.cat((loss_array_epoch,torch.tensor([[running_loss/len(train_loader_fold)]]).to(device=device) ), 1)
                acc_array_epoch=torch.cat((acc_array_epoch,torch.tensor([[running_accuracy.item()/total_elements]]).to(device=device)), 1)

            # Values for a single set of Fold of loss and accuracy both for training and validation
            loss_val_array_fold=torch.cat( (loss_val_array_fold,loss_val_array_epoch),0)
            acc_val_array_fold=torch.cat( (acc_val_array_fold,acc_val_array_epoch),0)
            loss_array_fold=torch.cat((loss_array_fold,loss_array_epoch),0)
            acc_array_fold=torch.cat((acc_array_fold,acc_array_epoch),0)

        #Check if this models is the better than the previous based on accuracy on the validation set
        tmp_val=torch.mean(acc_val_array_fold[:,-1],dtype=torch.float32)
        if(tmp_val>best_config_tmp['val_acc']):
            # Change the configuration file with the values of the best model
            config["loss"]=torch.mean(loss_array_fold[:,-1],dtype=torch.float32).item()
            config["acc"]=torch.mean(acc_array_fold[:,-1],dtype=torch.float32).item()
            config["val_loss"]=torch.mean(loss_val_array_fold[:,-1],dtype=torch.float32).item()
            config["val_acc"]=tmp_val.item()
            # Create the history file, it will be used for draw the graph with WandB
            best_history_tmp=[ loss_array_fold,acc_array_fold,loss_val_array_fold,acc_val_array_fold] 
            best_config_tmp=config
        
        return best_config_tmp,best_history_tmp



    def test_check(self,test_LL,device):
        correct_predictions = 0
        running_loss = 0
        total_elements=len(test_LL.dataset)
        
        self.eval()
        
        
        with torch.no_grad():
            for data, targets in test_LL:
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = self(data)
                loss = F.binary_cross_entropy(scores, targets)
                correct_predictions +=torch.sum(torch.argmax(scores, dim=1)==torch.argmax(targets, dim=1))
                running_loss += loss.item()

            
            loss=(running_loss/len(test_LL))
            acc=(round(correct_predictions.item()/total_elements,3))
            
        return torch.tensor([[loss]]).to(device=device),torch.tensor([[acc]]).to(device=device)


