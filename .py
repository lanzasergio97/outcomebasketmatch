
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch
import torch.nn.functional as F # All functions that don't have any parameters



class NN(nn.Module):
    
    def __init__(self, input_size, size_hidden_level,acti):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, size_hidden_level)
        self.fc2 = nn.Linear(size_hidden_level, 2)
        self.acti=acti
    def forward(self, x):
        if( self.acti=="relu"):
            x = F.relu(self.fc1(x))
        if(self.acti=="sigmoid"):
            x = F.sigmoid(self.fc1(x))

        x =F.softmax(input=self.fc2(x),dim=1) 
        return x 
    def MyTrain (self,config,train_LL,test_LL,optimizer,device):
        print("Start")
        loss_array=[]
        acc_array=[]
        loss_val_array=[]
        acc_val_array=[]
        total_elements=len(train_LL.dataset)

        for epoch in range(config["epochs"]):
            running_loss=0
            correct_predictions=0
            
            for batch_idx, (data, targets) in enumerate(train_LL):
                self.train()
                # Get data to cuda if possible
                data = data.to(torch.float32)
                data=data.to(device=device)
                targets = targets.to(device=device)
                
                # forward propagation
                scores = self(data)

                loss = F.binary_cross_entropy(scores, targets)
               
                # zero previous gradients
                optimizer.zero_grad()
                
                
                # back-propagation
                loss.backward()

                # gradient descent or adam step
                optimizer.step()
                
                        
                correct_predictions +=torch.sum(torch.argmax(scores, dim=1)==torch.argmax(targets, dim=1))
                
                running_loss += loss.item()
            loss_val,arr_val=self.test_check(test_LL=test_LL,device=device)
            loss_val_array.append(loss_val)
            acc_val_array.append(arr_val)
            loss_array.append(round(running_loss/len(train_LL),3))
            acc_array.append(round(correct_predictions.item()/total_elements,3))

        config["loss"]=loss_array[-1]
        config["acc"]=acc_array[-1]
        config["val_loss"]=loss_val_array[-1]
        config["val_acc"]=acc_val_array[-1]
        return loss_array,acc_array,loss_val_array,acc_val_array

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
            
        return loss,acc


"""

for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
  print('------------fold no---------{}----------------------'.format(fold))
  train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
  test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
 
  trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
  testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size, sampler=test_subsampler)
 
  model.apply(reset_weights)
 
  for epoch in range(1, epochs + 1):
    train(fold, model, device, trainloader, optimizer, epoch)
    test(fold,model, device, testloader)

"""