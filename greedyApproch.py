




import tensorflow as tf
from tensorflow import keras
import winsound
import numpy as np
import externalTensor as exT
from sklearn.model_selection import learning_curve, train_test_split
import external as ext
from boxScore import boxScore
from codetiming import Timer
from sklearn import preprocessing



# Prepare the training dataset

years="2019-20"
# stats="advance"
stats='advance'
box_score=boxScore(years,stats)
x_train, x_test, y_train, y_test = train_test_split(box_score.dfBoxscores,box_score.LabelResult,test_size=0.076,random_state=8 )
tmp=[[x,y] for x,y in zip(list(x_test['ID']),list(x_test['ID_O'])  ) ]

x_test=ext.uniteBoxScores(tmp,box_score)

del x_train['ID']
del x_train['ID_O']
del x_train['HOME']
del x_train['HOME_O']



#This method is used for cross-validation:
array_train,array_test=ext.separation(x_train,y_train)


activation=['relu','sigmoid']

number_neurons=[10,30,50,100,150,200]
# number_neurons=[10]
possible_learning_rate=[0.0001,0.001,0.01]
# possible_learning_rate=[0.001]

# modelsResult=[]
t = Timer()
t.start()

for nN in number_neurons:
    bestRes={'val_acc':0}
    for el in possible_learning_rate:
        for act in activation:
                # Initialize model
                model = exT.makeModel(nN,act,stats)
                # Instantiate an optimizer to train the model.
                optimizer = keras.optimizers.SGD(learning_rate=el)
                # Instantiate a loss function.
                loss_fn = tf.keras.losses.BinaryCrossentropy()
                res={
                    "learning_rate":el,
                    'num_neurons':nN,
                    'acti_fun':act
                }
                
                model.compile(optimizer=optimizer,loss=loss_fn,metrics=['accuracy'])

                res=exT.trainBasic(res,array_train,array_test,
                    x_test, y_test, 
                    model,
                    500
                
                )
                #Saved the best model based on the accuracy
                if(res['val_acc']>bestRes['val_acc']):
                    bestModel=model
                    bestRes=res
                        
                    
                
    print("Best model",bestRes)
    f = open("saved_model_"+stats+"/summary.txt", "a")
    tmpName=str(bestRes['num_neurons'])+"_"+str(bestRes['acti_fun'])+"_"+str(bestRes['learning_rate'])+"_LOSS_"+str(bestRes['loss'])+"_ACC_"+str(bestRes['acc'])+"_LOSSVAL_"+str(bestRes['val_loss'])+"_ACCVAL_"+str(bestRes['val_acc'])  
    f.write(tmpName+"\n")
    f.close()
    # bestModel.save("saved_model_"+stats+"/"+str(bestRes['num_neurons']))


t.stop()
winsound.Beep(440,3000)










