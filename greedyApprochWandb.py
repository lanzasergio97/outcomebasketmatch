




import tensorflow as tf
from tensorflow import keras
import winsound

import externalTensor as exT
import numpy as np

import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import learning_curve, train_test_split

from boxScore import boxScore
from codetiming import Timer
# wandb.login()



# Prepare the training dataset
BATCH_SIZE = 60
years="2019-20"
stats="traditional"
box_score=boxScore(years,stats)
x_train, x_test, y_train, y_test = train_test_split(box_score.dfBoxscores,box_score.LabelResult,test_size=0.076,random_state=8 )
tmp=[[x,y] for x,y in zip(list(x_test['ID']),list(x_test['ID_O'])  ) ]

box_score=boxScore(years,stats)
x_train, x_test, y_train, y_test=box_score.separation()








# build input pipeline using tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)








activation=['relu','sigmoid']

number_neurons=[10,30,50,100,150,200]
# numberNeurons=[10]
possible_learning_rate=[0.0001,0.001,0.01]
# modelsResult=[]
t = Timer()
t.start()

for nN in number_neurons:
    bestRes={'val_acc':0}
    for el in possible_learning_rate:
        for act in activation:
                config = {
                            "learning_rate": el,
                            "epochs": 500,
                            "batch_size": 200,
                            "log_step":200, 
                            "val_log_step":50,
                            "architecture": "CNN",
                            "dataset": "BoxScore"+years+stats,
                            "activationFunction":act
                    }

                
                projectName='Stats_'+stats+"_"+str(nN)+"Neurons"
                name=act+"_learningRate_"+str(el)
                # run = wandb.init(project=projectName, config=config,name=name)
                # config = wandb.config
                

                # Initialize mod.
                model = exT.makeModel(nN,act,stats)
                
                # Instantiate an optimizer to train the model.
                optimizer = keras.optimizers.SGD(learning_rate=el)
                loss_fn = tf.keras.losses.BinaryCrossentropy()
                res={
                    "learning_rate":el,
                    'num_neurons':nN,
                    'acti_fun':act
                }
                # Instantiate a loss function.
                # model.compile(optimizer=optimizer,loss=loss_fn,metrics=['accuracy'])
                # Prepare the metrics.
                train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
                val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

                res=exT.train(res,loss_fn,train_dataset, val_dataset,  model, optimizer,
                    train_acc_metric, val_acc_metric,
                    config["epochs"], config["log_step"],config["val_log_step"] )
                
                # run.finish()
  


t.stop()
winsound.Beep(440,3000)












#                 # Prepare the metrics.
#                 train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
#                 val_acc_metric = keras.metrics.SparseCategoricalAccuracy()



                # res,modelTmp=exT.train(loss_fn,train_dataset,
                #     val_dataset, 
                #     model,
                #     optimizer,
                #     train_acc_metric,
                #     val_acc_metric,
                #     epochs=config['epochs'], 
                #     log_step=config['log_step'], 
                #     val_log_step=config['val_log_step'],
                #     )
                # res["learning_rate"]=el
                # res['num_neurons']=nN
                # res['acti_fun']=act
                
                # if(bestRes['val_acc']<res['val_acc']):
                #     bestRes=res
                #     bestModel=modelTmp