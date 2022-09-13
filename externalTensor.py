from operator import truediv
from tabnanny import verbose
import wandb
import pandas as pd
import numpy as np
from tensorflow import keras

import tensorflow as tf
"""
input
hidden one level (change  neuron's number and activaton funtion)
output fixed softmax activation

"""
def makeModel(neuronNumbers,activation,mode):
    
    if(mode=="traditional"):
        inputStart=22
     
    else:
        inputStart=34
    inputs = keras.Input(shape=(inputStart,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    outputs = keras.layers.Dense(2,activation=tf.keras.activations.softmax ,name="predictions")(hidden)

    return keras.Model(inputs=inputs, outputs=outputs)

def trainStep(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    train_acc_metric.update_state(y, logits)

    return loss_value

def testStep(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_value



def train(result,loss_fn,train_dataset, val_dataset,  model, optimizer,
          train_acc_metric, val_acc_metric,
          epochs=10,  log_step=200, val_log_step=50):
   
    model.compile(optimizer=optimizer,loss=loss_fn)
    for epoch in range(epochs):
        # print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # Iterate over the batches of the dataset
        for _, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = trainStep(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # Run a validation loop at the end of each epoch
        for _, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = testStep(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        # print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        # print("Validation acc: %.4f" % (float(val_acc),))

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
    
        # # ⭐: log metrics using wandb.log
        # wandb.log({'epochs': epoch,
        #             'loss': np.mean(train_loss),
        #             'acc': float(train_acc), 
        #             'val_loss': np.mean(val_loss),
        #             'val_acc':float(val_acc)})
   
    result['loss'] =round(np.mean(train_loss),4)
    result['acc'] =round(float(train_acc),4)
    result['val_loss'] =round(np.mean(val_loss),4)
    result['val_acc'] =  round(float(val_acc),4)

    return result,model



def trainBasic(res,array_train,array_test, x_validate, y_validate, model,epochs=10):
    
    lossArray=[]
    accuracyArray=[]
    valAccuracyArray=[]
    valLossArray=[]
    
    for i in range(len(array_train)):  
        trainX=array_train[i][0]
        trainY=array_train[i][1]
        testX=array_test[i][0]
        testY=array_test[i][1]
        
       

        model.fit(trainX,trainY,batch_size=200,verbose=False,epochs=epochs)
       
        tmp=model.evaluate(np.asarray(testX),np.asarray(testY), verbose=False)
        lossArray.append(tmp[0])
        accuracyArray.append(tmp[1])
        tmp=model.evaluate(np.asarray(x_validate), np.asarray(y_validate), verbose=False)
        valLossArray.append(tmp[0])
        valAccuracyArray.append(tmp[1])

  

    res['loss']= round(np.mean(lossArray),4)
    res['acc']= round(np.mean(accuracyArray),4)
    res['val_loss']= round(np.mean(valLossArray),4)
    res['val_acc']=round(np.mean(valAccuracyArray),4)

   

    # wandb.log({'epochs': epoch,
        #             'loss': np.mean(train_loss),
        #             'acc': float(train_acc), 
        #             'val_loss': np.mean(val_loss),
        #             'val_acc':float(val_acc)})
    return res