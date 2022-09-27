
import wandb
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import KFold
import tensorflow as tf
"""
input
hidden one level (change  neuron's number and activaton funtion)
output fixed softmax activation

"""
def makeModel(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
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



def trainBasic(res,X,Y, x_validate, y_validate, model,kf,epochs=10):
    
    lossArray=[]
    accuracyArray=[]
    valAccuracyArray=[]
    valLossArray=[]
    
    
    
    validation_data=(np.asarray(x_validate), np.asarray(y_validate))
    
    for train_index , _ in kf.split(X):
        trainX  = X.iloc[train_index,:]
        trainY  = Y[train_index]
        
        
        tik=model.fit(trainX,trainY,batch_size=200,verbose=False,validation_data=validation_data,epochs=epochs)
        
      
        
        accuracyArray.append(round(np.mean(tik.history['accuracy']),4))
        lossArray.append(round(np.mean(tik.history['loss']),4))
        valLossArray.append(round(np.mean(tik.history['val_loss']),4))
        valAccuracyArray.append(round(np.mean(tik.history['val_accuracy']),4))


        

    
    
    
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