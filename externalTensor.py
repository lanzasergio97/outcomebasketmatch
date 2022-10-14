
import wandb
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

import tensorflow as tf
"""
input
hidden one level (change  neuron's number and activaton funtion)
output  softmax activation

"""
def makeModelSimple(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    outputs = keras.layers.Dense(2,activation=tf.keras.activations.softmax ,name="predictions")(hidden)

    return keras.Model(inputs=inputs, outputs=outputs)

def makeModelL2(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden",
    kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    
    outputs = keras.layers.Dense(2,name="predictions")(hidden)

    return keras.Model(inputs=inputs, outputs=outputs)



def makeModelDropout(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    droput= keras.layers.Dropout(.5,input_shape=(input_dimension,))(hidden)
    outputs = keras.layers.Dense(2,activation=tf.keras.activations.softmax ,name="predictions")(droput)

    return keras.Model(inputs=inputs, outputs=outputs)

def makeModelTwoLevel(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    hidden1 = keras.layers.Dense(neuronNumbers*2, activation=activation,name="hidden1")(hidden)
    droput= keras.layers.Dropout(.06,input_shape=(input_dimension,))(hidden1)
    outputs = keras.layers.Dense(2,activation=tf.keras.activations.softmax ,name="predictions")(droput)

    return keras.Model(inputs=inputs, outputs=outputs)

def autoencoder(activation,input_dimension,vanish_coefficent):

    inputs = keras.Input(shape=(input_dimension,), name="input")


    encoded = keras.layers.Dense(vanish_coefficent, activation=activation)(inputs)

    encoded = keras.layers.Dense(vanish_coefficent-5, activation=activation)(encoded)
    encoded = keras.layers.Dense(vanish_coefficent-10, activation=activation)(encoded)

    encoded = keras.layers.Dense(5, activation=activation)(encoded)
    decoded = keras.layers.Dense(vanish_coefficent-10, activation=activation)(encoded)
    decoded = keras.layers.Dense(vanish_coefficent-5, activation=activation)(decoded)
    

    decoded = keras.layers.Dense(vanish_coefficent, activation=activation)(decoded)

    
    decoded = keras.layers.Dense(input_dimension,activation="relu" )(decoded)
    
    autoencoder=keras.Model(inputs, decoded)
    encoder=keras.Model(inputs, encoded)
    
    encoded_input = keras.Input(shape=(vanish_coefficent,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    return autoencoder, encoder, decoder
    

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

def train_alternative(res,X,Y, x_validate, y_validate, model,kf,epochs=10):
    lossArray=[]
    accuracyArray=[]
  
    modelArray=[]
   
    
    for train_index , _ in kf.split(X):
        trainX  = X.iloc[train_index,:]
        trainY  = Y[train_index]
        tik=model.fit(trainX,trainY,batch_size=200,verbose=False,epochs=epochs)
        accuracyArray.append(round(np.mean(tik.history['accuracy']),4))
        lossArray.append(round(np.mean(tik.history['loss']),4))
        
        modelArray.append(model)
        
    index=accuracyArray.index(max(accuracyArray))
    tik=modelArray[index].evaluate(np.asarray(x_validate), np.asarray(y_validate),verbose=False,batch_size=200)
    res['loss']=lossArray[index] 
    res['acc']=accuracyArray[index]
    res['val_loss']= round(tik[0],4)
    res['val_acc']=round(tik[1],4)

    return res
