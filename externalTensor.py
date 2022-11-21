
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
#TODO fare autoencoder 1 level e anche mod quello di 2 level, vedi tabelle dei valori del livello hidden 
#TODO mod  makModelL2, param must change in 0.001,0.01,0.0001 //
#TODO mod makeModelDropout, p must change in 0.5 0.6 0.65 //
#TODO do ADA and random see slide as number of estimators 

def makeModelSimple(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    outputs = keras.layers.Dense(2,activation="sigmoid" ,name="predictions")(hidden)
    
    return keras.Model(inputs=inputs, outputs=outputs)

def makeModelL2(neuronNumbers,activation,input_dimension,pl2):
    
    
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden",
    kernel_regularizer= (pl2))(inputs)
    
    outputs = keras.layers.Dense(2,name="predictions")(hidden)

    return keras.Model(inputs=inputs, outputs=outputs)



def makeModelDropout(neuronNumbers,activation,input_dimension,p):
    
    
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    droput= keras.layers.Dropout(p,input_shape=(input_dimension,))(hidden)
    outputs = keras.layers.Dense(2,activation=tf.keras.activations.softmax ,name="predictions")(droput)

    return keras.Model(inputs=inputs, outputs=outputs)

def makeModelTwoLevel(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    hidden1 = keras.layers.Dense(neuronNumbers*2, activation=activation,name="hidden1")(hidden)
    droput= keras.layers.Dropout(.06,input_shape=(input_dimension,))(hidden1)
    outputs = keras.layers.Dense(2,activation=tf.keras.activations.softmax ,name="predictions")(droput)

    return keras.Model(inputs=inputs, outputs=outputs)

    
def autoencoderOneLevel(input_dimension,neurons_hidden,neurons_code):

    inputs = keras.Input(shape=(input_dimension,), name="input")


    encoded = keras.layers.Dense(neurons_hidden, activation="relu")(inputs)

    encoded = keras.layers.Dense(neurons_code, activation="relu")(encoded)

    decoded = keras.layers.Dense(neurons_hidden, activation="relu")(encoded)

    
    
    decoded = keras.layers.Dense(input_dimension,activation= "sigmoid" )(decoded)
    
    autoencoder=keras.Model(inputs, decoded)
    encoder=keras.Model(inputs, encoded)
    
    encoded_input = keras.Input(shape=(neurons_hidden,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    return autoencoder, encoder, decoder
def autoencoderTwoLevels(input_dimension,neurons_hidden_first_level,neurons_hidden_second_level,neurons_code):

    inputs = keras.Input(shape=(input_dimension,), name="input")


    encoded = keras.layers.Dense(neurons_hidden_first_level, activation="relu")(inputs)

    encoded = keras.layers.Dense(neurons_hidden_second_level, activation="relu")(encoded)

    encoded = keras.layers.Dense(neurons_code, activation="relu")(encoded)

    decoded = keras.layers.Dense( neurons_hidden_second_level, activation="relu")(encoded)

    decoded = keras.layers.Dense(neurons_hidden_first_level, activation="relu")(decoded)
    
    decoded = keras.layers.Dense(input_dimension,activation= "sigmoid" )(decoded)
    
    autoencoder=keras.Model(inputs, decoded)
    encoder=keras.Model(inputs, encoded)
    
    encoded_input = keras.Input(shape=(neurons_hidden_first_level,))
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
        
        
        history=model.fit(trainX,trainY,batch_size=200,verbose=False,validation_data=validation_data,epochs=epochs)
        
      
        
        accuracyArray.append(round(np.mean(history.history['accuracy']),4))
        lossArray.append(round(np.mean(history.history['loss']),4))
        valLossArray.append(round(np.mean(history.history['val_loss']),4))
        valAccuracyArray.append(round(np.mean(history.history['val_accuracy']),4))


    res['loss']= round(np.mean(lossArray),4)
    res['acc']= round(np.mean(accuracyArray),4)
    res['val_loss']= round(np.mean(valLossArray),4)
    res['val_acc']=round(np.mean(valAccuracyArray),4)

    
    return res

def trainModelCrossValidation(res,X,Y, x_validate, y_validate, model,kf,epochs=50,batch_size=50):
    lossArray=[]
    accuracyArray=[]
  
    modelArray=[]
   
    
    for train_index , _ in kf.split(X):
        trainX  = X.iloc[train_index,:]
        trainY  = Y[train_index]
        history=model.fit(trainX,trainY,batch_size=batch_size,verbose=False,epochs=epochs)
        accuracyArray.append(round(np.mean(history.history['accuracy']),4))
        lossArray.append(round(np.mean(history.history['loss']),4))
        
        modelArray.append(model)
        
    index=accuracyArray.index(max(accuracyArray))
    history=modelArray[index].evaluate(np.asarray(x_validate), np.asarray(y_validate),verbose=False,batch_size=batch_size)
    res['loss']=lossArray[index] 
    res['acc']=accuracyArray[index]
    res['val_loss']= round(history[0],4)
    res['val_acc']=round(history[1],4)

    return res
