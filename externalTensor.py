
import wandb
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

import tensorflow as tf

#TODO fare autoencoder 1 level e anche mod quello di 2 level, vedi tabelle dei valori del livello hidden //
#TODO do ADA and random see slide as number of estimators 


"""
input
hidden one level (change  neuron's number and activaton funtion)
output  softmax activation

"""
def makeModelSimple(neuronNumbers,activation,input_dimension):
    
     
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden")(inputs)
    
    outputs = keras.layers.Dense(2,activation="sigmoid" ,name="predictions")(hidden)
    
    return keras.Model(inputs=inputs, outputs=outputs)

"""
input
hidden one level (change  neuron's number, activaton funtion and parameter of L2 regularization)
output  softmax activation

"""

def makeModelL2(neuronNumbers,activation,input_dimension,pl2):
    
    
    inputs = keras.Input(shape=(input_dimension,), name="input")
    hidden = keras.layers.Dense(neuronNumbers, activation=activation,name="hidden",
    kernel_regularizer=tf.keras.regularizers.L2(l2=pl2) )(inputs)
    
    outputs = keras.layers.Dense(2,name="predictions")(hidden)

    return keras.Model(inputs=inputs, outputs=outputs)
"""
input
hidden one level (change  neuron's number, activaton funtion )
dropout (change percentage of drop )
output  softmax activation

"""


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

 
def trainModelCrossValidation(res,X,Y, x_validate, y_validate, model,kf,epochs=50,batch_size=50):

    loss_array=[]
    accuracy_array=[]
    val_loss_array=[]
    val_accuracy_array=[]
    
   
    validation_data=(np.asarray(x_validate), np.asarray(y_validate))
    for train_index , _ in kf.split(X):
        train_x  = X.iloc[train_index,:]
        train_y  = Y[train_index]
        history=model.fit(train_x,train_y,batch_size=batch_size,verbose=False,validation_data=validation_data,epochs=epochs)



        accuracy_array.append(history.history['accuracy'])
        loss_array.append(history.history['loss'])
        val_accuracy_array.append(history.history['val_accuracy'])
        val_loss_array.append(history.history['val_loss'])
        

    
    res['loss']=np.mean(loss_array) 
    res['acc']=np.mean(accuracy_array)
    res['val_loss']=np.mean(val_loss_array) 
    res['val_acc']=np.mean(val_accuracy_array)

    res_hisory=[ np.mean(loss_array,axis=0),np.mean(accuracy_array,axis=0),np.mean(val_loss_array,axis=0),np.mean(val_accuracy_array,axis=0)]    
    return res,res_hisory

def wandbWrite(project_name,best_model_array,best_config_array,x_train,y_train,x_test,y_test,batch_size,epochs):
   
    for model,config in zip(best_model_array,best_config_array):
        run = wandb.init(project=project_name, config=config)
        name="onehidden_"+str(config["num_neurons"])
        run.name=name
        config = wandb.config
        
        history=model.fit(x_train,y_train,batch_size=batch_size,verbose=False,validation_data=(x_test, y_test),epochs=epochs)
        for epoch in range (epochs):

            wandb.log({'epochs': epoch,
                'loss': history.history['loss'][epoch],
                'acc': float(history.history['accuracy'][epoch]), 
                'val_loss': history.history['val_loss'][epoch],
                'val_acc':float(history.history['val_accuracy'][epoch])})
        
        run.finish()   




  
# def trainModelCrossValidation(res,X,Y, x_validate, y_validate, model,kf,epochs=50,batch_size=50):

#     loss_array=[]
#     accuracy_array=[]
  
#     model_array=[]
   
#     validation_data=(np.asarray(x_validate), np.asarray(y_validate))
#     for train_index , _ in kf.split(X):
#         train_x  = X.iloc[train_index,:]
#         train_y  = Y[train_index]
#         history=model.fit(train_x,train_y,batch_size=batch_size,verbose=False,epochs=epochs)
#         accuracy_array.append(round(np.mean(history.history['accuracy']),4))
#         loss_array.append(round(np.mean(history.history['loss']),4))
        
#         model_array.append(model)
        
#     index=accuracy_array.index(max(accuracy_array))
#     history=model_array[index].evaluate(np.asarray(x_validate), np.asarray(y_validate),verbose=False,batch_size=batch_size)
#     res['loss']=loss_array[index] 
#     res['acc']=accuracy_array[index]
#     res['val_loss']= round(history[0],4)
#     res['val_acc']=round(history[1],4)       