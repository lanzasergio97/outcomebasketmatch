




import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.datasets import cifar10
import winsound
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import externalTensor as exT
from wandb.keras import WandbCallback
from sklearn.model_selection import learning_curve, train_test_split
import external as ext
from codetiming import Timer
# wandb.login()

"""> Side note: If this is your first time using W&B or you are not logged in, the link that appears after running `wandb.login()` will take you to sign-up/login page. Signing up is as easy as one click.

# 👩‍🍳 Prepare Dataset
"""

# Prepare the training dataset
BATCH_SIZE = 60
x_train, x_test, y_train, y_test = train_test_split(ext.dfBoxscores,ext.LabelResult,test_size=0.076,random_state=8 )
tmp=[[x,y] for x,y in zip(list(x_test['ID']),list(x_test['ID_O'])  ) ]
XFinalTest=ext.uniteBoxScores(tmp)

del x_train['ID']
del x_train['ID_O']
del x_train['HOME']
del x_train['HOME_O']



# x_train = np.reshape(x_train, (-1, 26))
# x_test = np.reshape(x_test, (-1, 26))

# build input pipeline using tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((XFinalTest, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)

array_train,array_test=ext.separation(x_train,y_train)



t=Timer()
t.start()
numberNeurons=[10,30,50,100,150,200]
for nN in numberNeurons:
    name="saved_model2/"+str(nN)
    model=keras.models.load_model(name)
    res={
        'num_neurons':nN,
    }
    tmp=model.evaluate(val_dataset, verbose=False)
    res['val_loss']=round(tmp[0],5)
    res['val_acc']=round(tmp[1],5)
    print(res)
t.stop()
winsound.Beep(440,1000)








