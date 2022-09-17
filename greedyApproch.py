




import tensorflow as tf
from tensorflow import keras
import externalTensor as exT
import winsound
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


from boxScore import boxScore
from codetiming import Timer
import lime
from lime import lime_tabular




# Prepare the training dataset
t = Timer()
t.start()

years="2019-20"
# stats="traditional"
stats='advance'
box_score=boxScore(years,stats)
x_train, x_test, y_train, y_test = train_test_split(box_score.dfBoxscores,box_score.LabelResult,test_size=0.076,random_state=8 )
tmp=[[x,y] for x,y in zip(list(x_test['ID']),list(x_test['ID_O'])  ) ]

box_score=boxScore(years,stats)
x_train, x_test, y_train, y_test=box_score.separation()





# activation=['relu','sigmoid']
activation=['relu']
# number_neurons=[10,30,50,100,150,200]
number_neurons=[10]
# possible_learning_rate=[0.0001,0.001,0.01]
possible_learning_rate=[0.001]



kf = KFold(n_splits=10, random_state=1, shuffle=True)
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

                res=exT.trainBasic(res,x_train,y_train,
                    x_test, y_test, 
                    model,
                    kf,
                    500
                   
                )
                
                #Saved the best model based on the accuracy
                if(res['val_acc']>bestRes['val_acc']):
                    bestModel=model
                    bestRes=res
                        
                    
                
    print("Best model",bestRes)

    # f = open("saved_model_"+stats+"/summary.txt", "a")
    # tmpName=str(bestRes['num_neurons'])+"_"+str(bestRes['acti_fun'])+"_"+str(bestRes['learning_rate'])+"_LOSS_"+str(bestRes['loss'])+"_ACC_"+str(bestRes['acc'])+"_LOSSVAL_"+str(bestRes['val_loss'])+"_ACCVAL_"+str(bestRes['val_acc'])  
    # f.write(tmpName+"\n")
    # f.close()
    # bestModel.save("saved_model_"+stats+"/"+str(bestRes['num_neurons']))

explainer = lime_tabular.LimeTabularExplainer(
                        training_data=np.array(x_train),
                        feature_names=x_train.columns,
                        class_names=['win', 'lose'],
                        mode='classification'
                    )
exp = explainer.explain_instance(
    data_row=x_test.iloc[1], 
    predict_fn=model.predict
)
exp.show_in_notebook(show_table=True)
t.stop()
winsound.Beep(440,2000)










