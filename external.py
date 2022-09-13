
import pandas as pd
import numpy as np

def uniteBoxScores(para,box_score):
    result=[]
    if(box_score.mode=='traditional'):
        for couple in para:
            tmp1=sum(box_score.dfAvg.loc[box_score.dfAvg['ID']==couple[0],'FG_PCT':'PTS'].values.tolist(),[])
            tmp2=sum(box_score.dfAvg.loc[box_score.dfAvg['ID']==couple[1],'FG_PCT':'PTS'].values.tolist(),[])  
            line=sum([tmp1,tmp2],[])
        
            result.append(line)

        result=pd.DataFrame(result)
        result.columns=[ 'FG_PCT', 'FG3_PCT', 'FT_PCT','OREB','DREB','AST','STL','BLK','TO','PF','PTS',
        'FG_PCT_O', 'FG3_PCT_O', 'FT_PCT_O','OREB_O','DREB_O','AST_O','STL_O','BLK_O','TO_O','PF_O','PTS_O']
        
    else:
        for couple in para:
            tmp1=sum(box_score.dfAvg.loc[box_score.dfAvg['ID']==couple[0],'OFF_RATING':'PIE'].values.tolist(),[])
            tmp2=sum(box_score.dfAvg.loc[box_score.dfAvg['ID']==couple[1],'OFF_RATING':'PIE'].values.tolist(),[])  
            line=sum([tmp1,tmp2],[])
        
            result.append(line)

        result=pd.DataFrame(result)
        result.columns=[ "OFF_RATING","DEF_RATING","NET_RATING","AST_PCT","AST_TOV",   
                            "AST_RATIO", "OREB_PCT","DREB_PCT","REB_PCT","TM_TOV_PCT","EFG_PCT","TS_PCT",       
                                "USG_PCT" ,"PACE", "PACE_PER40", "POSS","PIE",

                            "OFF_RATING_O","DEF_RATING_O","NET_RATING_O","AST_PCT_O","AST_TOV_O",   
                            "AST_RATIO_O", "OREB_PCT_O","DREB_PCT_O","REB_PCT_O","TM_TOV_PCT_O","EFG_PCT_O","TS_PCT_O",          
                                "USG_PCT_O" ,"PACE_O", "PACE_PER40_O", "POSS_O","PIE_O"]
    return result







        
#Used for cross validation:
#Dataset is divided in 10 fold
#For each i-th elements of resulTrain and resultTest there is a different mixed of this 10 parts
# example:
# resulTrain[0]: have from 0-8-th fold as train and resultTest[0] have just the 9-th as test
# 
def separation(x_train,y_train):
    resultTrain=[]
    resultTest=[]
    sliceLen=len(x_train)/10
    x_train=x_train.values.tolist()
    for i in range(0,10):  
            indexArray=range(int(i*sliceLen),int((i+1)*sliceLen))
            testX=x_train[int(i*sliceLen):int((i+1)*sliceLen)]
            testY=y_train[int(i*sliceLen):int((i+1)*sliceLen)]
            trainX=x_train.copy()
            trainY=y_train.copy()
            for ele in sorted(indexArray, reverse = True):
                del trainX[ele]
                del trainY[ele]
            resultTrain.append([trainX,trainY])
            resultTest.append([testX,testY])


    return resultTrain,resultTest