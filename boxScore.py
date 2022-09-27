from logging import exception
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nba_api.stats.static import teams

def averegeStats(avgTeamHome,avgTeamAway,allTeamId):
        avgTeam=[]
        for x1 in avgTeamHome:
            indexRes=np.where(avgTeamAway==x1[0])[0]
            if(indexRes.size>0):
                x2=np.take(avgTeamAway,indexRes,axis=0)[0]
                
            
                tmp=[   round( (t1+t2)/2,1) for t1,t2 in zip( x1,x2) ]
                
                avgTeam.append(np.array(tmp))
                
            else:
                avgTeam.append(x1)
            

        indexRes=np.where(avgTeamAway== list(set(allTeamId)-set(avgTeamHome[:,0])))[0]
        for el in indexRes:
            res=np.take(avgTeamAway,el,axis=0)
            avgTeam.append(res)
        
        return avgTeam

class boxScore:
    
    def __init__(self,years,mode):
        self.mode=mode
        self.name="BoxScore"+years+mode+".txt"
        #Just two modes: advance or traditional split
        try:
            
            assert (mode=="traditional" or mode=="advance"), "Only two modes"

        except Exception as e:
            raise Exception(e)

        allTeamId=pd.DataFrame(teams.get_teams())['id']
        BoxScores=[]
        labelResult=[]
        # indexIdSecondTeam 
        if(mode=='traditional'):
            indexSecondTeam=13
        else:
            indexSecondTeam=19
        # Retrive all BoxScores from the selected season
        with open('BoxScoresFile\\'+self.name,'r') as f:
                fr=f.readlines()  
                for line in fr:
            
                    line=line.replace('\n', '')
                    line=line.replace(',', '')
                    x=line.split() 
                    team=list(map(float, x))
                    team[0]=team[0]
                    team[indexSecondTeam]=team[indexSecondTeam]
                    # Regularize label to 0 (win first team ) and 1 (win second team)
                    singleLabel=int(team[-1]) -1
                    # Second Regularizion:  [1,0] (win first team ) and [0,1] (win second team)
                    if(singleLabel==0):
                        tmp=[1,0]
                    else:
                        tmp=[0,1]
                    labelResult.append(tmp)
                    #From the boxscores we don't include the label
                    team.pop(-1)
                    BoxScores.append(team)
        

        BoxScores=np.array(BoxScores)
        avgTeamHome=[]
        avgTeamAway=[]
        for TM in allTeamId:
            #For each team in allTeamId
            team=np.where(BoxScores==TM)
            
            gameHome=[x for x,y in zip(team[0],team[1]) if y==0]
            gameAway=[x for x,y in zip(team[0],team[1]) if y==indexSecondTeam]
            if(gameAway):
                res=np.take(BoxScores,gameAway,axis=0)
                
                avgTeamAway.append(np.round(res[:,indexSecondTeam:].mean(0),1) ) 
            if(gameHome):
                res=np.take(BoxScores,gameHome,axis=0)
                avgTeamHome.append(np.round(res[:,0:indexSecondTeam].mean(0),1) )



        avgTeamHome=np.array(avgTeamHome)
        avgTeamAway=np.array(avgTeamAway)

        #We make an array with the averege of all team's stats (no matter traditional or advance)
        avgTeam=averegeStats(avgTeamHome,avgTeamAway,allTeamId)

        


        #Now we set the dataframe given the correct names for each stats.
        # We differenziate by the mode 
        
        if(mode=='traditional'):
            columnNames=['ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT','OREB','DREB','AST','STL','BLK','TO','PF','PTS','HOME',
            'ID_O', 'FG_PCT_O', 'FG3_PCT_O', 'FT_PCT_O','OREB_O','DREB_O','AST_O','STL_O','BLK_O','TO_O','PF_O','PTS_O','HOME_O']
            notWantedStats=['ID','HOME',"PTS"]
            
        else:
            columnNames=["ID","OFF_RATING","DEF_RATING","NET_RATING","AST_PCT","AST_TOV",   
                            "AST_RATIO", "OREB_PCT","DREB_PCT","REB_PCT","TM_TOV_PCT","EFG_PCT","TS_PCT",       
                                "USG_PCT" ,"PACE", "PACE_PER40", "POSS","PIE",  "HOME",

                            "ID_O","OFF_RATING_O","DEF_RATING_O","NET_RATING_O","AST_PCT_O","AST_TOV_O",   
                            "AST_RATIO_O", "OREB_PCT_O","DREB_PCT_O","REB_PCT_O","TM_TOV_PCT_O","EFG_PCT_O","TS_PCT_O",          
                                "USG_PCT_O" ,"PACE_O", "PACE_PER40_O", "POSS_O","PIE_O", "HOME_O"]
            
            notWantedStats=["ID","HOME","NET_RATING","PACE_PER40","USG_PCT"]

        tmp=[el+"_O" for el in notWantedStats ]
        notWantedStats.extend(tmp)
        separator=int(len(columnNames)/2)
        
        dfAvg=pd.DataFrame(avgTeam)
        dfAvg.columns = columnNames[0:separator]

        dfBoxscores=pd.DataFrame(BoxScores)
    
        dfBoxscores.columns = columnNames
        
        #NORMALIZE THE DATA
        # for cN in columnNames:
        #     if(cN!="ID" and cN!="ID_O"):
        #         dfBoxscores[cN] = dfBoxscores[cN] /dfBoxscores[cN].abs().max()
        self.dfBoxscores=dfBoxscores
        self.dfAvg=dfAvg
        self.LabelResult=labelResult
        self.columnNames=columnNames
        self.notWantedStats=notWantedStats
    
   



    # Separate the corpus in 4 file: xtrain,ytrain xtest and ytest
    # x are the real data y just the label
    def separation(self):
        x_train, x_test, y_train, y_test = train_test_split(self.dfBoxscores,self.LabelResult,test_size=0.076,random_state=24 )
        tmp=[[x,y] for x,y in zip(list(x_test['ID']),list(x_test['ID_O'])  ) ]

        # Trasform the data-test:: the team's performance is substituted by his mean performance of any stats voice (both advance or traditional)
        x_test=self.create_data_test(tmp)
        # Column not usefull for training and testing
        for el in self.notWantedStats:
            del x_train[el]
            del x_test[el]
        
        return x_train, x_test, y_train, y_test
        
    def create_data_test(box_score,para):
        result=[]
     
        for couple in para:
            tmp1=sum(box_score.dfAvg.loc[box_score.dfAvg['ID']==couple[0],:].values.tolist(),[])
            tmp2=sum(box_score.dfAvg.loc[box_score.dfAvg['ID']==couple[1],:].values.tolist(),[])  
            line=sum([tmp1,tmp2],[])
        
            result.append(line)

        result=pd.DataFrame(result)
        result.columns=box_score.columnNames   
        
            
        

            
        
        return result


