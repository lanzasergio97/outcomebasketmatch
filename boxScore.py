from logging import exception
import pandas as pd
import numpy as np
import sys
from nba_api.stats.static import teams

def averegeStats(avgTeamHome,avgTeamAway,allTeamId):
        avgTeam=[]
        for x1 in avgTeamHome:
            indexRes=np.where(avgTeamAway==x1[0])[0]
            if(indexRes.size>0):
                x2=np.take(avgTeamAway,indexRes,axis=0)[0]
                
            
                tmp=[   round( (t1+t2)/2,2) for t1,t2 in zip( x1,x2) ]
                
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
        with open('include\BoxScoresFile\\'+self.name,'r') as f:
                fr=f.readlines()  
                for line in fr:
            
                    line=line.replace('\n', '')
                    line=line.replace(',', '')
                    x=line.split() 
                    team=list(map(float, x))
                    team[0]=team[0]
                    team[indexSecondTeam]=team[indexSecondTeam]
                    # Regularize label to 0 and 1
                    singleLabel=int(team[-1]) -1
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
                
                avgTeamAway.append(np.round(res[:,indexSecondTeam:].mean(0),2) ) 
            if(gameHome):
                res=np.take(BoxScores,gameHome,axis=0)
                avgTeamHome.append(np.round(res[:,0:indexSecondTeam].mean(0),2) )



        avgTeamHome=np.array(avgTeamHome)
        avgTeamAway=np.array(avgTeamAway)

        #We make an array with the averege of all team's stats (no matter traditional or advance)
        avgTeam=averegeStats(avgTeamHome,avgTeamAway,allTeamId)

        


        #Now we set the dataframe given the correct names for each stats.
        # We differenziate by the mode 
        
        if(mode=='traditional'):
            columnNames=['ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT','OREB','DREB','AST','STL','BLK','TO','PF','PTS','HOME',
            'ID_O', 'FG_PCT_O', 'FG3_PCT_O', 'FT_PCT_O','OREB_O','DREB_O','AST_O','STL_O','BLK_O','TO_O','PF_O','PTS_O','HOME_O']
        else:
            columnNames=["ID","OFF_RATING","DEF_RATING","NET_RATING","AST_PCT","AST_TOV",   
                            "AST_RATIO", "OREB_PCT","DREB_PCT","REB_PCT","TM_TOV_PCT","EFG_PCT","TS_PCT",       
                                "USG_PCT" ,"PACE", "PACE_PER40", "POSS","PIE",  "HOME",

                            "ID_O","OFF_RATING_O","DEF_RATING_O","NET_RATING_O","AST_PCT_O","AST_TOV_O",   
                            "AST_RATIO_O", "OREB_PCT_O","DREB_PCT_O","REB_PCT_O","TM_TOV_PCT_O","EFG_PCT_O","TS_PCT_O",          
                                "USG_PCT_O" ,"PACE_O", "PACE_PER40_O", "POSS_O","PIE_O", "HOME_O"]

        separator=int(len(columnNames)/2)
        
        dfAvg=pd.DataFrame(avgTeam)
        dfAvg.columns = columnNames[0:separator]

        dfBoxscores=pd.DataFrame(BoxScores)
    
        dfBoxscores.columns = columnNames
        
        
        #NORMALIZE THE DATA
        for cN in columnNames:
            if(cN!="ID" and cN!="ID_O"):
                dfBoxscores[cN] = dfBoxscores[cN] /dfBoxscores[cN].abs().max()
        self.dfBoxscores=dfBoxscores
        self.dfAvg=dfAvg
        self.LabelResult=labelResult
        






