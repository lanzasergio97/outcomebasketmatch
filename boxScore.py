from logging import exception
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nba_api.stats.static import teams

def averegeStats(avg_team_home,avg_team_away,all_team_id):
        avgTeam=[]
        for x1 in avg_team_home:
            indexRes=np.where(avg_team_away==x1[0])[0]
            if(indexRes.size>0):
                x2=np.take(avg_team_away,indexRes,axis=0)[0]
                
            
                tmp=[   round( (t1+t2)/2,1) for t1,t2 in zip( x1,x2) ]
                
                avgTeam.append(np.array(tmp))
                
            else:
                avgTeam.append(x1)
            

        indexRes=np.where(avg_team_away== list(set(all_team_id)-set(avg_team_home[:,0])))[0]
        for el in indexRes:
            res=np.take(avg_team_away,el,axis=0)
            avgTeam.append(res)
        
        return avgTeam
def create_average_data(box_scores,all_team_id,index_second_team):
        avg_team_home=[]
        avg_team_away=[]
        for TM in all_team_id:
            #For each team in allTeamId
            team=np.where(box_scores==TM)
            
            home_games=[x for x,y in zip(team[0],team[1]) if y==0]
            away_games=[x for x,y in zip(team[0],team[1]) if y==index_second_team]
            if away_games:
                res=np.take(box_scores,away_games,axis=0)
                
                avg_team_away.append(np.round(res[:,index_second_team:].mean(0),1) ) 
            if home_games:
                res=np.take(box_scores,home_games,axis=0)
                avg_team_home.append(np.round(res[:,0:index_second_team].mean(0),1) )



        avg_team_home=np.array(avg_team_home)
        avg_team_away=np.array(avg_team_away)

        #We make an array with the averege of all team's stats (no matter traditional or advance)
        avg_team=averegeStats(avg_team_home,avg_team_away,all_team_id)
        return avg_team

class boxScore:
    
    def __init__(self,years,mode):
        self.mode=mode
        self.name="BoxScore"+years+mode+".txt"
        #Just two modes: advance or traditional split
       
        try:
            
            assert (mode=="traditional" or mode=="advance" ), "Only two modes"

        except Exception as e:
            raise Exception(e)

        all_team_id=pd.DataFrame(teams.get_teams())['id']
        box_scores=[]
        label_result=[]
        # indexIdSecondTeam 
        dict_mode={
            "traditional":13,
            "advance":19
        }
        index_second_team=dict_mode[mode]
            
        # Retrive all BoxScores from the selected season
        with open('BoxScoresFile/'+self.name,'r') as f:
            
                fr=f.readlines()  
                for line in fr:
            
                    line=line.replace('\n', '')
                    line=line.replace(',', '')
                    x=line.split() 
                    team=list(map(float, x))
                    team[0]=team[0]
                    team[index_second_team]=team[index_second_team]
                    # Regularize label to 0 (win first team ) and 1 (win second team)
                    singleLabel=int(team[-1]) -1
                    # Second Regularizion:  [1,0] (win first team ) and [0,1] (win second team)
                    if(singleLabel==0):
                        tmp=[1,0]
                    else:
                        tmp=[0,1]
                    label_result.append(tmp)
                    #From the boxscores we don't include the label
                    team.pop(-1)
                    box_scores.append(team)
        

        box_scores=np.array(box_scores)
    
        #Now we fit the dataframe with the correct names for each stats.
        # We differenziate by mode 
        
        if mode=='traditional':
            column_names=['ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT','OREB','DREB','AST','STL','BLK','TO','PF','PTS','HOME',
            'ID_O', 'FG_PCT_O', 'FG3_PCT_O', 'FT_PCT_O','OREB_O','DREB_O','AST_O','STL_O','BLK_O','TO_O','PF_O','PTS_O','HOME_O']
            not_wanted_stats=['ID','HOME',"PTS"]
            
        if mode=="advance":
            column_names=["ID","OFF_RATING","DEF_RATING","NET_RATING","AST_PCT","AST_TOV",   
                            "AST_RATIO", "OREB_PCT","DREB_PCT","REB_PCT","TM_TOV_PCT","EFG_PCT","TS_PCT",       
                                "USG_PCT" ,"PACE", "PACE_PER40", "POSS","PIE",  "HOME",

                            "ID_O","OFF_RATING_O","DEF_RATING_O","NET_RATING_O","AST_PCT_O","AST_TOV_O",   
                            "AST_RATIO_O", "OREB_PCT_O","DREB_PCT_O","REB_PCT_O","TM_TOV_PCT_O","EFG_PCT_O","TS_PCT_O",          
                                "USG_PCT_O" ,"PACE_O", "PACE_PER40_O", "POSS_O","PIE_O", "HOME_O"]
            
            not_wanted_stats=["ID","HOME","NET_RATING",       
                                "USG_PCT" , "PACE_PER40"]
        if mode=="fourfactors":
            column_names=["ID","EFG_PCT","FTA_RATE", "TM_TOV_PCT","OREB_PCT", "HOME",
            			
                        "ID_O","EFG_PCT_O","FTA_RATE_O", "TM_TOV_PCT_O","OREB_PCT_O",  "HOME_O"]


            not_wanted_stats=["ID","HOME"]

        tmp=[el+"_O" for el in not_wanted_stats ]
        not_wanted_stats.extend(tmp)
        separator=int(len(column_names)/2)
        
        

        df_box_scores=pd.DataFrame(box_scores)

        df_box_scores.columns = column_names

        avg_team=create_average_data(box_scores,all_team_id,index_second_team)
        df_average_stats=pd.DataFrame(avg_team)
        df_average_stats.columns = column_names[0:separator]
        # NORMALIZE THE DATA
        for cN in column_names:
            if(cN!="ID" and cN!="ID_O"):
                df_box_scores[cN] = df_box_scores[cN] /df_box_scores[cN].abs().max()
                if(not cN.__contains__("_O")):
                    df_average_stats[cN]=df_average_stats[cN]/df_average_stats[cN].abs().max()

        self.df_box_scores=df_box_scores
        self.average_data=df_average_stats
        self.label_result=label_result
        self.column_names=column_names
        self.not_wanted_stats=not_wanted_stats
    
   



    # Separate the corpus in 4 file: xtrain,ytrain xtest and ytest
    # x are the real data y just the label
    def separation(self):
        x_train, x_test, y_train, y_test = train_test_split(self.df_box_scores,self.label_result,test_size=0.076,random_state=24 )
        
        tmp=[[x,y,z] for x,y,z in zip(   list(x_test['ID']),list(x_test['ID_O']),list(x_test.index)  ) ]
    

        # Trasform the data-test:: the team's performance is substituted by his mean performance of any stats voice (both advance or traditional)
        
        
        x_test=self.create_data_test(tmp)
        
        # Column not usefull for training and testing
        for el in self.not_wanted_stats:
            del x_train[el]
            del x_test[el]
        
        return x_train, x_test, y_train, y_test
        
    def create_data_test(self,list_parameters):
        result=[]
        index=[]
        for triple in list_parameters:
            tmp1=sum(self.average_data.loc[self.average_data['ID']==triple[0],:].values.tolist(),[])
            tmp2=sum(self.average_data.loc[self.average_data['ID']==triple[1],:].values.tolist(),[])  
            line=sum([tmp1,tmp2],[])
            index.append(triple[2])
            result.append(line)
        
        result=pd.DataFrame(result)

        result.columns=self.column_names   
        result.index=index
            
        

            
        
        return result


    