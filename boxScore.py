from logging import exception
from unittest import case
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nba_api.stats.static import teams

#Here we make the average for every statistical voice (either traditional or advance) for each team
# This will be used for testing part
def create_average_data(box_scores,all_team_TEAM_ID,index_second_team):
       
        avg_total=[]
        
        for TM in all_team_TEAM_ID:
          
            #For each team in allTeamTEAM_ID
            team=np.where(box_scores==TM)
            #Find the TEAM_IDs in the box_scores of home and away games  
            home_games=[x for x,y in zip(team[0],team[1]) if y==0]
            away_games=[x for x,y in zip(team[0],team[1]) if y==index_second_team]

            # Generaly we are both away and home games, but not always
            # Anyway the number of matches for every single club is nearly the same (42-43) 
            if away_games and home_games:
                res1=np.take(box_scores,away_games,axis=0)
                res2=np.take(box_scores,home_games,axis=0)
                avg_total.append(list(map(lambda x,y:np.round((x+y)/2,1),
                np.round(res1[:,index_second_team:].mean(0),1) ,np.round(res2[:,0:index_second_team].mean(0),1))))
            else:
                if away_games:
                    res1=np.take(box_scores,away_games,axis=0)
                    avg_total.append(np.round(res1[:,index_second_team:].mean(0),1) )
                    
                if home_games:
                    res2=np.take(box_scores,home_games,axis=0)
                    avg_total.append(np.round(res2[:,0:index_second_team].mean(0),1) )
            
        avg_total=np.array(avg_total)
        
        return avg_total

class boxScore:
    
    def __init__(self,years,mode):
        self.mode=mode
        self.name="BoxScore"+years+mode+".txt"
        #Just two modes: advance or traditional split

        try:
            assert (mode=="traditional" or mode=="advance" ), "Only two modes"

        except Exception as e:
            raise Exception(e)

        all_team_TEAM_ID=pd.DataFrame(teams.get_teams())['id']
        box_scores=[]
        label_result=[]
        # indexTEAM_IDSecondTeam 
        dict_mode={
            "traditional":16,
            "advance":19
        }
       
        index_second_team=dict_mode[mode]
            
        # Retrive all BoxScores from the selected season
        with open('BoxScoresFile/'+self.name,'r') as f:
                column_check=0
                fr=f.readlines()
                
                for line in fr:
            
                    line=line.replace('\n', '')
                    line=line.replace(',', '')
                    x=line.split()
                    # In the first row there column's name 
                    if column_check==0:
                        
                        x.pop(-1)
                        column_names=x
                        column_check+=1
                    else:
                        team=list(map(float, x))
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
            
            not_wanted_stats=['TEAM_ID','HOME',"PTS","FGA",'FG3A',"FTA"]

        if mode=="advance":
            not_wanted_stats=["TEAM_ID","HOME","NET_RATING",       
                                "USG_PCT" , "PACE_PER40"]
   
            
        #Put in not_wanted_stats set also the same stats for the opposite team
        tmp=[el+"_O" for el in not_wanted_stats ]
        not_wanted_stats.extend(tmp)
        separator=int(len(column_names)/2)
        
    
        #Create pandas box_scores 
        df_box_scores=pd.DataFrame(box_scores)
        
        #Create the average stats for all team, usefull for testing
        avg_team=create_average_data(box_scores,all_team_TEAM_ID,index_second_team)
        #Create pandas box_scores for average stats
        df_average_stats=pd.DataFrame(avg_team)
        #Give names for the column of both pandas
        df_box_scores.columns = column_names
        df_average_stats.columns = column_names[0:separator]

        # NORMALIZE THE DATA
        # for cN in column_names:
        #     if(cN!="TEAM_ID" and cN!="TEAM_ID_O"):
        #         df_box_scores[cN] = df_box_scores[cN] /df_box_scores[cN].abs().max()
        #         if(not cN.__contains__("_O")):
        #             df_average_stats[cN]=df_average_stats[cN]/df_average_stats[cN].abs().max()

        self.df_box_scores=df_box_scores
        self.average_data=df_average_stats
        self.label_result=label_result
        self.column_names=column_names
        self.not_wanted_stats=not_wanted_stats
    
   



    # Separate the corpus in 4 file: xtrain,ytrain xtest and ytest
    # x are the real data y just the label
    def separation(self):
        x_train, x_test, y_train, y_test = train_test_split(self.df_box_scores,self.label_result,test_size=0.076,random_state=24 )
        
        list_parameters=[[x,y,z] for x,y,z in zip(   list(x_test['TEAM_ID']),list(x_test['TEAM_ID_O']),list(x_test.index)  ) ]
    

        # Trasform the data-test:: the team's performance is substituted by the means of the performances w.r.t traing set for any stats voice (both advance or traditional)
        x_test=self.create_data_test(list_parameters)
        
        # Column not usefull and not wanted for training and testing
        for el in self.not_wanted_stats:
            del x_train[el]
            del x_test[el]
        
        return x_train, x_test, y_train, y_test

    # Here we used the pandas average_data create during the init function    
    def create_data_test(self,list_parameters):
        result=[]
        index=[]
        
        for triple in list_parameters:
            average_team1=sum(self.average_data.loc[self.average_data['TEAM_ID']==triple[0],:].values.tolist(),[])
            average_team2=sum(self.average_data.loc[self.average_data['TEAM_ID']==triple[1],:].values.tolist(),[])  
            line=sum([average_team1,average_team2],[])
            index.append(triple[2])
            result.append(line)
        
        result=pd.DataFrame(result)

        result.columns=self.column_names   
        result.index=index
            
        

            
        
        return result


    