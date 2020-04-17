
#Importing Libraries
import pandas as pd
import pickle
# Passing CSV into matches
matches=pd.read_csv('D:\matches.csv')
matches.head()
#Dropping Some Unuseful Columns
matches=matches.drop(['id','date','umpire1','umpire2','umpire3','player_of_match',],axis=1)
#Rows VS Columns
matches.shape
matches.info()
#Replacing Null Values With Draw
matches['winner'].fillna('Draw', inplace=True)
#Replacing Team Names with Shorter form and encoding with numbers
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors','Delhi Capitals']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW','DCC'],inplace=True)
encode={'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'DCC':14,'Draw':15}}
matches.replace(encode, inplace=True)
matches.head()
matches[pd.isnull(matches['city'])]
#Replacing Null Values with City Name (Manually)
matches['city'].fillna('Dubai',inplace=True)
matches.describe()
# Creating Dummy Variables
dummy1=pd.get_dummies(matches['city'])
dummy2=pd.get_dummies(matches['toss_decision'])
dummy3=pd.get_dummies(matches['result'])
dummy4=pd.get_dummies(matches['Pitch Type'])
dummy7=pd.get_dummies(matches['toss_winner'])
dummy8=pd.get_dummies(matches['season'])
# Joining all dummy variables to dataframe
matches=pd.concat([matches,dummy1,dummy2,dummy3,dummy4,dummy7,dummy8],axis=1)
matches.head()
#Dropping actual values of dummy variables
matches=matches.drop(['city','toss_decision','toss_winner','Pitch Type','result','season'],axis=1)
matches.loc[1:10]
#Dropping Venue
matches=matches.drop(['venue'],axis=1)
#Creating Dummy variables of team1 & team2
dummy9=pd.get_dummies(matches['team1'])
dummy10=pd.get_dummies(matches['team2'])
matches=pd.concat([matches,dummy9,dummy10],axis=1)
matches=matches.drop(['team1','team2'],axis=1)
matches.head()
#Passing Data
import numpy as np
#Passing all variables except Winner (dependent variable)
x = matches.iloc[:,:-5].values 
y = matches.iloc[:,5].values
from sklearn.model_selection import train_test_split
#Test Size of 20%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
import pickle
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)
accuracy

pickle.dump(model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
