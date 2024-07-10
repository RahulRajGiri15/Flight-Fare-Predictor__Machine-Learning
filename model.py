import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

##Source - https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh
train=pd.read_excel('Data_Train.xlsx',engine='openpyxl')
sample = pd.read_excel('Sample_submission.xlsx',engine='openpyxl')
test = pd.read_excel('Test_set.xlsx',engine='openpyxl')
test = pd.concat([test,sample],axis=1)
df= pd.concat([train,test])

##Droping columns that does not seem practical to ask to a customer.
df.drop(labels=['Route','Arrival_Time','Duration','Additional_Info'],axis=1,inplace=True)
df.dropna(inplace=True)
df['Day']= df['Date_of_Journey'].str.split('/').str[0]
df['Month']= df['Date_of_Journey'].str.split('/').str[1]
df['Year']= df['Date_of_Journey'].str.split('/').str[2]

df['Total_Stops']=df['Total_Stops'].str.replace('non-','0 ')
df['Stops'] = df['Total_Stops'].str.split().str[0]
df['Departure_Hour'] = df['Dep_Time'].str.split(':').str[0]
df['Departure_Minute'] = df['Dep_Time'].str.split(':').str[1]

#Converting the datatype o newly created features
df['Day'] = df['Day'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)
df['Stops'] = df['Stops'].astype(int)
df['Departure_Hour'] = df['Departure_Hour'].astype(int)
df['Departure_Minute'] = df['Departure_Minute'].astype(int)

#Now droping the parent features since we don't need them
df.drop(['Date_of_Journey','Dep_Time','Total_Stops'],axis=1,inplace=True)
#Label encoding executed manually
source_dict = {y:x for x,y in enumerate(df.Source.value_counts().index.sort_values())}
destination_dict = {'Banglore':0,'Cochin':1,'Delhi':2,'Kolkata': 3,'Hyderabad':4,'New Delhi':5}

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Airline_Encoded']= le.fit_transform(df['Airline'].values)

df3 = df[['Airline']].copy()
df3['Encoded']=df['Airline_Encoded']
df3=df3.drop_duplicates('Airline').reset_index().iloc[:,1:]
d5=df3.Airline.values
d6=df3.Encoded.values
airline_dict = dict(zip(d5,d6))

df['Source_Encoded']=df['Source'].map(source_dict)
df['Destination_Encoded']=df['Destination'].map(destination_dict)
df = df.drop(['Airline','Source','Destination'],axis=1)
#Feature Selection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

df_train = df[0:10600]
df_test = df[10600:]
X = df_train.drop(['Price'],axis=1)
y = df_train.Price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train)
features_selected = X_train.columns[model.get_support()]
##All features selected except Year
X_train = X_train.drop(['Year'],axis=1)
X_test = X_test.drop(['Year'],axis=1)

#Feature Normalization
import scipy.stats as stat
for x in list(X_train.columns):
    X_train[x] = stat.yeojohnson(X_train[x])[0]

for y in list(X_test.columns):
    X_test[y] = stat.yeojohnson(X_test[y])[0]

##Random forest regressor model
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(X_train,y_train)

pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
