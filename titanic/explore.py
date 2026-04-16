import pandas as pd

df= pd.read_csv('titanic.csv')

#Check Missing Values
print(df.isnull().sum())