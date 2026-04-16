import pandas as pd

#Load directly from URL
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

#Save Locally
df.to_csv('titanic/titanic.csv', index=False)

print(df.shape)
print(df.head())