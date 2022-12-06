# Mini-Project--Application-of-NN
## Project Title:
Titanic Survival predictions
## Project Description :
The Model Predicts Whether A Passenger Would Survive On The Titanic Taking Into Account And Comparing And Finding Relations Amongst Various Features.
## Algorithm:
1.Import necessary libraries.

2.upload the suitable Training And Testing Datasets.

3.To Start Analyzing The Given Test And Train Datasets To Find Out Patterns Between The Features And Finding Relations Of Essential Features With The Target Feature.

4.Start Comparing Individual Features With The Target Feature And Find Out The Effect Of Individual Features On The Target Label I.E. How Individual Features Determine Whether The Person Survived Or Not.

5.Study the final output.
## Program:
```
Developed By Team Members:
1.A.Sai Bandhavi
2.K.Sucharitha
3.N.Niharika
```
```
import numpy as np 
import pandas as pd
import os
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data.head()
test_data.head()
train_data.shape
test_data.shape
train_data.info()
train_data.isnull().sum()
train_data.drop("Cabin" , inplace= True ,axis = 1)
train_data.info()
med = train_data['Age'].median()
train_data['Age'].fillna(med , inplace = True)
train_data['Embarked'].dropna(inplace = True)
train_data.isnull().sum()
test_data.isnull().sum()
med = test_data['Fare'].median()
test_data['Fare'].fillna(med , inplace = True)
train_data.head(10)
import matplotlib.pyplot as plt
import seaborn as sns
train_data.describe()
train_data['Pclass'].value_counts()
plt.figure()
sns.countplot(train_data['Pclass'])
plt.show()
train_data['Sex'].value_counts()
plt.figure()
sns.countplot(train_data['Sex'])
plt.show()
train_data.head()
train_data.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
test_data['Sex'] = le.fit_transform(test_data['Sex'])
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])
test_data['Embarked'] = le.fit_transform(test_data['Embarked'])
train_data.head()
show()
```
## Output:
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/1.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/2.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/3.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/4.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/5.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/6.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/7.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/8.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/9.png?raw=true)
![output](https://github.com/Saibandhavi75/Mini-Project--Application-of-NN/blob/main/10.png?raw=true)
## Advantage :
Titanic Survival Prediction data set, the main task is to predict whether the passenger will survive or not.By this we can find out the number of members will alive .
## Result:
Thus Implementation of titanic survival Prediction was executed successfully.
