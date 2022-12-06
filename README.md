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
1.N.Niharika
2.K.Sucharitha
3.A.Sai Bandhavi
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

![1](https://user-images.githubusercontent.com/94165377/205953026-212c0830-8e00-47ff-b75d-e2a7057bcb25.png)
![2](https://user-images.githubusercontent.com/94165377/205953067-520de42e-e00b-4e65-a973-53dd0225c5a5.png)

![3](https://user-images.githubusercontent.com/94165377/205953139-bc3c104f-e2de-4e25-b044-ae9108650add.png)

![4](https://user-images.githubusercontent.com/94165377/205953177-d5eeccc6-e1f7-4ddc-a829-b221deeca09c.png)

![5](https://user-images.githubusercontent.com/94165377/205953198-802ef6c3-35d8-4538-8865-4686e76b0ca8.png)
![6](https://user-images.githubusercontent.com/94165377/205953233-b4f14c77-9b9f-4ffa-b471-effc88f3b4f6.png)


![7](https://user-images.githubusercontent.com/94165377/205953273-abceca97-c7d6-4b37-8b48-e6b4037b9481.png)
![8](https://user-images.githubusercontent.com/94165377/205953304-78214fa2-319e-412d-94db-e4ab0fbf331d.png)
![9](https://user-images.githubusercontent.com/94165377/205953316-55c601d9-4b96-4b67-9a44-909ffcb7cea5.png)

![10](https://user-images.githubusercontent.com/94165377/205953342-38bbd479-56b6-4940-8be0-cc0a96c878b2.png)





## Advantage :
Titanic Survival Prediction data set, the main task is to predict whether the passenger will survive or not.By this we can find out the number of members will alive .
## Result:
Thus Implementation of titanic survival Prediction was executed successfully.
