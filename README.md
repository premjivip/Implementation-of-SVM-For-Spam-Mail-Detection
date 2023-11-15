# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

### Step 1 :

Import the necessary python packages using import statements.

### Step 2 :

Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

### Step 3 :

Split the dataset using train_test_split.

### Step 4 :

Calculate Y_Pred and accuracy.

### Step 5 :

Print all the outputs.

### Step 6 :

End the Program.

## Program:

```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: premji p
RegisterNumber: 212221043004
*/


import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### DATA.HEAD() :

![image](https://github.com/Yogabharathi3/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118899387/8bed8939-43b6-4339-9c17-4964476c0e0b)

### DATA.INFO() :

![image](https://github.com/Yogabharathi3/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118899387/9a34b7de-d200-4005-a293-25f23c28081b)


### DATA.ISNULL().SUM() :
![image](https://github.com/Yogabharathi3/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118899387/20767971-8741-44c2-ae17-a5bdfbbe265c)

### Y_PRED :
![image](https://github.com/Yogabharathi3/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118899387/d7269074-efb3-4642-b257-4b0e99d2161b)

### ACCURACY :
![image](https://github.com/Yogabharathi3/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118899387/9d02aba9-278c-41f4-9584-defe906941c2)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
