# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: sangeetha.K
RegisterNumber: 212221230085 
*/
```

```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![200621130-7cbde7df-d1d2-449c-aafa-d5c404639720](https://user-images.githubusercontent.com/93992063/202885460-47bfad33-2e86-4623-bb47-5b22a63bf7b8.png)
![200621153-a98cc3b5-dd84-4317-bd9f-2d37a1a2ce55](https://user-images.githubusercontent.com/93992063/202885463-1811a5da-9c97-4f6b-9860-6197ddc4ade6.png)

![200621216-0705583c-3a84-428d-80ea-271772851342](https://user-images.githubusercontent.com/93992063/202885468-ff529ba5-4ad4-47cb-a729-32fcf9ce6594.png)

![200621264-dd3790d1-7b1f-4313-a5fb-a45d21e2b60a](https://user-images.githubusercontent.com/93992063/202885472-e150ec9c-49a4-427c-b3aa-6f38b26c57c3.png)
![200621314-54c9d8ea-c343-4841-aab8-80f93a9f4033](https://user-images.githubusercontent.com/93992063/202885474-2d3bf185-a311-4f0f-af8e-94c1a8164eae.png)

![200621361-90a5c368-a427-4c22-87a7-07d3e5731724](https://user-images.githubusercontent.com/93992063/202885483-fb44c0f4-ef90-4d87-8698-a01a7c94228a.png)
![200621441-3ba11f56-6c9e-43a6-9aff-91d02230ce5b](https://user-images.githubusercontent.com/93992063/202885494-d0598b2d-6fd1-428a-b375-5a69e51e927e.png)
![200621467-0b672fc5-3a3a-46a7-9da6-deb729dc56c6](https://user-images.githubusercontent.com/93992063/202885499-c800ec14-fe5e-48c7-8dc2-7053cd087697.png)
![200621467-0b672fc5-3a3a-46a7-9da6-deb729dc56c6](https://user-images.githubusercontent.com/93992063/202885532-e05f8e27-2331-432f-9abb-125efcb8a588.png)

![200621501-e7c247ef-b790-4b58-a32a-4810d716111b](https://user-images.githubusercontent.com/93992063/202885508-b0126681-1d69-44fb-96a4-964b3fec9444.png)

![200621542-1a828fb3-693e-45c2-81ca-b3692d303c74](https://user-images.githubusercontent.com/93992063/202885502-21b44337-516d-43ac-ad74-fe49da4579f2.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
