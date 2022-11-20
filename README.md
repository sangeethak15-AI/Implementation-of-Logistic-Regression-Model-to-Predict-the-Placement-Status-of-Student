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
![image](https://user-images.githubusercontent.com/93992063/202885460-47bfad33-2e86-4623-bb47-5b22a63bf7b8.png)
![image](https://user-images.githubusercontent.com/93992063/202885463-1811a5da-9c97-4f6b-9860-6197ddc4ade6.png)

![image](https://user-images.githubusercontent.com/93992063/202885468-ff529ba5-4ad4-47cb-a729-32fcf9ce6594.png)

![image](https://user-images.githubusercontent.com/93992063/202885600-b74e737b-c5ec-435e-ac96-c5c84951294e.png)

![image](https://user-images.githubusercontent.com/93992063/202885472-e150ec9c-49a4-427c-b3aa-6f38b26c57c3.png)
![image](https://user-images.githubusercontent.com/93992063/202885474-2d3bf185-a311-4f0f-af8e-94c1a8164eae.png)

![image](https://user-images.githubusercontent.com/93992063/202885483-fb44c0f4-ef90-4d87-8698-a01a7c94228a.png)

![image](https://user-images.githubusercontent.com/93992063/202885494-d0598b2d-6fd1-428a-b375-5a69e51e927e.png)

![image](https://user-images.githubusercontent.com/93992063/202885532-e05f8e27-2331-432f-9abb-125efcb8a588.png)

![image](https://user-images.githubusercontent.com/93992063/202885508-b0126681-1d69-44fb-96a4-964b3fec9444.png)

![image](https://user-images.githubusercontent.com/93992063/202885502-21b44337-516d-43ac-ad74-fe49da4579f2.png)

![image](https://user-images.githubusercontent.com/93992063/202885621-aab3ba12-9f84-4c04-810b-9c7694e61efb.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
