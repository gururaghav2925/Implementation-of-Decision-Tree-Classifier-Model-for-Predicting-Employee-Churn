# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Required Libraries**  
   Import `pandas`, `matplotlib`, `sklearn`, and other required modules.
2. **Load and Preprocess the Data**  
   - Load the dataset using `pandas.read_csv()`  
   - Handle missing values if any  
   - Encode categorical columns (e.g., `salary`) using `LabelEncoder`
3. **Define Features and Target**  
   - Select relevant input features  
   - Define the target variable (`left` column)
4. **Split the Dataset**  
   - Use `train_test_split` to divide the dataset into training and testing sets
5. **Train and Evaluate the Model**  
   - Initialize and train a `DecisionTreeClassifier`  
   - Make predictions and evaluate using accuracy  
   - Visualize the decision tree using `plot_tree()`
## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Guru Raghav Ponjeevith V
RegisterNumber:  212223220027
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
data=pd.read_csv("Employee.csv")
import pandas as pd
print(data.head())
data.info()
data.isnull().sum()
print(data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
print(dt.fit(x_train,y_train))
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
print(dt.predict([[0.5,0.8,9,260,6,0,1,2]]))
print(data.head())
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:

**data.head()**

![image](https://github.com/user-attachments/assets/ce1ab6ea-d94b-42a4-9e3a-e5fe87fbeb44)


**data.info()**


![image](https://github.com/user-attachments/assets/4d075c90-30f2-4302-8997-2e1fabc0b50f)



![image](https://github.com/user-attachments/assets/f8fabb62-e86a-4539-8eae-ed1828b5a0e9)


**Accuracy**


![image](https://github.com/user-attachments/assets/5af8344d-6740-41e1-a15e-48c05020da4e)

**Predicited Output**

![image](https://github.com/user-attachments/assets/a7a2987a-35ac-4aee-b495-1c0ef822a7e8)


**Plt.show()**


![image](https://github.com/user-attachments/assets/e85ccb9f-d6eb-43c0-bc34-5c778749e9cb)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
