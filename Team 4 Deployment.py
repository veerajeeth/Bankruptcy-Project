


import numpy as np
import pandas as pd



df = pd.read_csv(r'E:\ExcelR\Project_1\bankruptcy-prevention.csv')



df




df.info()




df_new = df.iloc[:,:]
df_new



df_new["class_yn"] = 1
df_new





df_new.loc[df[' class'] == 'df', 'class_yn'] = 0





df_new




df_new.drop(' class', inplace = True, axis =1)
df_new.head()




# Input
x = df_new.iloc[:,:-1]

# Target variable

y = df_new.iloc[:,-1]




from sklearn.model_selection import train_test_split # trian and test
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.metrics import classification_report




X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)



from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix



#Random Forest



#Training the Random Foest classification on the Training set




from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_model.fit(X_train, y_train)
y_pred = RF_model.predict(X_test)
acc5 = accuracy_score(y_test, y_pred)




# Train Score
train_score = RF_model.score(X_train, y_train)
print('Training Score: %0.4f'% train_score)


# Testing the Random classification model on the testing set



# Test score
print('Recall score: %0.4f'% recall_score(y_test, y_pred))
print('Precision score: %0.4f'% precision_score(y_test, y_pred))
print('F1-Score: %0.4f'% f1_score(y_test, y_pred))
print('Accuracy score: %0.4f'% accuracy_score(y_test, y_pred))




#Create a pickle file using serialization









