import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler as SS


dataset = "Processed_Cleveland.csv"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']
dataset = read_csv(dataset, names=columns)
dataset.isnull().sum()
dataset['target'] = dataset.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
dataset['sex'] = dataset.sex.map({0: 'female', 1: 'male'})
dataset['thal'] = dataset.thal.fillna(dataset.thal.mean())
dataset['ca'] = dataset.ca.fillna(dataset.ca.mean())
dataset['sex'] = dataset.sex.map({'female': 0, 'male': 1})


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_Train, X_Test, Y_Train, Y_Test=train_test_split(X,y,test_size=0.2,random_state=0)
SC = SS()
X_Train = SC.fit_transform(X_Train)
X_Test = SC.transform(X_Test)


print("\nGuassian NB")
GNB_Model = GaussianNB()
GNB_Model.fit(X_Train, Y_Train)
Gau_Pred = GNB_Model.predict(X_Test)
Accuracy=accuracy_score(Y_Test,Gau_Pred)
Accuracy_GNB=Accuracy*100
print(Accuracy_GNB)

print("\nSVM Classifier")
SVM_Model = SVC(kernel = 'linear',random_state=21)
SVM_Model.fit(X_Train, Y_Train)
SVM_Pred = SVM_Model.predict(X_Test)
Accuracy=accuracy_score(Y_Test,SVM_Pred)
Accuracy_SVM=Accuracy*100
print(Accuracy_SVM)

print("\nLogistic Regression")
LogReg_Model = LogisticRegression(random_state=1)
LogReg_Model.fit(X_Train, Y_Train)
LogReg_Pred = LogReg_Model.predict(X_Test)
Accuracy=accuracy_score(Y_Test,LogReg_Pred)
Accuracy_LR=Accuracy*100
print(Accuracy_LR)

print("\nNeural Network: MLP Classifier")
NNMLP_Model = MLPClassifier(random_state=1, max_iter=500,learning_rate_init=0.4)
NNMLP_Model.fit(X_Train, Y_Train)
NN_Pred = NNMLP_Model.predict(X_Test)
Accuracy=accuracy_score(Y_Test,NN_Pred)
Accuracy_NN=Accuracy*100
print(Accuracy_NN)

print("\nKNN Classifier")
KNN_Model= KNeighborsClassifier(n_neighbors=40,p=2)
KNN_Model.fit(X_Train, Y_Train)
KNN_Pred = KNN_Model.predict(X_Test)
Accuracy=accuracy_score(Y_Test,KNN_Pred)
Accuracy_NC=Accuracy*100
print(Accuracy_NC)

print("\nRandom Forest")
RandFor_Model = RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0,max_depth=1)
RandFor_Model.fit(X_Train, Y_Train)
RanFor_Pred = RandFor_Model.predict(X_Test)
Accuracy=accuracy_score(Y_Test,RanFor_Pred)
Accuracy_RF=Accuracy*100
print(Accuracy_RF)
print("\n")


##### Ensembling #######
from sklearn.ensemble import VotingClassifier

Voting_Model= VotingClassifier(
    estimators=[('lr', LogReg_Model), ('svc', SVM_Model),('NN',NNMLP_Model),('KNN',KNN_Model),('GNB',GNB_Model),('RF', RandFor_Model)],
    voting='hard', weights=[1,1,2,2,2,1])
for Model in (LogReg_Model, SVM_Model,NNMLP_Model,KNN_Model,GNB_Model,RandFor_Model, Voting_Model):
    Model.fit(X_Train, Y_Train)
    Pred = Model.predict(X_Test)
    Accuracy=accuracy_score(Y_Test, Pred)*100
    print(Model.__class__.__name__,":",f'{Accuracy:.2f}''\n')
    if (Model.__class__.__name__ == "VotingClassifier"):
      Accuracy_VC=Accuracy
