import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

train=pd.read_csv("train.csv") # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
test=pd.read_csv("test.csv") # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


print(train.head())
print(test.head())

print(train.dtypes)
train=train.drop(columns=["Name","Ticket","Cabin"])
x_train,x_test,y_train,y_test = model_selection.train_test_split(train[["Pclass","Parch","Fare"]],train[["Survived"]],test_size=0.25,random_state=0)

scoring ='accuracy'
models = [
    LogisticRegression(solver='liblinear', multi_class='ovr'),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC(gamma='auto'),
    MLPClassifier(hidden_layer_sizes=(10,5),activation='tanh',max_iter=800,
      learning_rate_init=0.005,learning_rate='adaptive'),
]
for model in models:
	kfold = model_selection.KFold(n_splits=5, random_state=6) # seed
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
	print(f"{type(model).__name__} : {cv_results.mean():.4f} ({cv_results.std():.4f})")