import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import column_or_1d
import re


pd.set_option('display.max_columns', 16)

name_y=["Survived"]
train=pd.read_csv("train.csv") # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
train_y=train[name_y]
train_x=train.drop(columns=name_y)
test_x=pd.read_csv("test.csv") # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

print(train_x.head())
print(train_y.head())
print(test_x.head())

print(train_x.dtypes)
print(train_y.dtypes)

def parse1stint(str):
	list=re.findall("[0-9]+",str)
	return int(list[0])/100 if len(list)>0 else 0
def parse1stchar(str):
	list=re.findall("[0-9]+",str)
	return ord(re.findall("[A-Z]",str)[0])-64 if len(list)>0 else 0


def feature_prep(df):
	df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
	df["SexInt"] = np.where(df.Sex=='female',0,1)
	# df.Embarked=df["Embarked"].fillna('C')
	# print(pd.get_dummies(df,columns=["Embarked"]))
	# df["Embarked"]=LabelEncoder().fit_transform(df["Embarked"].fillna('C'))
	df["EmbarkedInt"]=df.Embarked.apply(lambda x: ord(x) % 5 if isinstance(x,str) else 0)
	# df["Ageknown"]=np.where(df.Age>0,True,False)
	df["Ageknown"]=np.where(df.Age>0,1,0)
	df.Age=np.where(df.Age>0,df.Age,0)
	# df.Age=np.log10(df.Age+1)
	df.Age=np.sqrt(df.Age)
	df.Fare=np.log10(df.Fare+1)
	# df["CabinKnown"]=df.Cabin.apply(lambda x: False if isinstance(x,float) else True)
	df["CabinKnown"]=df.Cabin.apply(lambda x: 0 if isinstance(x,float) else 1)
	df["CabinLetter"]=df.Cabin.apply(lambda x: 0.0 if isinstance(x,float) else parse1stchar(x))
	df["CabinValue"]=df.Cabin.apply(lambda x: 0.0 if isinstance(x,float) else parse1stint(x))
	return df

feature_prep(train_x)
print(train_x.head())


print(train_x.dtypes)
print(train_y.dtypes)

# train=train.drop(columns=["Name","Ticket","Cabin"])
x_train,x_test,y_train,y_test = model_selection.train_test_split(
	train_x[["Pclass","SexInt","Age","SibSp","Parch","Fare","CabinLetter",
	#"CabinValue","CabinKnown","EmbarkedInt",
	]],
	column_or_1d(train_y),test_size=0.25,random_state=0)
# do NLP for Name

scoring ='accuracy'
models = [
	LogisticRegression(solver='liblinear', multi_class='ovr'),
	LinearDiscriminantAnalysis(),
	KNeighborsClassifier(),
	DecisionTreeClassifier(),
	RandomForestClassifier(n_estimators=100),
	# RandomForestClassifier(criterion='gini',n_estimators=1750,max_depth=7,min_samples_split=6,min_samples_leaf=6,max_features='auto',oob_score=True,n_jobs=-1,verbose=1),
	GaussianNB(),
	SVC(gamma='auto'),
	MLPClassifier(hidden_layer_sizes=(20,10),activation='tanh',max_iter=800,learning_rate_init=0.005,learning_rate='adaptive'),
]
for model in models:
	kfold = model_selection.KFold(n_splits=4, random_state=6) # seed
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
	print(f"{type(model).__name__} : {cv_results.mean():.4f} ({cv_results.std():.4f})")

