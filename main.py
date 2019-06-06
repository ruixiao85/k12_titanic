import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
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
	list=re.findall("[A-Z0-9]+",str)
	return ord(re.findall("[A-Z]",str)[0])-64 if len(list)>0 else 0
def to_lower(text):
	return text.lower()
def rm_puncs(text,punc=None):
	if not punc:
		import string
		punc=string.punctuation
	return "".join([char for char in text if char not in punc])
def rm_stops(text,stop=None):
	if not stop:
		from nltk.corpus import stopwords
		stop=stopwords.words('english')
	return "".join([char for char in text if char not in stop])
def token_keys(text,keys=None):
	tokens=re.split('\W+',text)
	if keys:
		tokens=[t for t in tokens if t in keys]
	return " ".join(tokens)
def trimcommaleft(text):
	sub=text.split(",")[-1]
	return sub

def vecfreqword(df,var,features):
	# cv=CountVectorizer(analyzer='word',stop_words=None,min_df=0.02)
	cv=TfidfVectorizer(preprocessor=trimcommaleft,analyzer='word',stop_words=None,max_features=features)
	cvo=cv.fit_transform(df[var])
	cvdf=pd.DataFrame(cvo.toarray(),index=df.index,columns=cv.get_feature_names())
	df=pd.concat([df,cvdf],axis=1)

	# min_count=len(df[var])*0.01
	# df[var]=df[var].apply(lambda x:to_lower(x)) # lower case
	# df[var]=df[var].apply(lambda x:rm_puncs(x)) # remove punctuation
	# df[var]=df[var].apply(lambda x:rm_stops(x)) # no need to remove stopwords such as "the" "a"
	# vec=CountVectorizer().fit(df[var])
	# bag_of_words=vec.transform(df[var])
	# sum_words=bag_of_words.sum(axis=0)
	# words_freq=[(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
	# words_freq=sorted(words_freq,key=lambda x:x[1],reverse=True)
	# keywords=[w for (w,c) in words_freq if c>min_count] # return keywords
	# df[var]=df[var].apply(lambda x:token_keys(x,keywords))

	return df

def diagnose(df):
	print(f"dataframe shape {df.shape}")
	print(df.head())
	# print(df.dtypes)
	for column in df.columns:
		idx_null=df[column].isnull()
		print(f' [{column:<12}] unique: {len(df[column].unique()):>5} missing: {idx_null.sum():>5} [{df[column].dtype}]')
		if idx_null.sum()>0:
			print(df[idx_null].head())


def feature_prep(df):
	df=vecfreqword(df,"Name",8)
	df=vecfreqword(df,"Ticket",5)
	# df['Age']=df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
	df["AgeKnown"]=np.where(df["Age"]>0,1,0); df["Age"]=np.where(df["Age"]>0,df["Age"],0)
	df["Age"]=np.sqrt(df["Age"])
	# df["Age"]=np.log(df["Age"]+1)
	# df=pd.get_dummies(df,columns=["Age"])
	# df["Sex"] = np.where(df["Sex"]=='female',0,1) # manual integer
	# df["Sex"]=LabelEncoder().fit_transform(df["Sex"]) # label encoder
	df=pd.get_dummies(df,columns=["Sex"]) # one hot encoder
	df["Embarked"]=df.Embarked.apply(lambda x: ord(x) % 5 if isinstance(x,str) else 0)
	# df=pd.get_dummies(df,columns=["Embarked"])
	# df["Fare"]=np.sqrt(df["Fare"])
	df["Fare"]=np.log10(df["Fare"]+1)
	df=pd.get_dummies(df,columns=["Pclass"]) # one hot encoder ,"SibSp","Parch"
	df["CabinKnown"]=df.Cabin.apply(lambda x: 0 if isinstance(x,float) else 1)
	df["CabinLetter"]=df.Cabin.apply(lambda x: 0 if isinstance(x,float) else parse1stchar(x))
	df["CabinValue"]=df.Cabin.apply(lambda x: 0.0 if isinstance(x,float) else parse1stint(x))
	df=pd.get_dummies(df,columns=["CabinLetter"])
	return df.drop(columns=["PassengerId","Name","Ticket","Cabin","CabinKnown",
		# "AgeKnown", "CabinValue","CabinLetter",
		])

diagnose(train_x)

x=pd.concat([train_x,test_x],axis=0,keys=['train','test'])
x=feature_prep(x)
train_x=x.loc['train']
test_x=x.loc['test']

print(train_x.head())

x_train,x_test,y_train,y_test = model_selection.train_test_split(
	train_x,column_or_1d(train_y),
	test_size=0.25,random_state=0)


scoring ='accuracy'
models = [
	# GaussianNB(),
	LogisticRegression(solver='liblinear', multi_class='ovr'),
	LinearDiscriminantAnalysis(),
	KNeighborsClassifier(),
	# DecisionTreeClassifier(),
	RandomForestClassifier(n_estimators=100),
	# RandomForestClassifier(criterion='gini',n_estimators=1750,max_depth=7,min_samples_split=6,min_samples_leaf=6,max_features='auto',oob_score=True,n_jobs=-1,verbose=1),
	SVC(gamma='auto'),
	MLPClassifier(hidden_layer_sizes=(32,24),activation='tanh',max_iter=800,early_stopping=True,learning_rate_init=0.01,learning_rate='adaptive'),
]
sum_score=[]
for model in models:
	kfold = model_selection.KFold(n_splits=4, random_state=6) # seed
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
	print(f"{type(model).__name__} : {cv_results.mean():.4f} ({cv_results.std():.4f})")
	sum_score.append(cv_results)
print(f"Average Score: {np.mean(sum_score):.4f}")


