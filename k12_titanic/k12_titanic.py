import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from sklearn import model_selection
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.utils import column_or_1d
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

train=pd.read_csv("train.csv",index_col=0)
test_x=pd.read_csv("test.csv",index_col=0)

def diagnose(df):
	print(f"dataframe shape {df.shape}")
	print(df.head())
	# print(df.dtypes)
	for column in df.columns:
		idx_null=df[column].isnull()
		print(f' [{column:<12}] unique: {len(df[column].unique()):>5} missing: {idx_null.sum():>5} [{df[column].dtype}]')
		if idx_null.sum()>0:
			print(df[idx_null].head())


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


def trimname(text):
	# return text
	# return text.split(",")[-1]
	return text.split(".")[0]
	# return text.split(",")[-1].split(".")[0]
def trimchar(text):
	# exclude=['.'] #
	# exclude=['.','O'] #
	# exclude=['.','/'] #
	# exclude=['.','Q'] #
	# exclude=['Q'] #
	exclude=['.','O','Q'] #
	return ''.join([c for c in text if c not in exclude])

def vecfreqword(df,var,cv):
	cvo=cv.fit_transform(df[var])
	cvdf=pd.DataFrame(cvo.toarray(),index=df.index,columns=cv.get_feature_names())
	print(cv.get_feature_names())
	df=pd.concat([df,cvdf],axis=1)
	return df

def feature_prep(df):
	# df['Family_Size']=df['SibSp']+df['Parch'] # engineer
	df=vecfreqword(df,"Name",CountVectorizer(preprocessor=trimname,analyzer='word',min_df=0.008,ngram_range=(1,1))) # ,max_features=8
	# df=vecfreqword(df,"Name",CountVectorizer(preprocessor=trimcommaleft,analyzer='word',lowercase=False,
	# 	vocabulary=	["Capt","Col","Major","Jonkheer","Don","Sir","Dr","Rev","Countess","Dona","Mme","Mlle","Ms","Mrs","Mr","Miss","Master","Lady"]))
	# df=vecfreqword(df,"Ticket",CountVectorizer(analyzer='word',max_features=8,ngram_range=(1,2)))
	df=vecfreqword(df,"Ticket",CountVectorizer(preprocessor=trimchar,analyzer='word',min_df=0.005,ngram_range=(1,1)))
	# df['Age']=df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
	df["AgeKnown"]=np.where(df["Age"]>0,1,0)
	df["Age"]=np.where(df["Age"]>0,df["Age"],0)
	# df['Age*Class']=df['Age']*df['Pclass'] # engineer
	df["Age"]=np.sqrt(df["Age"])
	# df["Age"]=np.log(df["Age"]+1)
	# df=pd.get_dummies(df,columns=["Age"])
	# df["Sex"] = np.where(df["Sex"]=='female',0,1) # manual integer
	# df["Sex"]=LabelEncoder().fit_transform(df["Sex"]) # label encoder
	df=pd.get_dummies(df,columns=["Sex"])  # one hot encoder
	df["Embarked"]=df.Embarked.apply(lambda x:ord(x)%5 if isinstance(x,str) else 0)
	# df=pd.get_dummies(df,columns=["Embarked"])
	df["Fare"]=np.where(df["Fare"]>0,df["Fare"],0)
	# df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1) # engineer
	# df["Fare"]=np.sqrt(df["Fare"])
	df["Fare"]=np.log10(df["Fare"]+1)
	df=pd.get_dummies(df,columns=["Pclass"])  # one hot encoder ,"SibSp","Parch"
	df["CabinKnown"]=df.Cabin.apply(lambda x:0 if isinstance(x,float) else 1)
	df["CabinLetter"]=df.Cabin.apply(lambda x:0 if isinstance(x,float) else parse1stchar(x))
	df["CabinValue"]=df.Cabin.apply(lambda x:0.0 if isinstance(x,float) else parse1stint(x))
	df=pd.get_dummies(df,columns=["CabinLetter"])
	return df.drop(columns=["Name","Ticket","Cabin", "CabinKnown", #   "AgeKnown","CabinValue", "CabinLetter",
	])


# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
train=pd.read_csv("train.csv",index_col=0)
name_y=["Survived"]
train_y=train[name_y]
train_x=train.drop(columns=name_y)
# print(train_x.head())

# print(test_x.head())

pd.set_option('display.max_columns',16)
diagnose(train_x)
# diagnose(test_x)


x=pd.concat([train_x,test_x],axis=0,keys=['train','test'])
x=feature_prep(x)
train_x=x.loc['train']
test_x=x.loc['test']

print(train_x.head())

# x_train,x_test,y_train,y_test=model_selection.train_test_split(train_x,column_or_1d(train_y),
# 	test_size=0.25,random_state=0)

scoring='accuracy'
# scoring='f1'
r=[0.0001,0.001,0.1,1,10,50,100]
models=[
	# GaussianNB(),
	# DecisionTreeClassifier(),
	# KNeighborsClassifier(),
	# LogisticRegression(solver='liblinear',multi_class='ovr'),
	# LinearDiscriminantAnalysis(),
	# GradientBoostingClassifier(),
	# HistGradientBoostingClassifier(),
	# AdaBoostClassifier(),
	# RandomForestClassifier(n_estimators=100),
	# RandomForestClassifier(criterion='gini',n_estimators=1750,max_depth=7,min_samples_split=6,min_samples_leaf=6,max_features='auto',oob_score=True,n_jobs=-1,verbose=1),
	SVC(gamma='auto'),
	# GridSearchCV(estimator=make_pipeline(StandardScaler(),SVC(gamma='auto',random_state=1)),cv=5,
	# 	param_grid=[{'svc__C':r, 'svc__kernel':['linear']}, {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]),
	MLPClassifier(hidden_layer_sizes=(128,64),activation='tanh',max_iter=800,early_stopping=True,learning_rate_init=0.01,learning_rate='adaptive'),
]
sum_score=[]
sum_result=None
for model in models:
	kfold=model_selection.KFold(n_splits=6,random_state=6)  # seed
	# cv_results=model_selection.cross_val_score(model,x_train,y_train,cv=kfold,scoring=scoring)
	cv_results=model_selection.cross_val_score(model,train_x,column_or_1d(train_y),cv=kfold,scoring=scoring)
	sum_score.append(cv_results)
	print(f"{type(model).__name__} : {cv_results.mean():.4f} ({cv_results.std():.4f})")
	best_model=model.fit(train_x,column_or_1d(train_y))
	test_y=pd.DataFrame(best_model.predict(test_x).astype(np.int),index=test_x.index,columns=["Survived"])
	sum_result=test_y if sum_result is None else sum_result.add(test_y)
	test_y.to_csv(f"submission_{type(model).__name__}.csv",index=True,index_label="PassengerId")
print(f"Average Score: {np.mean(sum_score):.4f}")
n_result=len(sum_score)
sum_result.div(n_result).round().astype(np.int32).to_csv(f"submission_ave{n_result}.csv",index=True,index_label="PassengerId")


