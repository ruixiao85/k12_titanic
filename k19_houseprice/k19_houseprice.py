import keras
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from keras import Sequential,layers
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import nadam,sgd,adam
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn
from scipy.special._ufuncs import boxcox1p
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,OrdinalEncoder,RobustScaler
from sklearn.utils import column_or_1d
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from vecstack import stacking
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor

train=pd.read_csv("train.csv",index_col=0)
train=train[train['GrLivArea']<4000] # outliers
test_x=pd.read_csv("test.csv",index_col=0)

def transe(x):
	return np.log1p(x)
def reverse(x):
	return np.expm1(x)

def diagnose(df):
	print(f"dataframe shape {df.shape}")
	print(df.head())
	# print(df.dtypes)
	for column in df.columns:
		idx_null=df[column].isnull()
		print(f' [{column:<12}] unique: {len(df[column].unique()):>5} missing: {idx_null.sum():>5} [{df[column].dtype}]')
		# if idx_null.sum()>0:
		# 	print(df[idx_null].head())

def vecfreqword(df,var,cv,upper_limit=None):
	cvo=cv.fit_transform(df[var])
	cvdf=pd.DataFrame(cvo.toarray(),index=df.index,columns=cv.get_feature_names())
	if upper_limit:
		cvdf=cvdf.clip(upper=upper_limit)
	print(cv.get_feature_names())
	df=pd.concat([df,cvdf],axis=1)
	return df

def feature_prep(df):
	year_ratio=500.0
	log_div=np.log(10)
	df['MSZoning']=df['MSZoning'].fillna(" ")
	# df=df.apply(lambda x:x.str.strip() if x.dtype=="object" else x) # consider remove space
	df=df.apply(lambda x:x.str.replace(" ","") if x.dtype=="object" else x) # consider remove space
	# df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
	df['LotArea']=(np.log(df['LotArea']+1))#.astype(np.int)
	# df['LotFrontage']=df['LotFrontage'].apply(lambda x: round(np.log(x+1)/log_div) if x>0 else 0) # log int
	# df['LotFrontage']=df['LotFrontage'].apply(lambda x: round(np.sqrt(x)) if x>0 else 0) # sqrt int
	df['LotFrontage']=df['LotFrontage'].apply(lambda x: np.log(x+1) if x>0 else 0) # log float
	# df['LotFrontage']=df['LotFrontage'].apply(lambda x: np.sqrt(x) if x>0 else 0) # sqrt float
	# df['LotArea']=(np.log(df['LotArea']+1)).astype(np.int) # bucket
	df['Conditions']=df["Condition1"].map(str) +" "+ df["Condition2"]
	df=vecfreqword(df,"Conditions",CountVectorizer(analyzer='word',min_df=0.005,ngram_range=(1,1)),upper_limit=None)

	df['YearBuilt']=(df['YearBuilt']/year_ratio)#.astype(np.int)
	df['YearRemodAdd']=(df['YearRemodAdd']/year_ratio)#.astype(np.int)

	df['Exterior2nd'][df['Exterior1st']==df['Exterior2nd']]=""
	# df['Exteriors']=(df["Exterior1st"].map(str)+" "+df["Exterior2nd"]).fillna("")
	# df=vecfreqword(df,"Exteriors",CountVectorizer(analyzer='word',max_features=20,ngram_range=(1,1)),upper_limit=1)

	# df['MasVnrType']=df['MasVnrType'].fillna('None')
	# df['MasVnrType']=df['MasVnrType'].fillna('NA')
	df['MasVnrArea']=(np.log(df['MasVnrArea'].fillna(0)+1)/log_div).astype(np.int)
	df['BsmtFinSF1']=(np.log(df['BsmtFinSF1'].fillna(0)+1)/log_div).astype(np.int)
	df['BsmtFinSF2']=(np.log(df['BsmtFinSF2'].fillna(0)+1)/log_div).astype(np.int)
	df['BsmtUnfSF']=(np.log(df['BsmtUnfSF'].fillna(0)+1)/log_div).astype(np.int)
	df['TotalBsmtSF']=(np.log(df['TotalBsmtSF'].fillna(0)+1)/log_div).astype(np.int)
	# df['1st2ndFlrSF']=
	# df['1st2ndFlrSF']=(np.log(df['1st2ndFlrSF']+1)/log_div).astype(np.int)
	# df['HighQualFinSF']=(np.log(df['TotalBsmtSF']+df['1stFlrSF']+df['2ndFlrSF']-df['LowQualFinSF']+1)/log_div).astype(np.int)
	df['1stFlrSF']=(np.log(df['1stFlrSF']+1)/log_div)#.astype(np.int)
	df['2ndFlrSF']=(np.log(df['2ndFlrSF']+1)/log_div)#.astype(np.int)
	df['LowQualFinSF']=(np.log(df['LowQualFinSF']+1)/log_div)#.astype(np.int)
	df['GrLivArea']=(np.log(df['GrLivArea']+1)/log_div).astype(np.int)

	df['TotRmsAbvGrd']=(df['TotRmsAbvGrd']/2).astype(np.int)
	df['GarageYrBlt']=(df['GarageYrBlt'].fillna(0)/year_ratio)#.astype(np.int)
	df['GarageArea']=(np.log(df['GarageArea'].fillna(0)+1)/log_div).astype(np.int)

	df['WoodDeckSF']=(np.log(df['WoodDeckSF']+1)/log_div).astype(np.int)
	df['OpenPorchSF']=(np.log(df['OpenPorchSF']+1)/log_div).astype(np.int)
	df['EnclosedPorch']=(np.log(df['EnclosedPorch']+1)/log_div).astype(np.int)
	df['3SsnPorch']=(np.log(df['3SsnPorch']+1)/log_div).astype(np.int)
	df['ScreenPorch']=(np.log(df['ScreenPorch']+1)/log_div).astype(np.int)
	df['PoolArea']=(np.log(df['PoolArea']+1)/log_div).astype(np.int)

	df['MiscVal']=(np.log(df['MiscVal']+1)/log_div).astype(np.int)
	df['MoSold']=(df['MoSold']/3).astype(np.int)
	df['YrSold']=(df['YrSold']/year_ratio)#.astype(np.int)

	df['YearBuilt']=df['YrSold']-df['YearBuilt']
	df['YearRemodAdd']=df['YrSold']-df['YearRemodAdd']
	df['GarageYrBlt']=df['YrSold']-df['GarageYrBlt']

	RIII={'Reg':4, #	Regular
	   'IR1':3, # Slightly irregular
	   'IR2':2, # Moderately Irregular
	   'IR3':1, # Irregular
	}
	EGTFPN={'Ex':6, # Excellent (100+ inches)
		   'Gd':5, # Good (90-99 inches)
		   'TA':4, # Typical (80-89 inches)
		   'Fa':3, # Fair (70-79 inches)
		   'Po':2, # Poor (<70 inches
		   'NA':1, # No Basement
	}
	GMS={
       'Gtl':1, # Gentle slope
       'Mod':2, # Moderate Slope
       'Sev':3, # Severe Slope
	}
	GAMNN={
       'Gd':5, # Good Exposure
       'Av':4, # Average Exposure (split levels or foyers typically score average or above)
       'Mn':3, # Mimimum Exposure
       'No':2, # No Exposure
       'NA':1, # No Basement
	}
	QABRLUN={
       'GLQ':7, # Good Living Quarters
       'ALQ':6, # Average Living Quarters
       'BLQ':5, # Below Average Living Quarters
       'Rec':4, # Average Rec Room
       'LwQ':3, # Low Quality
       'Unf':2, # Unfinshed
       'NA':1, # No Basement
	}
	FRUN={
       'Fin':4, # Finished
       'RFn':3, # Rough Finished
       'Unf':2, # Unfinished
       'NA':1, # No Garage
	}
	YPN={
       'Y':3, # Paved
       'P':2, # Partial Pavement
       'N':1, # Dirt/Gravel
	}
	GMGMN={
		   'GdPrv':5, # Good Privacy
		   'MnPrv':4, # Minimum Privacy
		   'GdWo':3, # Good Wood
		   'MnWw':2, # Minimum Wood/Wire
		   'NA':1, # No Fence
	}
	def map_fillna_know(df,var,map,na=0,ifknown=False):
		if ifknown:
			df[var+"Known"]=np.where(df[var].isnull(),0,1)
		df[var]=df[var].map(map).fillna(na)
		return df
	df=map_fillna_know(df,'LotShape',RIII)
	df=map_fillna_know(df,'LandSlope',GMS)
	df=map_fillna_know(df,'ExterQual',EGTFPN)
	df=map_fillna_know(df,'ExterCond',EGTFPN)
	df=map_fillna_know(df,'BsmtQual',EGTFPN)
	df=map_fillna_know(df,'BsmtCond',EGTFPN)
	df=map_fillna_know(df,'BsmtExposure',GAMNN)
	df=map_fillna_know(df,'BsmtFinType1',QABRLUN)
	df=map_fillna_know(df,'BsmtFinType2',QABRLUN)
	df=map_fillna_know(df,'HeatingQC',EGTFPN)
	df=map_fillna_know(df,'KitchenQual',EGTFPN)
	df=map_fillna_know(df,'FireplaceQu',EGTFPN)
	df=map_fillna_know(df,'GarageFinish',FRUN)
	df=map_fillna_know(df,'GarageQual',EGTFPN)
	df=map_fillna_know(df,'GarageCond',EGTFPN)
	df=map_fillna_know(df,'PavedDrive',YPN)
	df=map_fillna_know(df,'PoolQC',EGTFPN)
	df=map_fillna_know(df,'Fence',GMGMN)

	# 'OverallQual','OverallCond',
	# 	 'LotArea',
	# 'YearBuilt','YearRemodAdd','YrSold','GarageYrBlt',
	onehot_features=['MSSubClass','MSZoning','Street','Alley', 'LandContour',
					 'Utilities','LotConfig','Neighborhood',
					 'Conditions',  'BldgType', 'HouseStyle',
					 'RoofStyle','RoofMatl',
					 # 'Exteriors',
					 'LotFrontage', 'MasVnrArea','BsmtFinSF1','BsmtFinSF2',
	 # 'LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2',
	# 'HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',
					 'Exterior1st','Exterior2nd','MasVnrType','BsmtUnfSF','TotalBsmtSF','GrLivArea',
					 'Foundation',
					 #'1stFlrSF','2ndFlrSF', 'LowQualFinSF',
					 'Heating','CentralAir','Electrical',
					 'BsmtFullBath','BsmtHalfBath', 'FullBath','HalfBath','BedroomAbvGr',
					 'KitchenAbvGr','TotRmsAbvGrd','Functional','Fireplaces','GarageType',
					 'GarageArea',
					 'GarageCars',
					 'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
					 'MiscFeature','MiscVal','MoSold','SaleType','SaleCondition']
	ord_features=[
	]
	rank_features=[
	]
	remove_features=[
		'Condition1','Condition2',
	]
	df=pd.get_dummies(df,columns=onehot_features)
	for f in rank_features:
		df[f]=df[f].rank()
	for f in ord_features:
		temp=OrdinalEncoder().fit_transform(df[[f]])
		df[f]=temp.astype(np.int)
	return df.drop(columns=remove_features)

# Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,
# LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,
# OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,
# Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,
# Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,
# TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,
# BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,
# Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,
# PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,
# MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice
name_y=["SalePrice"]
train_y=train[name_y]
train_y=transe(train_y)
train_x=train.drop(columns=name_y)
# print(train_x.head())

# print(test_x.head())

pd.set_option('display.max_columns',16)
diagnose(train_x)
# diagnose(test_x)


x=pd.concat([train_x,test_x],axis=0,keys=['train','test'])
x=feature_prep(x)
# x=pd.DataFrame(RobustScaler(with_centering=True,with_scaling=True,quantile_range=(25.0, 75.0), copy=True).fit_transform(x),index=x.index,columns=x.columns)
x=pd.DataFrame(RobustScaler(with_centering=False,with_scaling=True,quantile_range=(25.0, 75.0), copy=True).fit_transform(x),index=x.index,columns=x.columns)
train_x=x.loc['train']
test_x=x.loc['test']

# x=sc.fit_transform(x)

# x_train,x_test,y_train,y_test=model_selection.train_test_split(train_x,column_or_1d(train_y),
# 	test_size=0.25,random_state=0)
diagnose(train_x)

run_keras=False
if run_keras:
	inlayer=layers.Input(shape=(train_x.shape[1],))
	x=layers.Dense(1024,activation='tanh')(inlayer)
	x=layers.Dense(768,activation='tanh')(x)
	# x=layers.Dense(384,activation='tanh')(x)
	x=layers.Dense(1,kernel_initializer='normal')(x)
	model=keras.Model(inputs=inlayer, outputs=x)

	# y=layers.Dense(300,activation='tanh')(inlayer)
	# y=layers.Dense(160,activation='tanh')(y)
	# y=layers.Dense(70,activation='tanh')(y)
	# m =layers.concatenate([x, y])
	# m=layers.Dense(350,activation='tanh')(m)
	# m=layers.Dense(350,activation='tanh')(m)
	# m=layers.Dense(768,activation='tanh')(m)
	# m=layers.Dense(1,kernel_initializer='normal')(m)
	# model=keras.Model(inputs=inlayer, outputs=m)

	print(model.summary())

	model.compile(loss='mean_squared_error',optimizer=nadam(lr=5e-5),metrics=['mse'])
	best_model=model.fit(np.array(train_x),column_or_1d(train_y),epochs=500,batch_size=32,validation_split=0.2,callbacks=[
	EarlyStopping(monitor='val_loss',min_delta=0,patience=16,verbose=1,mode='auto',baseline=None,restore_best_weights=True)
	])
	test_y=pd.DataFrame(model.predict(test_x),index=test_x.index,columns=["SalePrice"])
	test_y=reverse(test_y)
	test_y.to_csv(f"submission_{type(model).__name__}.csv",index=True,index_label="Id")

run_sklearn=True
if run_sklearn:
	scoring='neg_mean_squared_error'
	models=[
		# DecisionTreeRegressor(),
		# KNeighborsRegressor(),
		# GradientBoostingRegressor(),
		# AdaBoostRegressor(),
		Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
	   normalize=False, positive=False, precompute=False, random_state=1,
	   selection='cyclic', tol=0.0001, warm_start=False),
		RandomForestRegressor(n_estimators=100),
		# ExtraTreesRegressor(),
		# XGBRegressor(),
		SVR(gamma='auto'),
		Ridge(),
		# MLPRegressor(hidden_layer_sizes=(512),activation='tanh',max_iter=800,early_stopping=True,learning_rate_init=0.001,learning_rate='adaptive'),
		# MLPRegressor(hidden_layer_sizes=(512,384),activation='tanh',max_iter=800,early_stopping=True,learning_rate_init=0.01,learning_rate='adaptive'),
	]
	sum_score=[]
	sum_result=None
	for model in models:
		kfold=model_selection.KFold(n_splits=5,random_state=6)  # seed
		# cv_results=model_selection.cross_val_score(model,x_train,y_train,cv=kfold,scoring=scoring)
		cv_results=model_selection.cross_val_score(model,train_x,column_or_1d(train_y),cv=kfold,scoring=scoring)
		cv_results=np.sqrt(-1*cv_results)
		sum_score.append(cv_results)
		print(f"{type(model).__name__} : {cv_results.mean():.4f} ({cv_results.std():.4f})")
		best_model=model.fit(train_x,column_or_1d(train_y))
		test_y=pd.DataFrame(best_model.predict(test_x),index=test_x.index,columns=["SalePrice"])
		test_y=reverse(test_y)
		sum_result=test_y if sum_result is None else sum_result.add(test_y)
		test_y.to_csv(f"submission_{type(model).__name__}.csv",index=True,index_label="Id")
	print(f"Average Score: {np.mean(sum_score):.4f}")
	n_result=len(sum_score)
	sum_result.div(n_result).round().astype(np.int32).to_csv(f"submission_sum{n_result}.csv",index=True,index_label="Id")

	# https://www.kaggle.com/c/allstate-claims-severity/discussion/25743
	S_1_train, S_1_test = stacking(models, train_x,column_or_1d(train_y), test_x, regression = True, verbose = 2)
	# S_2_train,S_2_test=stacking([Ridge()],S_1_train,column_or_1d(train_y),S_1_test,regression=True,verbose=2)
	model=XGBRegressor(seed=0,nthread=-1,learning_rate=0.1,n_estimators=100,max_depth=3)
	model=model.fit(S_1_train,column_or_1d(train_y))
	test_y=model.predict(S_1_test)
	test_y=pd.DataFrame(test_y,index=test_x.index,columns=["SalePrice"])
	test_y=reverse(test_y)
	test_y.to_csv(f"submission_stacking.csv",index=True,index_label="Id")

# note that we compute only oof (mode='oof').
# S_train, _ =StackingCVRegressor(models_L1,
#                       X_train, y_train, None,
#                       regression=True,
#                       mode='oof',
#                       random_state=0,
#                       verbose=2)

# model_L2 = LinearRegression()
# _ = model_L2.fit(S_train, y_train)
# # save model in file if you need
# [4] Then new test set (X_test_new) comes. We load our 1st level models and predict new test set to get stacked features (S_test_new):
#
# y_pred_L1_0 = model_L1_0.predict(X_test_new)
# y_pred_L1_1 = model_L1_1.predict(X_test_new)
# S_test_new = np.c_[y_pred_L1_0, y_pred_L1_1]
# [5] Then we load our 2nd level model and predict S_test_new to get final prediction:
#
# y_pred_new = model_L2.predict(S_test_new)
# [6] Each time new test set comes we just repeat [4] and [5]

