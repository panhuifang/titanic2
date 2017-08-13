# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:36:39 2017

@author: hp
"""
'''
survival:是否幸存，0为否，1为是
pclass:票的等级，1代表一等仓，2代表二等仓，3代表三等仓
sex:性别，male：男性，female：女性
Age:年龄
sibsp:兄弟姐妹或配偶同行的数量
parch:父母或子女同行的数量
ticket:票号
fare:票价
cabin:客仓号
embarked:登船口
'''
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
import time
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA
import random as rd
from sklearn.cluster import KMeans


data_train=pd.read_csv('...train.csv')
data_test=pd.read_csv('...test.csv')

data_titanic=pd.concat([data_train,data_test])
data_titanic.reset_index(inplace=True)
data_titanic.drop('index',axis=1,inplace=True)
data_titanic=data_titanic.reindex_axis(data_train.columns,axis=1)
#数据初探
fig=plt.figure()
fig.set(alpha=0.02)
plt.subplot2grid((2,3),(0,0))
data_titanic.Survived.value_counts().plot(kind='bar')

plt.subplot2grid((2,3),(0,1))
data_titanic.Pclass.value_counts().plot(kind='bar')

plt.subplot2grid((2,3),(1,0),colspan=2)
data_titanic.Age[data_titanic.Pclass==1].plot(kind='kde')
data_titanic.Age[data_titanic.Pclass==2].plot(kind='kde')
data_titanic.Age[data_titanic.Pclass==3].plot(kind='kde')
plt.legend((u'头等舱',u'二等舱',u'三等舱'),loc='best')#头等舱人数最多的集中在40岁。三等舱集中在20岁

fig=plt.figure(figsize=(100,2))
fig.set(alpha=0.2)
survived_1=data_titanic.Pclass[data_titanic.Survived==1].value_counts()
survived_2=data_titanic.Pclass[data_titanic.Survived==0].value_counts()
Pclass_df=pd.DataFrame({u'获救':survived_1,u'未获救':survived_2})
Pclass_df.plot(kind='bar',stacked=True)
plt.show()#一等舱被救比例高

fig=plt.figure(figsize=(100,2))
fig.set(alpha=0.2)
survivedsex_1=data_titanic.Sex[data_titanic.Survived==1].value_counts()
survivedsex_2=data_titanic.Sex[data_titanic.Survived==0].value_counts()
Pclass_df=pd.DataFrame({u'获救':survivedsex_1,u'未获救':survivedsex_2})
Pclass_df.plot(kind='bar',stacked=True)
plt.show()#女性被救比例高

fig=plt.figure()
fig.set(alpha=0.2)
plt.title(u"船舱等级和性别获救关系")

ax1=fig.add_subplot(141)
data_titanic.Survived[data_titanic.Sex=='female'][data_titanic.Pclass!=3].value_counts().plot(kind='bar',label=u"女性头等舱",color='red')
ax1.set_xticklabels([u"获救",u"未获救"],rotation=0)

ax2=fig.add_subplot(142,sharey=ax1)
data_titanic.Survived[data_titanic.Sex=='female'][data_titanic.Pclass==3].value_counts().plot(kind='bar',label=u"女性普通舱",color='red')
ax2.set_xticklabels([u"获救",u"未获救"],rotation=0)

ax3=fig.add_subplot(143,sharey=ax1)
data_titanic.Survived[data_titanic.Sex=='male'][data_titanic.Pclass!=3].value_counts().plot(kind='bar',label=u"男性头等舱",color='red')
ax3.set_xticklabels([u"获救",u"未获救"],rotation=0)

ax4=fig.add_subplot(144,sharey=ax1)
data_titanic.Survived[data_titanic.Sex=='male'][data_titanic.Pclass==3].value_counts().plot(kind='bar',label=u"男性普通舱",color='red')
ax4.set_xticklabels([u"获救",u"未获救"],rotation=0)


#Cabin
def getCabinLetter(Cabin):
    match=re.compile("(['a-zA-Z']+)").search(str(Cabin))
    if match:
        return match.group()
    else:
        return 'U'

def getCabinNumber(Cabin):
    match=re.compile("([0-9]+)").search(str(Cabin))
    if match:
        return match.group()
    else:
        return 0

#def precesscabin():
data_titanic['cabin_Letter']=data_titanic['Cabin'].map(lambda x:getCabinLetter(x))
#data_titanic['cabin_Number']=data_titanic['Cabin'].map(lambda x:getCabinNumber(x)).astype(int)+1
data_titanic.cabin_Letter[data_titanic['cabin_Letter']=='nan']='U'
data_titanic['cabin_Letter']=pd.factorize(data_titanic['cabin_Letter'])[0]
#scaler=preprocessing.StandardScaler()
#data_titanic['cabinnumber_scaler']=scaler.fit_transform(data_titanic['cabin_Number'])
#cabin_letter_dum=pd.get_dummies(data_titanic['cabin_Letter']).rename(columns=lambda x:'cabin'+str(x))
#data_titanic = pd.concat([data_titanic,cabin_letter_dum],axis=1)
#Embarked
data_titanic['Embarked'].groupby(data_titanic['Embarked']).agg('count')
data_titanic['Embarked'][data_titanic['Embarked'].isnull()]='S'
data_titanic['Embarked_fac']=pd.factorize(data_titanic['Embarked'])[0]
'''
Embarked_data=pd.get_dummies(data_titanic['Embarked'])
Embarked_data=Embarked_data.rename(columns=lambda x:'Embarked_'+str(x))
data_titanic=pd.concat([data_titanic,Embarked_data],axis=1)
'''
#scaler
#Fare
#if data_titanic['Fare']:
#def precessfare():    
data_titanic['Fare'].describe()
scaler=preprocessing.StandardScaler()
fare_scaler_fit=scaler.fit(data_titanic['Fare'])
data_titanic['fare_scaler']=fare_scaler_fit.fit_transform(data_titanic['Fare'])

#Ticket
def getticketletter(data_titanic):
    match=re.compile("([a-zA-Z\.\/]+)").search(data_titanic)
    if match:
        return match.group()
    else:
        return 'U'

def getticketnumber(data_titanic):
    match=re.compile("[\d]+$").search(data_titanic)
    if match:
        return match.group()
    else:
        return '0'
    '''
#def precessticket():
data_titanic['ticket_zz']=data_titanic['Ticket'].map(lambda x : getticketletter(x.upper()))
data_titanic['ticket_zz'].groupby(data_titanic['ticket_zz']).agg('count')
data_titanic['ticket_zz']=data_titanic['ticket_zz'].map(lambda x:re.sub('[\.?\/?]',"",x))
data_titanic['ticket_zz']=data_titanic['ticket_zz'].map(lambda x:re.sub('STON',"SOTON",x))
data_titanic['ticket_zz']=pd.factorize(data_titanic['ticket_zz'])[0]

ticket_letter_dum=pd.get_dummies(data_titanic['ticket_zz']).rename(columns=lambda x:'ticket'+str(x))
data_titanic=pd.concat([data_titanic,ticket_letter_dum],axis=1)
data_titanic['ticket_zs']=data_titanic['Ticket'].map(lambda x : getticketnumber(x))
data_titanic['ticketnumberlen']=data_titanic['ticket_zs'].map(lambda x : len(x)).astype(int)
data_titanic['ticketstartnumber']=data_titanic['ticket_zs'].map(lambda x : x[0:1]).astype(int)
#scaler
ticket_zs_scaler=preprocessing.StandardScaler().fit(data_titanic['ticket_zs'])
data_titanic['ticket_zs_scaler']=ticket_zs_scaler.fit_transform(data_titanic['ticket_zs'])
data_titanic.drop('ticket_zs',axis=1,inplace=True)
 '''  
#Parch sibsp
#def precesspach_sibsp():
x=data_titanic.groupby(['Parch','Survived'])
pd.DataFrame(x.count()['PassengerId'])
data_titanic['Parch'].groupby([data_titanic['Parch'],data_titanic['Survived']]).agg('count')
data_titanic['txrs']=data_titanic['Parch']+data_titanic['SibSp']
#scaler
scaler=preprocessing.StandardScaler()
data_titanic['Parch_scaler']=scaler.fit_transform(data_titanic['Parch'])        
data_titanic['SibSp_scaler']=scaler.fit_transform(data_titanic['SibSp']) 
data_titanic['txrs_scaler']=scaler.fit_transform(data_titanic['txrs'])       
#dummy
'''
parchdum=pd.get_dummies(data_titanic['Parch']).rename(columns=lambda x:'parch'+str(x))
sibspdum=pd.get_dummies(data_titanic['SibSp']).rename(columns=lambda x:'sibsp'+str(x))
txrsdum=pd.get_dummies(data_titanic['txrs']).rename(columns=lambda x:'sibsp'+str(x))
'''
#age
#def precessage():
data_titanic['Age'].describe()
fig=plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((1,1),(0,0))

data_titanic.Age[data_titanic['Survived']==1].plot(kind='kde')
data_titanic.Age[data_titanic['Survived']==0].plot(kind='kde')
Age_data=data_titanic[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
Age_notnull=Age_data[Age_data['Age'].notnull()]
Age_null=Age_data[Age_data['Age'].isnull()].as_matrix()
y=Age_notnull.ix[:,0]
x=Age_notnull.ix[:,1:]
#随即森林预测年龄
model = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
model.fit(x,y)
predictedage=model.predict(Age_null[:,1:])
data_titanic.loc[(data_titanic.Age.isnull()),'Age']=predictedage

kmeans=KMeans(n_clusters=4,random_state=4,init='random')
age_cluster=kmeans.fit_transform(data_titanic[['Age']])
data_titanic['age_lable']=kmeans.labels_
scaler = preprocessing.StandardScaler()
data_titanic['age_lable_scaled'] = scaler.fit_transform(data_titanic['age_lable'])

#def precesssexpclass():
#Sex Pclass
sex_dum=pd.get_dummies(data_titanic['Sex']).rename(columns=lambda x : 'Sex_'+str(x))
#pclass_dum=pd.get_dummies(data_titanic['Pclass']).rename(columns=lambda x : 'Pclass_'+str(x))
data_titanic=pd.concat([data_titanic,sex_dum],axis=1)
#data_titanic['Pclass_fac']=pd.factorize(data_titanic['Pclass'])[0]
#scaler=preprocessing.StandardScaler()
#data_titanic['Pclass_scaler']=scaler.fit_transform(data_titanic['Pclass_fac'])
#Name
#def precessname():
data_titanic['Names']=data_titanic['Name'].map(lambda x:len(re.split(' ',x)))
data_titanic['Title'] =data_titanic['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
data_titanic['Title'][data_titanic.Title == 'Jonkheer'] = 'Master'
data_titanic['Title'][data_titanic.Title.isin(['Ms','Mlle'])] = 'Miss'
data_titanic['Title'][data_titanic.Title == 'Mme'] = 'Mrs'
data_titanic['Title'][data_titanic.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
data_titanic['Title'][data_titanic.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
data_titanic['Title_id'] = pd.factorize(data_titanic['Title'])[0]+1
#scaler = preprocessing.StandardScaler()
#data_titanic['Title_id_scaled'] = scaler.fit_transform(data_titanic['Title_id'])

#特征工程
#def processDrops():
#data_titanic.drop(['Title_id','ticket_zs','Parch','SibSp','Fare','cabin_Number','PassengerId','Age','Fare','Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Names','Title'],axis=1,inplace=True)
    numeric=data_titanic.loc[:,['cabinnumber_scaler','fare_scaler','Parch_scaler','SibSp_scaler','age_scale','Title_id_scaled','Pclass_scaler']]
        '''
    
    #相关系数
    data_corr=data_titanic.drop(['Survived'],axis=1).corr(method='spearman')
    mask=np.ones(data_corr.columns.size)-np.eye(data_corr.columns.size)
    data_corr=data_corr*mask
        
    drops=[]
    for col in data_corr.columns.values:
        corr=data_corr[abs(data_corr[col])>0.98].index
        drops=np.union1d(drops,corr)
    data_titanic.drop(drops,axis=1,inplace=True)
    
    input_df=data_titanic[:input_df.shape[0]]
    submit_df=data_titanic[input_df.shape[0]:]

   # if pca:
     #   input_df,submit_df=reduceAndcluster(input_df,submit_df)
   # else:
    #    submit_df.drop('Survived',axis=1,inplace=True)
    if balanced:
        randomindex=rd.sample(list(input_df[input_df.Survived==0].index.values),input_df[input_df['Survived']==1].shape[0])
        input_df=pd.concat([input_df.ix[randomindex],input_df[input_df['Survived']==1]])
        input_df.sort(inplace=True)
    return input_df,submit_df
'''
def reduceAndcluster(input_df,submit_df,cluster=3):
    data_titanic=pd.concat([input_df,submit_df])
    data_titanic.reset_index(inplace=True)
    data_titanic.drop('index',axis=1,inplace=True)
    data_titanic=data_titanic.reindex_axis(input_df.columns,axis=1)
    survivedseries=pd.Series(data_titanic['Survived'],name='Survived')
    data_titanic.drop(['cabinnumber_scaler','age_scale','Pclass_scaler'],axis=1,inplace=True)
    X=data_titanic.values[:,1::]
    y=data_titanic.values[:,0]
   # print(X[0:5])
 #   data_titanic.drop(['age_scale/fare_scaler','age_scale/cabinnumber_scaler'],axis=1,inplace=True)
#    data_titanic.drop(['SibSp_scaler+Title_id_scaled','SibSp_scaler/Title_id_scaled'],axis=1,inplace=True)
    data_titanic.drop(['Title_id_scaled'],axis=1,inplace=True)
    data_titanic.describe()
    varence_pct=.99

    pca=PCA(n_components=varence_pct)
    x_transform=pca.fit_transform(X)
    pcadataframe=pd.DataFrame(x_transform)
    pca.explained_variance_ratio_
    kmeans=KMeans(n_clusters=3,random_state=np.random.RandomState(4), init='random')
   
    trainClusterIds = kmeans.fit(x_transform[:input_df.shape[0]])
    testClusterIds = kmeans.predict(x_transform[input_df.shape[0]:])
    clusterIds = np.concatenate([trainClusterIds, testClusterIds])
    print ("all clusterIds shape: ", clusterIds.shape)
    clusterIdSeries = pd.Series(clusterIds, name='ClusterId')
    df = pd.concat([survivedseries, clusterIdSeries, pcadataframe], axis=1) 
    t=trainClusterIds.cluster_centers_
    for i in range(x_transform.shape[0]):
        for j in range(x_transform.shape[1]):            
            plt.scatter(x=x_transform[i,j],y=clusterIdSeries[i], marker = "x")
    plt.show()
'''
#one-hot后进行pca降维

#建模
data=data_titanic.loc[:,['Survived','Pclass','Age','SibSp','Parch','Fare','cabin_Letter','Embarked_fac','txrs','Sex_male','Sex_female','Title_id']]
input_df=data[:data_train.shape[0]]
submit_df=data[data_train.shape[0]:]
featurelist=input_df.columns.values[1::]
train_X=input_df.values[:,1::]
train_y=input_df.values[:,0]
test_X=submit_df.values[:,1::]

forest=RandomForestClassifier(n_estimators=300,min_samples_leaf=4,oob_score=True,n_jobs=-1,class_weight={0:0.75,1:0.25})

forest.fit(train_X,train_y)
feature_importance=forest.feature_importances_
model_results = model_rf.predict(X_test)
submission=pd.DataFrame()
submission['PassengerId']=data_test.PassengerId
submission['result']=model_results
submission.set_index(['PassengerId'],inplace=True)
submission.to_csv('C:/Users/hp/Desktop/在家学习/tatanic/titanic_submission_3.csv')
'''
feature_importance = 100.0 * (feature_importance / feature_importance.max())
fi_threshold = 18    
important_idx = np.where(feature_importance > fi_threshold)[0]
important_features = featurelist[important_idx]
sorted_idx = np.argsort(feature_importance[important_idx])[::-1]

pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.title('Feature Importance')
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]],color='r',align='center')
plt.yticks(pos, important_features[sorted_idx[::-1]])
plt.xlabel('Relative Importance')
plt.draw()
plt.show()

train_X=train_X[:,important_idx][:,sorted_idx]
submit_df=submit_df.iloc[:,important_idx].iloc[:,sorted_idx]
sqrtfeat=int(np.sqrt(train_X.shape[1]))
minsampsplit = int(train_X.shape[0]*0.015)
params_score = { "n_estimators"      : 10000,
                     "max_features"      : sqrtfeat,
                     "min_samples_split" : minsampsplit }
params = params_score
forest = RandomForestClassifier(n_jobs=-1, oob_score=True, **params)
    # sort to ensure the passenger IDs are in the correct sequence
output = submission[submission[:,0].argsort()]
    
    # write results to a file
name = "rfc" + str(int(time.time())) + ".csv"
print ("Generating results file:", name)
predictions_file = open("./" + name, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(output)
'''


















