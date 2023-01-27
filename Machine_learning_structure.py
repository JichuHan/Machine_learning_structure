#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


# In[2]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


# In[3]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.metrics import classification_report,roc_curve,auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
scaler=preprocessing.StandardScaler()
from sklearn.metrics import plot_roc_curve, accuracy_score, recall_score


# # 一. 数据的读取以及地址的命名

# In[4]:


path='/Users/jason/Desktop/ML-Project/'
Data=pd.read_csv(path+'Data/ML-data.csv')
#out_path是下载文件的存储地址
out_path=path+'ML-Graph'


# # 二. 描述性统计

# ## （一）根据画图初步确定的分类标准

# In[16]:


Data


# In[55]:


Groups=[['x_0', 'x_1', 'x_2'],['x_3', 'x_4', 'x_5'],['x_6'],['x_7'],['x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15',
       'x_16', 'x_17'],['x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24'],['x_25', 'x_26'],['x_27'],['x_28'],['x_29'],['x_30'],['x_31', 'x_32'],['x_33',
       'x_34', 'x_35'],['x_36', 'x_37', 'x_38', 'x_39', 'x_40'],['x_41', 'x_42',
       'x_43', 'x_44', 'x_45', 'x_46', 'x_47'],['x_48', 'x_49', 'x_50', 'x_51',
       'x_52', 'x_53', 'x_54', 'x_55'],['x_56', 'x_57', 'x_58'],['x_59', 'x_60',
       'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68'],['x_69',
       'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76'],['x_77', 'x_78'],['x_79', 'x_80', 'x_81'],['x_82', 'x_83'],['x_84', 'x_85'],['x_86'],['x_87'],
        ['x_88', 'x_89'],['x_90', 'x_91', 'x_92'],['x_93'],['x_94', 'x_95']]


# In[17]:


Data['code'].value_counts()


# ## （二）定义画图函数

# 1. Draw_line是直接画图的函数，依次取出Groups中的每一个类别，画出每一只股票的曲线。因此，对于每一个Groups中的变量都进行了画图
# 2. Draw_hist是绘制直方图的函数，原理和Draw_line一样
# 3. Corr_test针对每一个时间序列进行时间序列分析。我进行了Ljung-Box检验（一共5阶滞后），并进行了ADF检验

# In[18]:


codes=Data['code'].unique()
def Draw_line(group_number):
    plt.figure(figsize=(50,100))
    for i in range(0,len(codes)):
        plt.subplot(30,5,1+i)
        Data1=np.array(Data[Data['code']==codes[i]][Groups[group_number]])
        plt.plot(Data1)
        plt.title(str(codes[i]))
    plt.savefig(out_path+'/ML-Line/'+str(group_number)+'line.jpg')
    print('Line'+str(group_number)+' done')
    return 0

def Draw_hist(group_number):
    plt.figure(figsize=(50,100))
    for i in range(0,len(codes)):
        plt.subplot(30,5,1+i)
        Data1=np.array(Data[Data['code']==codes[i]][Groups[group_number]].replace(float('inf'),0).replace(float('-inf'),0))
        plt.hist(Data1)
        plt.title(str(codes[i]))
    plt.savefig(out_path+'/ML-Hist/'+str(group_number)+'hist.jpg')
    print('Hist'+str(group_number)+' done')
    return 0

def Corr_test(group_number):
    Aver_total=[]
    for i in Groups[group_number]:
        print(i)
        Data1=Data[['code',i]]
        result_total=[]
        for j in range(0,len(codes)):
            print(j)
            result=[]
            D=Data1[Data1['code']==j].iloc[:,1]
            D=D.fillna(method='bfill').dropna().replace(float('inf'),0).replace(float('-inf'),0)
            result=result+list(acorr_ljungbox(D, lags=5)[1])
            result.append(adfuller(D)[1])
            result_total.append(result)
        Average=pd.DataFrame(result_total).describe().iloc[1,:]
        Aver_total.append(list(Average))
    DL_test=pd.DataFrame(Aver_total)
    DL_test.columns=['Lj-L1','Lj-L2','Lj-L3','Lj-L4','Lj-L5','ADF']
    DL_test.index=Groups[group_number]
    print('Corr'+str(group_number)+' done')
    return DL_test


# In[ ]:


Corr=[]
for i in range(0,len(Groups)):
    if i==23:
        continue
    else:
        Draw_line(i)
        Draw_hist(i)
        Corr.append(Corr_test(i))


# 4. 绘制所有变量总体情况下的分布图（看整体分布，适不适合降低维度）

# In[ ]:


plt.figure(figsize=(50,100))
for i in range(3,len(Data.columns)):
    name=Data.columns[i]
    if name=='x_90' or name=='x_91' or name=='x_92':
        continue
    else:
        plt.subplot(20,5,1+i)
        plt.hist(Data.iloc[:,i])
        plt.title(name)


# # 三. 数据预处理

# 1. 只针对有数据的部分进行标准化，没有数据的部分在标准化之后进行填补（只用bfill和dropna，避免信息超前于市场）

# In[80]:


Data1=Data.copy(deep=True)


# In[81]:


def Scal(Group_name):
    scaler_param=scaler.fit(pd.DataFrame(Data[Group_name].dropna().replace(float('inf'),0).replace(float('-inf'),0)))
    a=scaler.fit_transform(pd.DataFrame(Data[Group_name].dropna()).replace(float('inf'),0).replace(float('-inf'),0),scaler_param)
    Index=Data[Group_name][-np.isnan(Data[Group_name])].index
    for i in range(0,len(a)):
        Data1[Group_name][Index[i]]=a[i]


# In[ ]:


for i in range(3,len(Data1.columns)):
    Scal(Data1.columns[i])
    print(Data1.columns[i]+'Done')


# 2. 根据对数据的理解对数据进行调整

# （1）针对90-92之间的变量（有inf，-inf的极端情况）进行处理

# In[ ]:


def f(x):
    if x<-1000:
        return -2
    elif x<-10:
        return -1
    elif x>1000:
        return 2
    elif x>10:
        return 1
    else:
        return 0
    return x
Data1['x_90']=Data['x_90'].apply(f)
Data1['x_91']=Data['x_91'].apply(f)
Data1['x_92']=Data['x_92'].apply(f)


# （2）删除x_93，这个变量全部都是0

# In[ ]:


Data1=Data1.drop(axis=1, columns='x_93',inplace=False)


# 3. 填补空白

# In[ ]:


Data1=Data1.fillna(method='bfill')
Data1=Data1.dropna()


# 4. 生成个体哑元变量

# In[ ]:


Data1=pd.concat([Data1,pd.get_dummies(Data1.code,prefix=None)],axis=1)


# 5. 将数据导入excel文件，转移至SPSS进行因子分析（在文件夹中已经有处理好的excel文件Data_norm.xlsx）

# In[ ]:


#Data1.iloc[:,1:98].to_excel('/Users/jason/downloads/Data_norm.xlsx')


# # 四. 降维

# 1. 相关性分析

# In[ ]:


Data3=Data1.iloc[:,1:98]
C=Data3.corr()


# In[ ]:


#这个删掉了93号变量
Group_new=[['x_0', 'x_1', 'x_2'],['x_3', 'x_4', 'x_5'],['x_6'],['x_7'],['x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15',
       'x_16', 'x_17'],['x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24'],['x_25', 'x_26'],['x_27'],['x_28'],['x_29'],['x_30'],['x_31', 'x_32'],['x_33',
       'x_34', 'x_35'],['x_36', 'x_37', 'x_38', 'x_39', 'x_40'],['x_41', 'x_42',
       'x_43', 'x_44', 'x_45', 'x_46', 'x_47'],['x_48', 'x_49', 'x_50', 'x_51',
       'x_52', 'x_53', 'x_54', 'x_55'],['x_56', 'x_57', 'x_58'],['x_59', 'x_60',
       'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68'],['x_69',
       'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76'],['x_77', 'x_78'],['x_79', 'x_80', 'x_81'],['x_82', 'x_83'],['x_84', 'x_85'],['x_86'],['x_87'],['x_88', 'x_89'],
           ['x_90', 'x_91', 'x_92'],['x_94', 'x_95']]
#这个是后期命名坐标用的
Group_name=['x_0~x_2','x_3~x_5','x_6','x_7','x_8~x_17','x_18~x_24','x_25~x_26','x_27','x_28','x_29','x_30'
            ,'x_31~x_32','x_33~x_35','x_36~x_40','x_41~x_47','x_48~x_55','x_56~x_58','x_59~x_68','x_69~x_76'
            ,'x_77~x_78','x_79~x_81','x_82~x_83','x_84~x_85','x_86','x_87','x_88~x_89','x_90~x_92','x_94~x_95']


# In[ ]:


Col=[]
for i in range(0,len(Group_new)):
    row=[]
    for j in range(0,len(Group_new)):
        Cor_tol=np.mean(C.loc[Group_new[i],Group_new[j]].describe().loc['mean',:])
        row.append(Cor_tol)
    Col.append(row)
Cor_group=pd.DataFrame(Col)


# In[ ]:


Cor_group.index=np.array(Group_name)
Cor_group.columns=np.array(Group_name)


# In[ ]:


plt.figure(figsize=(30,14))
sns.heatmap(Cor_group,annot=True)
plt.savefig(path+'Other-graph/Heatmap.jpg')


# 2. SPSS分析（过程在SPSS中进行）

# # 五. 机器学习

# 1. 读入进行过因子分析后的数据

# In[ ]:


Data_dim=pd.read_excel(path+'Data/Data_norm.xlsx')


# In[ ]:


Data_learn=pd.concat([Data1['date'],Data_dim.iloc[:,2:],Data1.iloc[:,98:]],axis=1)


# 2. 划分训练测试数据集

# In[ ]:


X=Data_learn.drop(columns='y')
Y=Data_learn['y']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.6, test_size=0.4, random_state=100)


# In[ ]:


Data_learn


# 3. 模型的选择

# In[ ]:


models=[
        ('LDA',LinearDiscriminantAnalysis()),          # LinearDiscriminantAnalysis
        ('LR',LogisticRegression(max_iter=1000)),      # Logistic Regression
        ('KNN',KNeighborsClassifier(n_neighbors=10)),  # K近邻算法
        ('DTC',DecisionTreeClassifier()),              # 决策树
        ('GNB',GaussianNB()),                          # 朴素贝叶斯
        ('BNB',BernoulliNB()),                         # 伯努利贝叶斯分类器
        ('RF',RandomForestClassifier()),               # 随机森林
        ('ADA',AdaBoostClassifier()),                  # AdaBoost
        ('XGB',GradientBoostingClassifier())           # 梯度提升
       ]
results=[]
names=[]
finalResults=[]
for name,model in models:
    model.fit(X_train, y_train)
    model_results=model.predict(X_test)
    
    y_test_proba=model.predict_proba(X_test)
    fpr,tpr,thresholds=roc_curve(y_test,y_test_proba[:,1])
    roc_auc=auc(fpr,tpr)
    results.append(roc_auc)
    names.append(name)
    finalResults.append((name,roc_auc))
    print(name+'done')
    
finalResults.sort(key=lambda k:k[1],reverse=True)
finalResults


# 4. 对随机森林模型进行参数调整

# In[ ]:


model = RandomForestClassifier(random_state=100,class_weight = 'balanced')
params = {'n_estimators':[1000],
          'min_samples_leaf':[20,50,100],
          'ccp_alpha':[0.1,0.3],
          'max_depth':[20,50],
          'max_features':[0.3,0.5],
          'criterion':["gini","entropy"]}
grid_search = GridSearchCV(estimator=model,param_grid=params,verbose=1,n_jobs=-1,scoring='recall')
grid_search.fit(X_train,y_train)


# In[ ]:


model_best = RandomForestClassifier(
    random_state=100,
    class_weight = 'balanced',max_depth=22.58
    ,max_features='sqrt',max_samples=0.69,ccp_alpha=0.8,
    n_estimators=1000)

model_best.fit(X_train,y_train)


# In[ ]:


plot_roc_curve(model_best,X_train,y_train)
plt.show()
y_train_pred = model_best.predict(X_train)
print("Train_accuracy: ", accuracy_score(y_train, y_train_pred))
print("Train_recall: ", recall_score(y_train, y_train_pred))


# In[ ]:


plot_roc_curve(model_best,X_test,y_test)
plt.show()
y_test_pred = model_best.predict(X_test)
print("Test_accuracy: ", accuracy_score(y_test, y_test_pred))
print("Test_recall: ", recall_score(y_test, y_test_pred))


# 5. 特征重要性排序

# In[ ]:


Feature_Importance = pd.DataFrame({'Features':X_train.columns,'Importance_coef':model_best.feature_importances_})
Feature_Importance.set_index('Features',inplace=True)
Feature_Importance.sort_values('Importance_coef',ascending=False,inplace=True)
Feature_Importance


# In[38]:



def Draw_line1(group_number):
    plt.figure(figsize=(50,100))
    for i in range(0,len(codes)):
        plt.subplot(30,5,1+i)
        Data1=np.array(Data[Data['code']==codes[i]][group_number])
        plt.plot(Data1)
        plt.legend(group_number)
        plt.title(str(codes[i]))
    plt.savefig(path+'Other-graph/line.jpg')
    print('Line'+str(group_number)+' done')
    return 0


# In[ ]:





# In[ ]:





# In[ ]:





# 1. 预测模型的构造（所有股票下一天收益与当日因子取值的关联）
# 2. 描述性统计
#    （1）因子统计
#    （2）因子绘制
# 3. 因子相关性统计
#    （1）因子相关性
#    （2）因子组合（线性、非线性）
# 4. 因子贡献
#     单个因子对于收益率的贡献+因子组合对收益率的贡献
#     目的：探究因子之间的共线性以及因子的可能组合
# 5. 模型训练
#     线性模型的意义
#     其他模型
#    

# In[24]:



因子筛选
因子表现，模型筛选
了解因子主要是因子表现方面的筛选。因子的线性表现。
树模型有节点，有筛选功能。样本内分布与样本外分布基本一致，不然会出现过拟合。
应用场景——社会规律，没有过多样本内外影响；但是股市变化非常快速。
划分样本的方法：
之后进行预测的样本：


# In[26]:


pd.read_csv('/Users/jason/Downloads/eod_yhzhou_alpha003.csv')


# In[ ]:




