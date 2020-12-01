#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[4]:


## REGRESSION PACKAGES ##
# Ridge
from sklearn.linear_model import RidgeCV
# Lasso
from sklearn.linear_model import LassoCV
# Random Forest
from sklearn.ensemble import RandomForestRegressor
# Elastic Net
from sklearn.linear_model import ElasticNetCV
# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
# Support Vector Regressor
from sklearn.svm import SVR
# kNN Regressor
from sklearn.neighbors import KNeighborsRegressor


# In[5]:


df = pd.read_csv('task3df.csv')
df.drop(columns = ['Unnamed: 0'], inplace=True)

X = df[['fm.mean', 'Wind.mean', 'LATITUDE', 'LONGITUDE', 'DISCOVERY_DOY',
       'FIRE_YEAR', 
       'STAT_CAUSE_DESCR_Arson', 'STAT_CAUSE_DESCR_Campfire',
       'STAT_CAUSE_DESCR_Children', 'STAT_CAUSE_DESCR_Debris Burning',
       'STAT_CAUSE_DESCR_Equipment Use', 'STAT_CAUSE_DESCR_Fireworks',
       'STAT_CAUSE_DESCR_Lightning', 'STAT_CAUSE_DESCR_Miscellaneous',
       'STAT_CAUSE_DESCR_Powerline', 'STAT_CAUSE_DESCR_Railroad',
       'STAT_CAUSE_DESCR_Smoking', 'STAT_CAUSE_DESCR_Structure',
       'FUEL_MODEL_Chaparral (6 feet)', 'FUEL_MODEL_Closed Timber Litter',
       'FUEL_MODEL_Dormant Brush, Hardwood Slash',
       'FUEL_MODEL_Hardwood Litter', 'FUEL_MODEL_Heavy Logging Slash',
       'FUEL_MODEL_Light Logging Slash', 'FUEL_MODEL_Medium Logging Slash',
       'FUEL_MODEL_Short Grass (1 foot)', 'FUEL_MODEL_Southern Rough',
       'FUEL_MODEL_Tall Grass (2.5 feet)','FUEL_MODEL_Brush (2 feet)',
       'FUEL_MODEL_Timber (Grass and Understory)',
       'FUEL_MODEL_Timber (Litter and Understory)', 'GROWTH_POTENTIAL_Extreme',
       'GROWTH_POTENTIAL_High', 'GROWTH_POTENTIAL_Low',
       'GROWTH_POTENTIAL_Medium', 'TERRAIN_Extreme', 'TERRAIN_High',
       'TERRAIN_Low', 'TERRAIN_Medium']]
regions = [
     'eco1_10  NORTH AMERICAN DESERTS',
 'eco1_11  MEDITERRANEAN CALIFORNIA',
 'eco1_12  SOUTHERN SEMIARID HIGHLANDS',
 'eco1_13  TEMPERATE SIERRAS',
 'eco1_15  TROPICAL WET FORESTS',
 'eco1_5  NORTHERN FORESTS',
 'eco1_6  NORTHWESTERN FORESTED MOUNTAINS',
 'eco1_7  MARINE WEST COAST FOREST',
 'eco1_8  EASTERN TEMPERATE FORESTS',
 'eco1_9  GREAT PLAINS',
]

y0 = []
for i in df.columns:
    if i not in X:
        if i not in regions:
            y0.append(i)


# In[6]:


X.head()


# In[7]:


for col in X.columns[6:]:
    print(col, ':', sum(X[col]))


# In[8]:


X.shape


# In[ ]:





# In[9]:


# y0
yAll = df[y0]

yIdx = [0,1,3,6,7] 

yDescr = [
    'cont_time',
    'PROJECTED_FINAL_IM_COST', 
    'TOTAL_PERSONNEL',
    'FATALITIES',
    'FIRE_SIZE'
]


# In[9]:




for reg in regions:
    print('----------------------------------')
    print(reg)
    print('----------------------------------')
    idx = df[(df[reg] == 1)].index
    X_reg = X.loc[idx]
    scaler = StandardScaler()
    X_reg.loc[:,X_reg.columns[0:6]] = scaler.fit_transform(X_reg[(X_reg.columns[0:6])])
    
    
    cnt = 0
    for yi in yIdx:
        print(yDescr[cnt])
        cnt += 1
        
        
        y = yAll.iloc[:,yi]
        # log transform
        y = np.log(y+1)
        
        
        
                
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
        
        # ridge
        regRidge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
        print('ridge: ', regRidge.score(X_test, y_test))
        
        # lasso
        regLasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train.ravel())
        print('lasso: ', regLasso.score(X_test, y_test))
        
        # forest
        regForest = RandomForestRegressor(random_state=0, max_features='sqrt', oob_score=True, min_samples_leaf=0.05).fit(X_train, y_train.ravel())
        print('random forest: ', regForest.score(X_test, y_test))
        
        # elastic net
        regElastic = ElasticNetCV(cv=5, random_state=0, selection='random', l1_ratio=[.1, .5, .7, .9, .95, .99, 1]).fit(X_train, y_train.ravel())
        print('elastic net: ', regElastic.score(X_test, y_test))
        
        # decision tree regression
        regTree = DecisionTreeRegressor(max_features='sqrt', min_samples_leaf=0.05, random_state=0).fit(X_train, y_train.ravel())
        print('decision tree regression: ', regTree.score(X_test, y_test))
        
        # kNN
        regkNN = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train.ravel())
        print('kNN regression: ', regkNN.score(X_test, y_test))
        
        


# In[10]:



# params = {'n_neighbors':[2,3,4,5,6,7,8,9,10]}

# knn = KNeighborsRegressor()

# model = GridSearchCV(knn, params, cv=5)

# # model.best_params_



# dfResults = pd.DataFrame(columns = ['region', 'y', 'yType', 'r^2', 'bestK'])
# for reg in regions:
# #     print('----------------------------------')
# #     print(reg)
# #     print('----------------------------------')
#     idx = df[(df[reg] == 1)].index
#     X_reg = X.loc[idx]
#     scaler = StandardScaler()
#     X_reg.loc[:,X_reg.columns[0:6]] = scaler.fit_transform(X_reg[(X_reg.columns[0:6])])
    
#     cnt = 0
#     for yi in yIdx:
# #         print(yDescr[cnt])
                
#         y = yAll.iloc[:,yi]
        
#         # no transform
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
               
#         # kNN
#         regkNN = model.fit(X_train, y_train.ravel())
# #         print('untransformed kNN regression: ', regkNN.score(X_test, y_test))
#         dfResults = dfResults.append({'region': reg, 'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)
        
#         # log transform
#         ylog = np.log(y+1)
        
#         X_train, X_test, y_train, y_test = train_test_split(X, ylog, test_size=0.2, random_state=12)
               
#         # kNN
#         regkNN = model.fit(X_train, y_train.ravel())
# #         print('log transform kNN regression: ', regkNN.score(X_test, y_test))
#         dfResults = dfResults.append({'region': reg, 'y': yDescr[cnt], 'yType': 'log', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)
      
        
#         # sqrt transform
#         ysqrt = np.sqrt(y)
        
#         X_train, X_test, y_train, y_test = train_test_split(X, ysqrt, test_size=0.2, random_state=12)
               
#         # kNN
#         regkNN = model.fit(X_train, y_train.ravel())
# #         print('sqrt transform kNN regression: ', regkNN.score(X_test, y_test))
#         dfResults = dfResults.append({'region': reg, 'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)

#         cnt += 1
        


# In[15]:


def replaceK(item):
    return item['n_neighbors']

dfResults['bestK'] = dfResults['bestK'].apply(replaceK)

dfResults


# In[12]:


dfResultsBest = pd.DataFrame(columns = ['region', 'y', 'yType', 'r^2', 'bestK'])
for reg in regions:

    cnt = 0
    for yi in yDescr:
        idx = dfResults[(dfResults.region == reg)][(dfResults.y == yi)]['r^2'].idxmax()
        
        yTypeWinner = dfResults['yType'].iloc[idx]
        val = dfResults['r^2'].loc[idx]
        bK = dfResults['bestK'].loc[idx]
        
        dfResultsBest = dfResultsBest.append({'region': reg, 'y': yi, 'yType': yTypeWinner, 'r^2': val, 'bestK': bK}, ignore_index=True)
        cnt += 1


# In[37]:


# dfResultsBest.to_csv('task3_fullFactorial.csv')


# In[38]:


dfResultsType = pd.DataFrame(columns = ['y', 'modeType', 'modeK', 'r^2'])

for yi in yDescr:
    val = dfResultsBest[(dfResultsBest.y == yi)]['yType'].mode()[0]
    kval = dfResultsBest[(dfResultsBest.y == yi)]['bestK'].mode()[0]
    r2 = dfResultsBest[(dfResultsBest.y == yi)]['r^2'].mode()[0]

    dfResultsType = dfResultsType.append({'y': yi, 'modeType': val, 'modeK': kval, 'r^2':r2}, ignore_index=True)
    
# dfResultsType.to_csv('task3_resultsByType.csv')


# In[20]:


dfResultsType = pd.read_csv('task3_resultsByType.csv')


# In[13]:


# # verify ecoregions are okay being dropped 
params = {'n_neighbors':[2,3,4,5,6,7,8,9,10]}

knn = KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)

dfResults2 = pd.DataFrame(columns = ['y', 'yType', 'r^2', 'bestK'])


scaler = StandardScaler()
X.loc[:,X.columns[0:6]] = scaler.fit_transform(X[(X.columns[0:6])])

cnt = 0
for yi in yIdx:
#         print(yDescr[cnt])

    y = yAll.iloc[:,yi]

    # no transform
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # kNN
    regkNN = model.fit(X_train, y_train.ravel())
#         print('untransformed kNN regression: ', regkNN.score(X_test, y_test))
    dfResults2 = dfResults2.append({'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)

    # log transform
    ylog = np.log(y+1)

    X_train, X_test, y_train, y_test = train_test_split(X, ylog, test_size=0.2, random_state=12)

    # kNN
    regkNN = model.fit(X_train, y_train.ravel())
#         print('log transform kNN regression: ', regkNN.score(X_test, y_test))
    dfResults2 = dfResults2.append({'y': yDescr[cnt], 'yType': 'log', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)


    # sqrt transform
    ysqrt = np.sqrt(y)

    X_train, X_test, y_train, y_test = train_test_split(X, ysqrt, test_size=0.2, random_state=12)

    # kNN
    regkNN = model.fit(X_train, y_train.ravel())
#         print('sqrt transform kNN regression: ', regkNN.score(X_test, y_test))
    dfResults2 = dfResults2.append({'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)

    cnt += 1


# In[16]:


dfResults2['bestK'] = dfResults2['bestK'].apply(replaceK)
dfResults2


# In[17]:


dfResultsBest2 = pd.DataFrame(columns = ['y', 'yType', 'r^2', 'bestK'])

cnt = 0
for yi in yDescr:
    idx = dfResults2[(dfResults2.y == yi)]['r^2'].idxmax()

    yTypeWinner = dfResults2['yType'].iloc[idx]
    val = dfResults2['r^2'].loc[idx]
    bK = dfResults2['bestK'].loc[idx]

    dfResultsBest2 = dfResultsBest2.append({'y': yi, 'yType': yTypeWinner, 'r^2': val, 'bestK': bK}, ignore_index=True)
    cnt += 1
# dfResultsBest2.to_csv('task3_removeRegion_bestResults.csv')


# In[22]:


dfResultsBest2
dfResultsBest2.to_csv('task3_removeRegion_bestResults.csv')


# In[21]:


dfResultsType


# In[10]:


### start looking at fatalities


# In[23]:


params = {'n_neighbors':[2,3,5,10]}

knn = KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)

dfResults3 = pd.DataFrame(columns = ['model', 'y', 'yType', 'r^2', 'bestK'])


scaler = StandardScaler()
X.loc[:,X.columns[0:6]] = scaler.fit_transform(X[(X.columns[0:6])])

cnt = 0
for yi in yIdx:
#         print(yDescr[cnt])

    y = yAll.iloc[:,yi]

    ##############################
    # no transform
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # kNN
    regkNN = model.fit(X_train, y_train.ravel())
#         print('untransformed kNN regression: ', regkNN.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'knn', 'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)


    # ridge
    regRidge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
#     print('ridge: ', regRidge.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'ridge', 'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regRidge.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # lasso
    regLasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train.ravel())
#     print('lasso: ', regLasso.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'lasso', 'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regLasso.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)


    # forest
    regForest = RandomForestRegressor(random_state=0, max_features='sqrt', oob_score=True, min_samples_leaf=0.05).fit(X_train, y_train.ravel())
#     print('random forest: ', regForest.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'forest', 'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regForest.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # elastic net
    regElastic = ElasticNetCV(cv=5, random_state=0, selection='random', l1_ratio=[.1, .5, .7, .9, .95, .99, 1]).fit(X_train, y_train.ravel())
#     print('elastic net: ', regElastic.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'elastic', 'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regElastic.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # decision tree regression
    regTree = DecisionTreeRegressor(max_features='sqrt', min_samples_leaf=0.05, random_state=0).fit(X_train, y_train.ravel())
#     print('decision tree regression: ', regTree.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'tree', 'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regTree.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    
  #################################  
    # log transform
    ylog = np.log(y+1)

    X_train, X_test, y_train, y_test = train_test_split(X, ylog, test_size=0.2, random_state=12)

    # kNN
    regkNN = model.fit(X_train, y_train.ravel())
#         print('untransformed kNN regression: ', regkNN.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'knn', 'y': yDescr[cnt], 'yType': 'log', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)


    # ridge
    regRidge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
#     print('ridge: ', regRidge.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'ridge', 'y': yDescr[cnt], 'yType': 'log', 'r^2': regRidge.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # lasso
    regLasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train.ravel())
#     print('lasso: ', regLasso.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'lasso', 'y': yDescr[cnt], 'yType': 'log', 'r^2': regLasso.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)


    # forest
    regForest = RandomForestRegressor(random_state=0, max_features='sqrt', oob_score=True, min_samples_leaf=0.05).fit(X_train, y_train.ravel())
#     print('random forest: ', regForest.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'forest', 'y': yDescr[cnt], 'yType': 'log', 'r^2': regForest.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # elastic net
    regElastic = ElasticNetCV(cv=5, random_state=0, selection='random', l1_ratio=[.1, .5, .7, .9, .95, .99, 1]).fit(X_train, y_train.ravel())
#     print('elastic net: ', regElastic.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'elastic', 'y': yDescr[cnt], 'yType': 'log', 'r^2': regElastic.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # decision tree regression
    regTree = DecisionTreeRegressor(max_features='sqrt', min_samples_leaf=0.05, random_state=0).fit(X_train, y_train.ravel())
#     print('decision tree regression: ', regTree.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'tree', 'y': yDescr[cnt], 'yType': 'log', 'r^2': regTree.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    
    
   ################################## 
    # sqrt transform
    ysqrt = np.sqrt(y)

    X_train, X_test, y_train, y_test = train_test_split(X, ysqrt, test_size=0.2, random_state=12)
    
    # kNN
    regkNN = model.fit(X_train, y_train.ravel())
#         print('untransformed kNN regression: ', regkNN.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'knn', 'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)


    # ridge
    regRidge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
#     print('ridge: ', regRidge.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'ridge', 'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regRidge.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # lasso
    regLasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train.ravel())
#     print('lasso: ', regLasso.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'lasso', 'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regLasso.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)


    # forest
    regForest = RandomForestRegressor(random_state=0, max_features='sqrt', oob_score=True, min_samples_leaf=0.05).fit(X_train, y_train.ravel())
#     print('random forest: ', regForest.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'forest', 'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regForest.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # elastic net
    regElastic = ElasticNetCV(cv=5, random_state=0, selection='random', l1_ratio=[.1, .5, .7, .9, .95, .99, 1]).fit(X_train, y_train.ravel())
#     print('elastic net: ', regElastic.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'elastic', 'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regElastic.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    # decision tree regression
    regTree = DecisionTreeRegressor(max_features='sqrt', min_samples_leaf=0.05, random_state=0).fit(X_train, y_train.ravel())
#     print('decision tree regression: ', regTree.score(X_test, y_test))
    dfResults3 = dfResults3.append({'model': 'tree', 'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regTree.score(X_test, y_test), 'bestK': 'NA'}, ignore_index=True)

    cnt += 1


# In[29]:


dfResults3.sort_values(by='r^2', ascending=False).head(20)


# In[33]:


df[(df['eco1_11  MEDITERRANEAN CALIFORNIA'] == 1)].columns
'fm.mean', 'Wind.mean', 'LATITUDE', 'LONGITUDE', 'DISCOVERY_DOY',
       'FIRE_YEAR', 'FUEL_MODEL_Chaparral (6 feet)', 'FUEL_MODEL_Closed Timber Litter',
       'FUEL_MODEL_Dormant Brush, Hardwood Slash',
       'FUEL_MODEL_Hardwood Litter', 'FUEL_MODEL_Heavy Logging Slash',
       'FUEL_MODEL_Light Logging Slash', 'FUEL_MODEL_Medium Logging Slash',
       'FUEL_MODEL_Short Grass (1 foot)', 'FUEL_MODEL_Southern Rough',
       'FUEL_MODEL_Tall Grass (2.5 feet)',
       'FUEL_MODEL_Timber (Grass and Understory)',
       'FUEL_MODEL_Timber (Litter and Understory)', 'GROWTH_POTENTIAL_Extreme',
       'GROWTH_POTENTIAL_High', 'GROWTH_POTENTIAL_Low',
       'GROWTH_POTENTIAL_Medium', 'TERRAIN_Extreme', 'TERRAIN_High',


# In[94]:


X_cause = df[(df['eco1_11  MEDITERRANEAN CALIFORNIA'] == 1)][['LATITUDE', 'LONGITUDE', 'DISCOVERY_DOY',
       'FIRE_YEAR', 'cont_time', 'PROJECTED_FINAL_IM_COST', 'STR_DAMAGED', 'FIRE_SIZE']]
y_cali1 = df[(df['eco1_11  MEDITERRANEAN CALIFORNIA'] == 1)]['STAT_CAUSE_DESCR_Campfire']
y_cali2 = df[(df['eco1_11  MEDITERRANEAN CALIFORNIA'] == 1)]['STAT_CAUSE_DESCR_Miscellaneous']
y_cali3 = df[(df['eco1_11  MEDITERRANEAN CALIFORNIA'] == 1)]['STAT_CAUSE_DESCR_Arson']

yCali = [y_cali1, y_cali2, y_cali3]


# In[95]:


from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_cause.loc[:,X_cause.columns[0:10]]= scaler.fit_transform(X_cause[(X_cause.columns[0:10])])

caliCause = ['campfire', 'misc', 'arson']

i = 0
for yC in yCali:
    print(caliCause[i])
    X_train, X_test, y_train, y_test = train_test_split(X_cause, yC, test_size=0.2, random_state=12)


    grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
    logreg=LogisticRegression()
    logreg_cv=GridSearchCV(logreg,grid,cv=5)
    logreg_cv.fit(X_train,y_train)

    print("tuned hpyerparameters:(best parameters) ",logreg_cv.best_params_)
    print("accuracy :",logreg_cv.best_score_)
    i += 1


# In[96]:


bestC = [0.1, 0.1, 0.001]

coefs = []


i = 0
for yC in yCali:
    print(caliCause[i])
    X_train, X_test, y_train, y_test = train_test_split(X_cause, yC, test_size=0.2, random_state=12)

    logreg=LogisticRegression(C = bestC[i], penalty='l2').fit(X_train, y_train)

    print("accuracy :", logreg.score(X_test, y_test))
    coefs.append(logreg.coef_)
    i += 1


# In[97]:


# for i in [0,1,2]:
#     print(r'\verb!', caliCause[i], '! =')
#     for j in np.arange(0,len(X_cause.columns)):
#         print(round(coefs[0][0][j],3), r'\verb!', X_cause.columns[j], '! +')


# In[98]:


import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix,  
                           accuracy_score)


# In[99]:


## verify and get better output in statsmodels

i = 0
for yC in yCali:
    print(caliCause[i])
    X_train, X_test, y_train, y_test = train_test_split(X_cause, yC, test_size=0.2, random_state=12)
    X_train = sm.add_constant(X_train, prepend=False)
    X_test = sm.add_constant(X_test, prepend=False)
#     logreg=LogisticRegression(C = bestC[i], penalty='l2').fit(X_train, y_train)
    log_reg = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
#     log_reg = sm.Logit(y_train, X_train).fit() 

    yhat = log_reg.predict(X_test) 
    prediction = list(map(round, yhat))
    
    print('Test accuracy = ', accuracy_score(y_test, prediction))
    
    print(log_reg.summary())
    i += 1


# In[ ]:




