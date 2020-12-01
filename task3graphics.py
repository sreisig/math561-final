#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[3]:


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


# In[4]:


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


# In[5]:


X.head()


# In[6]:


for col in X.columns[6:]:
    print(col, ':', sum(X[col]))


# In[7]:


X.shape


# In[ ]:





# In[8]:


# y0


# In[9]:


yAll = df[y0]

yIdx = [0,1,3,6,7] 

yDescr = [
    'cont_time',
    'PROJECTED_FINAL_IM_COST', 
    'TOTAL_PERSONNEL',
    'FATALITIES',
    'FIRE_SIZE'
]


# for reg in regions:
#     print('----------------------------------')
#     print(reg)
#     print('----------------------------------')
#     idx = df[(df[reg] == 1)].index
#     X_reg = X.loc[idx]
#     scaler = StandardScaler()
#     X_reg.loc[:,X_reg.columns[0:6]] = scaler.fit_transform(X_reg[(X_reg.columns[0:6])])
    
#     cnt = 0
#     for yi in yIdx:
#         print(yDescr[cnt])
#         cnt += 1
        
        
#         y = yAll.iloc[:,yi]
#         # log transform
#         y = np.log(y+1)
        
        
        
                
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
        
#         # ridge
#         regRidge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
#         print('ridge: ', regRidge.score(X_test, y_test))
        
#         # lasso
#         regLasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train.ravel())
#         print('lasso: ', regLasso.score(X_test, y_test))
        
#         # forest
#         regForest = RandomForestRegressor(random_state=0, max_features='sqrt', oob_score=True, min_samples_leaf=0.05).fit(X_train, y_train.ravel())
#         print('random forest: ', regForest.score(X_test, y_test))
        
#         # elastic net
#         regElastic = ElasticNetCV(cv=5, random_state=0, selection='random', l1_ratio=[.1, .5, .7, .9, .95, .99, 1]).fit(X_train, y_train.ravel())
#         print('elastic net: ', regElastic.score(X_test, y_test))
        
#         # decision tree regression
#         regTree = DecisionTreeRegressor(max_features='sqrt', min_samples_leaf=0.05, random_state=0).fit(X_train, y_train.ravel())
#         print('decision tree regression: ', regTree.score(X_test, y_test))
        
#         # kNN
#         regkNN = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train.ravel())
#         print('kNN regression: ', regkNN.score(X_test, y_test))
        
        


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
        


# In[11]:


def replaceK(item):
    return item['n_neighbors']

# dfResults['bestK'] = dfResults['bestK'].apply(replaceK)

# dfResults


# In[12]:


# dfResultsBest = pd.DataFrame(columns = ['region', 'y', 'yType', 'r^2', 'bestK'])
# for reg in regions:

#     cnt = 0
#     for yi in yDescr:
#         idx = dfResults[(dfResults.region == reg)][(dfResults.y == yi)]['r^2'].idxmax()
        
#         yTypeWinner = dfResults['yType'].iloc[idx]
#         val = dfResults['r^2'].loc[idx]
#         bK = dfResults['bestK'].loc[idx]
        
#         dfResultsBest = dfResultsBest.append({'region': reg, 'y': yi, 'yType': yTypeWinner, 'r^2': val, 'bestK': bK}, ignore_index=True)
#         cnt += 1


# In[13]:


# dfResultsBest.to_csv('task3_fullFactorial.csv')


# In[14]:


# dfResultsType = pd.DataFrame(columns = ['y', 'modeType', 'modeK', 'r^2'])

# for yi in yDescr:
#     val = dfResultsBest[(dfResultsBest.y == yi)]['yType'].mode()[0]
#     kval = dfResultsBest[(dfResultsBest.y == yi)]['bestK'].mode()[0]
#     r2 = dfResultsBest[(dfResultsBest.y == yi)]['r^2'].mode()[0]

#     dfResultsType = dfResultsType.append({'y': yi, 'modeType': val, 'modeK': kval, 'r^2':r2}, ignore_index=True)
    
dfResultsType = pd.read_csv('task3_resultsByType.csv')


# In[15]:


# # verify ecoregions are okay being dropped 
# params = {'n_neighbors':[2,3,4,5,6,7,8,9,10]}

# knn = KNeighborsRegressor()

# model = GridSearchCV(knn, params, cv=5)

# dfResults2 = pd.DataFrame(columns = ['y', 'yType', 'r^2', 'bestK'])


# scaler = StandardScaler()
# X_reg.loc[:,X_reg.columns[0:6]] = scaler.fit_transform(X_reg[(X_reg.columns[0:6])])

# cnt = 0
# for yi in yIdx:
# #         print(yDescr[cnt])

#     y = yAll.iloc[:,yi]

#     # no transform
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

#     # kNN
#     regkNN = model.fit(X_train, y_train.ravel())
# #         print('untransformed kNN regression: ', regkNN.score(X_test, y_test))
#     dfResults2 = dfResults2.append({'y': yDescr[cnt], 'yType': 'untransform', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)

#     # log transform
#     ylog = np.log(y+1)

#     X_train, X_test, y_train, y_test = train_test_split(X, ylog, test_size=0.2, random_state=12)

#     # kNN
#     regkNN = model.fit(X_train, y_train.ravel())
# #         print('log transform kNN regression: ', regkNN.score(X_test, y_test))
#     dfResults2 = dfResults2.append({'y': yDescr[cnt], 'yType': 'log', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)


#     # sqrt transform
#     ysqrt = np.sqrt(y)

#     X_train, X_test, y_train, y_test = train_test_split(X, ysqrt, test_size=0.2, random_state=12)

#     # kNN
#     regkNN = model.fit(X_train, y_train.ravel())
# #         print('sqrt transform kNN regression: ', regkNN.score(X_test, y_test))
#     dfResults2 = dfResults2.append({'y': yDescr[cnt], 'yType': 'sqrt', 'r^2': regkNN.score(X_test, y_test), 'bestK': regkNN.best_params_}, ignore_index=True)

#     cnt += 1


# In[16]:


# dfResults2['bestK'] = dfResults2['bestK'].apply(replaceK)
# dfResults2


# In[21]:


# dfResultsBest2 = pd.DataFrame(columns = ['y', 'yType', 'r^2', 'bestK'])

# cnt = 0
# for yi in yDescr:
#     idx = dfResults2[(dfResults2.y == yi)]['r^2'].idxmax()

#     yTypeWinner = dfResults2['yType'].iloc[idx]
#     val = dfResults2['r^2'].loc[idx]
#     bK = dfResults2['bestK'].loc[idx]

#     dfResultsBest2 = dfResultsBest2.append({'y': yi, 'yType': yTypeWinner, 'r^2': val, 'bestK': bK}, ignore_index=True)
#     cnt += 1


# In[22]:


dfResultsBest2 = pd.read_csv('task3_removeRegion_bestResults.csv')
dfResultsBest2


# In[23]:


dfResultsType


# In[24]:


## TODO
# histograms of categorical variables
# graphs of y distributions


# In[25]:


# !pip install mglearn


# In[26]:


# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt


# def plot_correlation(data):
#     '''
#     plot correlation's matrix to explore dependency between features 
#     '''
#     # init figure size
#     rcParams['figure.figsize'] = 15, 20
#     fig = plt.figure()
#     sns.heatmap(data.corr(), annot=True, fmt=".2f")
#     plt.show()
# #     fig.savefig('corr.png')

# plot_correlation(X_reg)


# In[27]:


# from scipy.stats import pearsonr
cause = []
causeCols = []
causeVals = []
fuel_model = []
fuelCols = []
fuelVals = []
growth_potential = []
growthCols = []
growthVals = []
terrain = []
terrCols = []
terrVals = []
for col in X.columns:
    category = col.split('_')[-1]
    if 'STAT_CAUSE' in col:
        cause.append(category)
        causeCols.append(col)
        causeVals.append(sum(X[col]))
    elif 'FUEL_MODEL' in col:
        fuel_model.append(category)
        fuelCols.append(col)
        fuelVals.append(sum(X[col]))
    elif 'GROWTH' in col:
        growth_potential.append(category)
        growthCols.append(col)
        growthVals.append(sum(X[col]))
    elif 'TERRAIN' in col:
        terrain.append(category)
        terrCols.append(col)
        terrVals.append(sum(X[col]))
        


# In[28]:


fig = go.Figure([go.Bar(x=terrain, y=terrVals, text=terrVals, textposition='auto')])
fig.update_layout(xaxis={'categoryorder':'total descending'}, title="Number of Fires by Terrain Type")
fig.show()


# In[29]:


fig = go.Figure([go.Bar(x=cause, y=causeVals, text=causeVals, textposition='auto')])
fig.update_layout(xaxis={'categoryorder':'total descending'}, title="Number of Fires by Category")
fig.show()


# In[30]:


fig = go.Figure([go.Bar(x=fuel_model, y=fuelVals, text=fuelVals, textposition='auto')])
fig.update_layout(xaxis={'categoryorder':'total descending'}, title="Number of Fires by Fuel Type")
fig.show()


# In[31]:


fig = go.Figure([go.Bar(x=growth_potential, y=growthVals, text=growthVals, textposition='auto')])
fig.update_layout(xaxis={'categoryorder':'total descending'}, title="Number of Fires by Growth Potential")
fig.show()


# In[32]:


y0


# In[33]:


cnt = 0
yTitles = [
    'Containment Time (days)', 
    'Projected Final Incident Management Cost ($)',
    'Total Personnel (#)',
    'Fatalities (#)',
    'Fire Size (Acres)'  
]

for yi in yIdx:
    y = yAll.iloc[:,yi]
    descr = ' (untransformed) '
    if cnt == 1:
        descr = ' (log transformed) '
        y = np.log(y+1)
    elif cnt == 2 or cnt == 3:
        y = np.sqrt(y)
        descr = ' (sqrt transformed) '
    
    fig = go.Figure(data=[go.Histogram(x=y)])
    fig.update_layout(title=yTitles[cnt]+descr)
    fig.show()
    cnt += 1


# In[82]:


# df[(df['FIRE_SIZE']<50000)][(df['FIRE_SIZE']>-1)]
tmp = df[(df['FATALITIES']>1)]['FATALITIES'].sort_values(ascending=True)
tmp1 = tmp[:int(42872)]
fig = go.Figure(data=[go.Histogram(x=tmp)])
fig.update_layout(title='Distribution of Fires with > 1 Fatality')
fig.show()


# In[96]:


task3df = pd.read_csv('t3dfPreDummy.csv')
task3df = task3df[['eco1', 'fm.mean', 'Wind.mean', 'FATALITIES', 'STAT_CAUSE_DESCR']].drop_duplicates()


# In[97]:


t3copy = task3df[(task3df['FATALITIES']>1)]
labels = list(t3copy['eco1'])
fatal = list(t3copy['FATALITIES'])
fig = go.Figure(data=[
    go.Bar(name='Fatalities', x=labels, y=fatal, text=fatal, textposition='auto')
])
# Change the bar mode
# fig.update_layout(barmode='group')
fig.update_layout(title='Fatalities by Eco-Region')
fig.show()


# In[98]:


task3df.columns


# In[100]:


t3copy = task3df[(task3df['eco1']=='11  MEDITERRANEAN CALIFORNIA')]
labels = list(t3copy['STAT_CAUSE_DESCR'])
fatal = list(t3copy['FATALITIES'])
fig = go.Figure(data=[
    go.Bar(name='Fatalities', x=labels, y=fatal, text=fatal, textposition='auto')
])
# Change the bar mode
# fig.update_layout(barmode='group')
fig.update_layout(xaxis={'categoryorder':'total descending'}, title='Fatalities by Cause within Mediterranean California')
fig.show()


# In[36]:


labels = list(task3df['eco1'])
fm = list(task3df['fm.mean'])
ws = list(task3df['Wind.mean'])

fig = go.Figure(data=[
    go.Bar(name='Fuel Moisture', x=labels, y=fm),
    go.Bar(name='Wind Speed', x=labels, y=ws)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title='Fuel Moisture and Wind Speed by Eco-Region')
fig.show()


# In[37]:


cause


# In[42]:


label = df['cont_time'].value_counts().index
value = df['cont_time'].value_counts()
fig = go.Figure([go.Bar(x=label, y=value)])
fig.update_layout(title="Containment Time (Days) (untransformed)")
fig.show()


# In[47]:


# label = [str(i) for i in df['FIRE_SIZE'].value_counts().index]
label = df['FIRE_SIZE'].value_counts().index
value = df['FIRE_SIZE'].value_counts()
fig = go.Figure([go.Bar(x=label, y=value)])
# fig.update_layout(title="Fire Size (Acres) (untransformed)")
fig.show()


# In[46]:


label


# In[ ]:




