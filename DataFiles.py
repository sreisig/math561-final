#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('fm_ws_monthly_ecn.csv')
df.head()


# In[3]:


df1 = pd.read_pickle('fires_eco.zip')


# In[4]:


# df1['STAT_CAUSE_DESCR'].unique()


# In[5]:


df2 = pd.read_csv('fires.csv', low_memory=False)
df2_2 = df2.dropna(subset=['ICS_209_INCIDENT_NUMBER'])
# sum(pd.isnull(df2['ICS_209_INCIDENT_NUMBER']))


# In[8]:


## add fuel moisture and windspeed to df1
fm_ws = pd.read_csv('l1_windspeed_fuelmoisture.csv')
df1['NA_L1CODE'] = df1['eco1'].str.split(' ').map(lambda x:x[0])


# In[9]:



# drop 2 and 3
df1 = df1[(df1.NA_L1CODE != '2')]
df1 = df1[(df1.NA_L1CODE != '3')]
df1['NA_L1CODE'].unique()


# In[10]:


conditions = [
#     (df1['NA_L1CODE'] == '2'), 
#     (df1['NA_L1CODE'] == '3'),
    (df1['NA_L1CODE'] == '5'),
    (df1['NA_L1CODE'] == '6'),
    (df1['NA_L1CODE'] == '7'),
    (df1['NA_L1CODE'] == '8'),
    (df1['NA_L1CODE'] == '9'),
    (df1['NA_L1CODE'] == '10'),
    (df1['NA_L1CODE'] == '11'),
    (df1['NA_L1CODE'] == '12'),
    (df1['NA_L1CODE'] == '13'),
    (df1['NA_L1CODE'] == '15')
    ]

# create a list of the values we want to assign for each condition
values1 = [
#     fm_ws[(fm_ws.NA_L1CODE == 2)]['fm.mean'].values[0],
#     fm_ws[(fm_ws.NA_L1CODE == 3)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 5)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 6)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 7)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 8)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 9)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 10)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 11)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 12)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 13)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 15)]['fm.mean'].values[0]
]

values2 = [
#     fm_ws[(fm_ws.NA_L1CODE == 2)]['fm.mean'].values[0],
#     fm_ws[(fm_ws.NA_L1CODE == 3)]['fm.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 5)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 6)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 7)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 8)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 9)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 10)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 11)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 12)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 13)]['Wind.mean'].values[0],
    fm_ws[(fm_ws.NA_L1CODE == 15)]['Wind.mean'].values[0]
]

# create a new column and use np.select to assign values to it using our lists as arguments
df1['fm.mean'] = np.select(conditions, values1)
df1['Wind.mean'] = np.select(conditions, values2)


# In[11]:


df1.head()


# In[12]:


files = [
    'historical_cleaned_ll-fod.csv', #3
    'ics209-plus-wf-complex_assocs_2014.csv', #4
    'ics209-plus-wf_complex_assocs_1999to2013.csv', #5
    'ics209-plus-wf_incidents_1999to2014.csv', #6
    'ics209-plus-wf_sitreps_1999to2014.csv', #7
    'ics209-plus_sitreps_1999to2014.csv', #8
    'ics209_sitreps_deleted_hist1_1999to2002.csv', #9
    'ics209_sitreps_deleted_hist2_2001to2013.csv', #10
    'legacy_cleaned_ll-fod.csv' #11
]


# In[13]:


## read in all files
i = 3
for f in files:
    dfStr = 'df{}'.format(i)
    vars()[dfStr] = pd.read_csv(f, low_memory=False)
    i += 1


# In[14]:


for i in np.arange(1,12):
    print('------------------------')
    dfStr = 'df{}'.format(i)
    print(dfStr)
    print(eval(dfStr).shape)
    print('------------------------')
    for col in eval(dfStr).columns:
        print(col)
    


# In[15]:


df2Sub = df2[['FPA_ID','ICS_209_INCIDENT_NUMBER', 'FOD_ID', 'CONT_DATE', 'CONT_DOY', 'CONT_TIME', 'DISCOVERY_DOY', 'FIRE_YEAR']]
df1and2 = df1.merge(df2Sub, on='FPA_ID')


# In[16]:


sitrep = pd.read_csv('ics209-plus_sitrep_definitions.csv')


# In[17]:


sitrep.columns


# In[18]:


keep_set = []
for idx,row in sitrep.iterrows():
    if row[' Data Type'].strip() == 'Categorical' or row[' Data Type'].strip() == 'Numeric':
        keep_set.append(row['Column Name'])
        
keep_set


# In[19]:


for idx,row in sitrep.iterrows():
    print(row['Column Name'], row[' Description'], row[' Data Type'])


# In[20]:


df2.columns
df7temp = df7.copy()
keep_set.append('INCIDENT_NUMBER')
df7 = df7[keep_set]


# In[22]:


df7.rename(columns={'INCIDENT_NUMBER': 'ICS_209_INCIDENT_NUMBER'},inplace=True)
df1and2and7 = df1and2.merge(df7, on='ICS_209_INCIDENT_NUMBER')
epoch = pd.to_datetime(0, unit='s').to_julian_date()
df1and2and7['cont_date']  = pd.to_datetime(df1and2and7['CONT_DATE'] - epoch, unit="D")
df1and2and7['cont_time']  = df1and2and7['CONT_DATE'] - df1and2and7['DISCOVERY_DATE']


# In[23]:


for c in df1and2and7.columns:
    print(c, df1and2and7.loc[0][c])
    # ['STAT_CAUSE_CODE', 'LATITUDE', 'LONGITUDE', 'eco1', 'DISCOVERY_DOY', 'FIRE_YEAR', 'cont_time', 'FUEL_MODEL', 'GROWTH_POTENTIAL', 'TERRAIN', 'PROJECTED_FINAL_IM_COST', 'ACRES', 'TOTAL_PERSONNEL', 'STR_DAMAGED', 'WF_FSR', 'FATALITIES']


# In[24]:


finalKeep = ['fm.mean','Wind.mean', 'STAT_CAUSE_DESCR', 'LATITUDE', 'LONGITUDE', 'eco1', 'DISCOVERY_DOY', 'FIRE_YEAR', 'FUEL_MODEL', 'GROWTH_POTENTIAL', 'TERRAIN', 'cont_time', 'PROJECTED_FINAL_IM_COST', 'ACRES', 'TOTAL_PERSONNEL', 'STR_DAMAGED', 'WF_FSR', 'FATALITIES', 'FIRE_SIZE', 'FIRE_SIZE_CLASS']
task3df = df1and2and7[finalKeep]


# In[25]:


task3df.shape


# In[26]:


task3df.dropna(inplace=True)
task3df.shape


# In[27]:


task3df.head()


# In[28]:


task3df[(task3df['eco1']=='WATER')]


# In[29]:


for c in task3df['eco1'].unique():
    print(c)


# In[30]:


task3dfFinal = pd.get_dummies(task3df)
task3dfFinal.drop(columns=['eco1_0  WATER', 'eco1_0 NOT FOUND'], inplace=True)


# In[31]:


for c in task3dfFinal.columns:
    print(c)


# In[32]:


task3dfFinal.to_csv('task3df.csv')


# In[ ]:





# In[ ]:




