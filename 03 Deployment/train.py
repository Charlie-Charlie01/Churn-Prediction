#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[2]:


# setting model parameters
C = 1.0
n_splits = 5

output_file = f'model_C={C}.bin'


# In[3]:


# data preparation
df = pd.read_csv('data-week-3.csv')


# In[4]:


# standardizing the dataset
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

for c in categorical_columns:
    df[c] = df[c].astype(str).str.lower().str.replace(' ', '_')


# In[5]:


df.head().T


# In[6]:


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)


# In[7]:


df.churn = (df.churn == 'yes').astype(int)


# In[8]:


# data splitting
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[9]:


# let's reset the indices
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# now we need to extract our target variable, which is 'y' (churn)
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# certainly, to prevent accidental use of the 'churn' variable when building model, we should remove it from our dataframes
del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[10]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[11]:


# training
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


# In[12]:


# prediction pipeline
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[13]:


# validation
print(f'doing validation with C={C}')


# In[14]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold{fold} is {auc}')
    fold += 1

print('validation result:' )
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[15]:


# train the final model
print('train the final model')

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# In[16]:


# saving the model with pickle
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')


# In[ ]:




