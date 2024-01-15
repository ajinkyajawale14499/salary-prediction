#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# In[2]:


import os
for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[7]:


df_train_salaries = pd.read_csv('data/train_salaries.csv', header=0, sep=',', quotechar='"')


# In[8]:


df_train_salaries.head(5)


# In[9]:


df_train_salaries.tail()


# In[10]:


df_train_features = pd.read_csv('data/train_features.csv', header=0, sep=',', quotechar='"')
df_train_features.tail()


# In[11]:


df_test_features = pd.read_csv('data/test_features.csv', header=0, sep=',', quotechar='"')
df_test_features.tail()


# ### EDA

# In[18]:


df_train_salaries.describe()


# ### understanding data

# In[51]:


print(" \n Train salaries \n")

df_train_salaries.tail().info()

print(" \n Train features \n")

df_train_features.tail().info()

print(" \n Test features \n")

df_test_features.tail().info()


# ### Null Values Handling

# In[55]:


print(" \n Train features \n")

print(" in train dataset \n\n",df_train_features.isna().sum())


print(" \n Test features \n")

print("Null Values in test dataset \n\n",df_test_features.isna().sum())

print(" \n Train salaries \n")

print("Null Values in train salaries dataset \n\n",df_train_salaries.isna().sum())



# In[62]:


print(df_train_features.head(5))
print(len(df_train_features))

print(df_train_salaries.head(5))
print(len(df_train_salaries))

print(df_test_features.head(5))
print(len(df_test_features))

#combined the features and salaries in the training data
training_df = pd.merge(df_train_features,df_train_salaries, how = 'inner', on = 'jobId')
print(training_df.head(5))
print(len(training_df))

#look for duplicated data and invalid data
training_df = training_df.drop_duplicates(subset="jobId")

print(len(training_df))

test_df = df_test_features.drop_duplicates(subset="jobId")
print(len(test_df))


# In[63]:


#summarize each variable in the training set
training_df.info()
training_df.describe(include = 'all')


# ### feature engineering
# 

# In[64]:


#observation: the minimum salary is 0, which might be missing data
print(training_df.loc[training_df["salary"]==0])


# In[67]:


def convert_to_category(df, col):
    df[col] = df[col].astype('category')
    return df


# In[68]:


training_df = convert_to_category(training_df, 'companyId')
training_df = convert_to_category(training_df, 'jobType')
training_df = convert_to_category(training_df, 'degree')
training_df = convert_to_category(training_df, 'major')
training_df = convert_to_category(training_df, 'industry')
training_df.info()


# In[69]:


#removing the cases with salary = 0
training_df = training_df.drop(training_df[training_df['salary']==0].index)
training_df.info()


# ### Salary normal distribution

# In[47]:


# !sudo apt-get install matplotlib


# In[46]:


# import matplotlib.pyplot as plt

# plt.figure(figsize = (14, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(training_df.num)
# plt.subplot(1, 2, 2)
# sns.distplot(training_df.num, bins = 20)
# plt.show()


# ## Baseline models
# 
# 1. Calculating Mean Squared Error (MSE) in the training dataset
# 
# - use the average salary of each industry
# - use the average salary of each major
# - use the average salary of each degree
# - use the averaged salary of each jobType 

# In[15]:



def salary_get_variable_list():
    variable_list = ['companyId', 'jobType', 'degree', 'major', 'industry']
    return variable_list

#transform a variable based on the average of the target variable of each value
#for example, transfor each industry into the averaged salary of each industry
def transform_categorical(df, col, target, training_df): 
    category_mean = {}
    value_list = df[col].cat.categories.tolist()
    for value in value_list:
        category_mean[value] = training_df[training_df[col] == value][target].mean()
    df[col+'_transformed'] = df[col].map(category_mean)
    df[col+'_transformed'] = df[col+'_transformed'].astype('int64')
    return df


#make sure training and test has the same categorical variables
def encode_categorical(training, test):
    from sklearn import preprocessing
    cols = training.select_dtypes(include=['category']).columns.to_list()
    for col in cols:
        le = preprocessing.LabelEncoder()
        le.fit(training[col])
        training[col+'_encoded'] = le.transform(training[col])
        test[col+'_encoded'] = le.transform(test[col])
    return training, test

def convert_to_category(df, col):
    df[col] = df[col].astype('category')
    return df

def drop_duplicates(df, col):
    df = df.drop_duplicates(subset = col)
    return df


# In[17]:



def salary_preprocess():
    #define constants
    variable_list = salary_get_variable_list()
    target = 'salary'
    
    #read the data and merge feature and salary for the training data
    features = pd.read_csv('data/train_features.csv')
    salaries = pd.read_csv('data/train_salaries.csv')
    test = pd.read_csv('data/test_features.csv')
    training = pd.merge(features,salaries, how = 'inner', on = 'jobId')

    #remove duplicates
    training = drop_duplicates(training,'jobId')
    test = drop_duplicates(test,'jobId')

    #remove salary = 0 in the training set
    training = training.drop(training[training[target]==0].index)

    #convert object to categorial variables
    #and transform them based on mean target (salary)
    for variable in variable_list:
        training = convert_to_category(training, variable)
        training = transform_categorical(training, variable, target, training)
        test = convert_to_category(test, variable)
        test = transform_categorical(test, variable, target, training)

    #encode categorical variables to dummies
    training, test = encode_categorical(training, test)
    
    #save scaler for later
    
    #print results on the screen
    training.info()
    training.head()
    test.info()
    test.head()
    return training, test


training, test = salary_preprocess()


# In[19]:


import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

# the variable 'industry_transformed' had transformed each indusstry with the mean salary of that industry
# same with 'major_transformed', 'degree_transformed', 'jobType_transformed'
prediction_by_industry = mean_squared_error(training['salary'], training['industry_transformed'])
print('Prediction by industry is {}'.format(prediction_by_industry))
prediction_by_major = mean_squared_error(training['salary'], training['major_transformed'])
print('Prediction by major is {}'.format(prediction_by_major))
prediction_by_degree = mean_squared_error(training['salary'], training['degree_transformed'])
print('Prediction by degree is {}'.format(prediction_by_degree))
prediction_by_jobType = mean_squared_error(training['salary'], training['jobType_transformed'])
print('Prediction by jobType is {}'.format(prediction_by_jobType))


# In[20]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# In[21]:


models = []
mean_mse = {}
cv_std = {}
res = {}
n_procs = 4

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators = 80, n_jobs = n_procs, max_depth = 20, min_samples_split = 70,
                          max_features = 7, verbose = 0)
gbm = GradientBoostingRegressor(n_estimators = 40, max_depth = 7, loss = 'ls', verbose = 0)

models.extend([lr, rf, gbm])


# In[22]:


feature_transformed = ['yearsExperience', 'milesFromMetropolis', 'companyId_transformed', 
                'jobType_transformed', 'degree_transformed', 'major_transformed', 'industry_transformed']
feature_encoded = ['yearsExperience', 'milesFromMetropolis', 'companyId_transformed', 
                'jobType_encoded', 'degree_encoded', 'major_encoded', 'industry_encoded']


# In[23]:



from sklearn.model_selection import cross_val_score
def cross_val_model(model, feature_df, target, n_procs, mean_mse, cv_std):
    neg_mse = cross_val_score(model, feature_df, target, cv = 5, n_jobs = n_procs, 
                              scoring = 'neg_mean_squared_error')
    mean_mse[model] = -1.0 * np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)
    

#print a short summary
def print_summary(model, mean_mse, cv_std):
    print('\nmodel:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Standard deviation during cross validation:\n', cv_std[model])


#feature importance
def get_model_feature_importances(model, feature_df):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0] * len(feature_df.columns)
    
    feature_importances = pd.DataFrame({'feature': feature_df.columns, 'importance': importances})
    feature_importances.sort_values(by = 'importance', ascending = False, inplace = True)
    ''' set the index to 'feature' '''
    feature_importances.set_index('feature', inplace = True, drop = True)
    return feature_importances


# In[24]:


#  Cross validation - feature transformed not scaled
for model in models:
    cross_val_model(model, training[feature_transformed], training['salary'], n_procs, mean_mse, cv_std)
    print_summary(model, mean_mse, cv_std)


# In[25]:


# feature_encoded & not scaled
for model in models:
    cross_val_model(model, training[feature_encoded], training['salary'], n_procs, mean_mse, cv_std)
    print_summary(model, mean_mse, cv_std)


# In[26]:


from sklearn.preprocessing import StandardScaler
def apply_scaler(train, test, target):
    scale = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    colnames = train_scaled.select_dtypes(include=['float64','int64']).columns.to_list()
    colnames.remove(target)
    print('\nThe following columns are scaled:\n')
    print(colnames)
    scale.fit(train_scaled[colnames])
    train_scaled[colnames] =  scale.transform(train_scaled[colnames])
    test_scaled[colnames] =  scale.transform(test_scaled[colnames])
    return train_scaled, test_scaled

#scale numeric variables
#weight to see whether to run it or not
training, test = apply_scaler(training, test, 'salary')


# In[43]:


# # feature transformed & scaled 
# for model in models:
#     cross_val_model(model, training[feature_transformed], training['salary'], n_procs, mean_mse, cv_std)
#     print_summary(model, mean_mse, cv_std)


# In[28]:


# feature encoded scaled
for model in models:
    cross_val_model(model, training[feature_encoded], training['salary'], n_procs, mean_mse, cv_std)
    print_summary(model, mean_mse, cv_std)


# In[29]:


# Best model based on MSE - transformed and scaled features using Gradient boosting
bestModel = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=7,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=40,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)


# In[ ]:




### Training
# In[30]:


bestModel.fit(training[feature_transformed], training['salary'])


# In[31]:


GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=7,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=40,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)


# ### Model Testing

# In[32]:


predicted_salaries = bestModel.predict(test[feature_transformed])


# In[33]:


test['predicted_salary'] = predicted_salaries.tolist()


# In[34]:


test.head(10)


# In[35]:


test.to_csv('data/test_salary.csv', index = False)


# ### feature importance

# In[36]:


feature_importances = get_model_feature_importances(bestModel, training[feature_transformed])


# In[42]:


# feature_importances.plot.bar(figsize=(20,10))


# In[ ]:




