#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Internship TEST2
## Logistic Regression

import pandas as pd
df = pd.read_csv("data.csv")


# In[2]:


df


# In[3]:


df.shape


# In[4]:


df.isnull().any(axis = 0)


# In[5]:


df.dropna(inplace = True)


# In[6]:


df.shape


# In[7]:


features = df.iloc[:,0:2]


# In[8]:


labels = df.iloc[:,-1]


# In[9]:


features


# In[10]:


labels


# In[15]:


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features, labels, train_size = 0.8, random_state = 41)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


logmodel = LogisticRegression()
logmodel.fit(features_train,labels_train)


# In[18]:


labels_pred = logmodel.predict(features_test)


# In[19]:


print("Accuracy",(logmodel.score(features_test,labels_test)))


# In[20]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(labels_test,labels_pred)
print(confusion_matrix)


# In[ ]:


## We predicted that True positive as 212 ,False positive as 136, False Negative as 44 and True Negative as 52...
##Number of class are correctly 212 (predicted)


# In[21]:


### 2.Training SVM model with linear kernel

# Loading the data

df


# In[22]:


## Spliting the data as features and labels as in Logostic Regression

print(features)
print(labels)


# In[23]:


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features, labels, test_size = 0.8, random_state = 41)


# In[26]:


## R is the SVM regularization parameter
C = 1.0



# In[30]:


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

model = SVC()
model.fit(features_train,labels_train)


# In[32]:


predict = model.predict(features_test)
print("Accuracy ",accuracy_score(labels_test,predict))


# In[ ]:


## Accuracy of the model is 94.81%


# In[37]:


### 3Decision Tree model

#loading the data
df


# In[38]:


## Creating the train and test data
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features, labels, test_size = 0.8, random_state = 41)


# In[39]:


labels_train.value_counts(normalize = True)


# In[41]:


features_train.shape, labels_train.shape


# In[56]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# In[43]:


dt_model = DecisionTreeClassifier(random_state = 10)


# In[45]:


dt_model.fit(features_train,labels_train)
dt_model.score(features_train,labels_train)


# In[46]:


dt_model.score(features_test,labels_test)


# In[47]:


dt_model.predict(features_test)


# In[48]:


dt_model.predict_proba(features_test)


# In[49]:


labels_pred = dt_model.predict_proba(features_test)[:,1]


# In[51]:


new_labels = []
for i in range(len(labels_pred)):
    if labels_pred[i]< 0.0:
        new_labels.append(0)
    else:
        new_labels.append(1)
            


# In[52]:


from sklearn.metrics import accuracy_score
accuracy_score(labels_test, new_labels)


# In[57]:


#### Accuracy for Decision Tree Model is 44.03%


# In[58]:


## 4 KNN model
# Loading the data
df


# In[1]:


from sklearn.preprocessing import StandardScalar
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transfore(features_train)
features_test = scaler.transfore(features_test)


# In[ ]:


from sklearn

