
# coding: utf-8

# In[1]:


import pandas as pd


# ## Collection of Data

# In[2]:


legitimate_urls = pd.read_csv("datasets/legitimate-urls.csv")
phishing_urls = pd.read_csv("datasets/phishing-urls.csv")


# In[3]:


print(len(legitimate_urls))
print(len(phishing_urls))


# ## Data PreProcessing
# Since the dataset is in two DataFrames, I merged the two datasets together.
# I ensured the two data have the same column names

# In[4]:


phish_data = legitimate_urls.append(phishing_urls)


# In[5]:

phish_data.head(5)

# In[6]:

print(len(phish_data))
print(phish_data.columns)


# Removing Unnecessary columns

# In[7]:

phish_data = phish_data.drop(phish_data.columns[[0,3,5]],axis=1)
print(phish_data.columns)

# #### Since we merged two dataframes top 1000 rows will have legitimate urls and bottom 1000 rows will have phishing urls. So if we split the data now and create a model for it will overfit or underfit so we need to shuffle the rows before splitting the data into training set and test set

# In[8]:


# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
phish_data = phish_data.sample(frac=1).reset_index(drop=True)


# #### Removing class variable from the dataset
no_label_phish_data = phish_data.drop('label',axis=1)
no_label_phish_data.columns
y = phish_data['label']

# #### splitting the data into train data and test data

# X = data
# y = label

import random
random.seed(100)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(no_label_phish_data, y, test_size=0.20, random_state=100)
print(len(X_train),len(X_test),len(y_train),len(y_test))
print(y_train.value_counts())
print(y_test.value_counts())

# ## Creating Random Forest Model

from sklearn.ensemble import RandomForestClassifier
rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)
y_pred = rd_clf.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
train_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Confusion matrix: ")
print(conf_matrix)
print("Classification Report :")
print(classification_rep)
print("Train Accuracy :")
print(train_accuracy)
"""
# Saving the model to a file
import pickle
file_name = "RandomForestModel.sav"
pickle.dump(rf_model, open(file_name,'wb'))
"""

