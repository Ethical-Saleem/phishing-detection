
# coding: utf-8

# In[1]:


import pandas as pd


# ## Collection of Data

# In[2]:


legitimate_urls = pd.read_csv("datasets/legitimate-urls.csv")
phishing_urls = pd.read_csv("datasets/phishing-urls.csv")


# In[3]:


legitimate_urls.head(10)
phishing_urls.head(10)


# ## Data PreProcessing
# Since the dataset is in two DataFrames, I merged the two datasets together.
# I ensured the two data have the same column names

# In[4]:


phish_data = legitimate_urls.append(phishing_urls)


# In[5]:

phish_data.head(5)

# In[6]:

phish_data.columns

# Removing Unnecessary columns

# In[7]:

phish_data = phish_data.drop(phish_data.columns[[0,3,5]],axis=1)

# #### Since we merged two dataframes top 1000 rows will have legitimate urls and bottom 1000 rows will have phishing urls. So if we split the data now and create a model for it will overfit or underfit so we need to shuffle the rows before splitting the data into training set and test set

# In[8]:

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
phish_data = phish_data.sample(frac=1).reset_index(drop=True)

# #### Removing class variable from the dataset

# In[9]:

no_label_phish_data = phish_data.drop('label',axis=1)
no_label_phish_data.columns
y = phish_data['label']

# #### splitting the data into train data and test data

# In[49]:

# X = data
# y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(no_label_phish_data, y, test_size=0.30, random_state=100)
print(len(X_train),len(X_test),len(y_train),len(y_test))
print(y_train.value_counts())
print(y_test.value_counts())

# #### Checking the data is split in equal distribution or not

# In[158]:

train_0_dist = 711/1410
print(train_0_dist)
train_1_dist = 699/1410
print(train_1_dist)
test_0_dist = 306/605
print(test_0_dist)
test_1_dist = 299/605
print(test_1_dist)

# #### creating the model and fitting the data into the model

# In[50]:

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# #### predicting the result for test data

# In[51]:

y_pred = clf.predict(X_test)

print(y_pred)
print(list(y_test))

# #### creating confusion matrix and checking the accuracy

# In[54]:

from sklearn.metrics import confusion_matrix,accuracy_score
conf_matrix = confusion_matrix(y_test, y_pred)
train_accuracy = accuracy_score(y_test, y_pred)
print("Confusion matrix: ")
print(conf_matrix)
print("Train Accuracy :")
print(train_accuracy)

# ## Creating Random Forest Model

# In[55]:

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# In[56]:

rf_pred = rf_model.predict(X_test)

# In[57]:

print(list(y_test)),print(list(rf_pred))

# In[58]:

conf_matrix_2 = confusion_matrix(y_test, rf_pred)
conf_matrix_2

# In[60]:

train_accuracy_2 = accuracy_score(y_test, rf_pred)

# ### Improving the efficiency 

# In[138]:

eff_model = RandomForestClassifier(n_estimators=100,max_depth=30,max_leaf_nodes=10000)

# In[140]:

eff_model.fit(X_train, y_train)

# In[142]:

y_eff = eff_model.predict(X_test)

# In[144]:

conf_matrix_3 = confusion_matrix(y_test, y_eff)
conf_matrix_3

# In[146]:

training_accuracy_3 = accuracy_score(y_test, y_eff)
