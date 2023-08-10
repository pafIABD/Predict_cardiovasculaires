#!/usr/bin/env python
# coding: utf-8

# In[31]:


#!pip install spark


# In[11]:


# Importer PySpark
import pyspark
from pyspark.sql import SparkSession
#Create SparkSession
spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
sc=spark.sparkContext


# In[32]:


#Installer findspark
#!pip install findspark 


# In[24]:


# Importer findspark
import findspark
findspark.init()
#importer pyspark
import pyspark
from pyspark.sql import SparkSession
#Creer  SparkSession qui va créer SparkContext.

spark = SparkSession.\
        builder.\
        appName("pyspark-nb-3-analysis").\
        master("spark://spark-master:7077").\
        config("spark.executor.memory", "512m").\
        config("spark.eventLog.enabled", "true").\
        config("spark.eventLog.dir", "file:///opt/workspace/events").\
        getOrCreate()  


# In[25]:


import pyspark
from pyspark.sql import SparkSession
spark=SparkSession(sc)


# In[26]:


df = spark.read.format('csv').load('CVD_cleaned.csv', header=True, sep=",")


# In[27]:


df.show(1)


# In[53]:


# Lire le dataframe
health = spark.read.option("inferSchema", True).option('delimiter',',').option('header', True).option('encoding', 'UTF-8').csv("CVD_cleaned.csv")


# In[54]:


health.show(10)


# In[59]:


# convertir le dataset en dataframe pandas

df = health.toPandas()


# In[63]:


df.head(10)


# ANALYSE DESCRIPTIVE DES DONNEES

# In[60]:


df.describe()


# In[75]:


sns.catplot(data=df, x="General_Health", kind="count", palette="ch:.25")


# In[66]:


#ETAT SANITAIRE ET INDICE TAILLE CORPORELLE
import seaborn as sns
sns.catplot(data=df, x="General_Health", y="BMI", kind="box")


# In[ ]:


#ETAT SANITAIRE ET CONSOMMATION DE LEGUMES_FRUITS


# In[73]:


sns.displot(df, x="Fruit_Consumption", hue="General_Health", kind="kde",multiple="stack")


# In[76]:


sns.countplot(data=df,x='General_Health',hue='Heart_Disease')


# In[79]:


#Analyse des corrélations entre les variables

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df.corr()


# In[80]:


sns.heatmap(df.corr())


# In[81]:


df.groupby('Heart_Disease').count()


# In[82]:


#K Nearest Neighbors

#CLASSIFICATION ET PREDICTION


# In[83]:


#verification des données vides
print(df.isnull().sum())


# In[86]:


from sklearn.preprocessing import LabelEncoder
#Encodage des données
def label_encoder(y):
    le = LabelEncoder()
    df[y] = le.fit_transform(df[y])
 
label_list = ["General_Health","Checkup", "Exercise","Heart_Disease","Skin_Cancer", "Other_Cancer","Depression","Diabetes", "Arthritis","Sex","Age_Category","Smoking_History"]
 
for l in label_list:
    label_encoder(l)
 
#Afficher les données
df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#MACHINE LEARNING KNN CLASSIFICATION


# In[87]:


#Importer les librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[89]:


#Diviser le dataset en variables en dépendantes et indépendantes
X = df.drop(["Heart_Disease"],axis=1)
y = df['Heart_Disease']
 
#Scinder en données d'entrainement(80%) et de test(20%)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=42, shuffle=True) 
 
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
 
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)


# In[90]:


#Standardization des données

#Mise en echelle
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[91]:


#Stocker les résultats du modele  dans deux  dictionnaires
result_dict_train = {}
result_dict_test = {}


# In[92]:


knn = KNeighborsClassifier()
accuracies = cross_val_score(knn, X_train, y_train, cv=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
 
#Afficher la précision
print("Train Score:",np.mean(accuracies))
print("Test Score:",knn.score(X_test,y_test))


# In[98]:





# In[ ]:





# In[ ]:




