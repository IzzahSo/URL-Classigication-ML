#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask 
from flask import Flask
from flask import jsonify
from flask import request 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[5]:


def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/') #get tokens after splitting by slash
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-') #get tokens after splitting by dash
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.') #get tokens after splitting by dot
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens)) #remove redundant tokens
    if 'com' in allTokens:
        allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return allTokens


# In[6]:


def TL():
    allurls = r'C:\Users\candra\Pictures\SecQR\ml\data.txt'	#path data
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
    allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe

    allurlsdata = np.array(allurlsdata)	#converting it into an array
    random.shuffle(allurlsdata)	#shuffling

    y = [d[1] for d in allurlsdata]	#all labels 
    corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio

    lgs = LogisticRegression()	#using logistic regression
    lgs.fit(X_train, y_train)
    #print(lgs.score(X_test, y_test))	#print the score.
    return vectorizer, lgs


# In[16]:


app = Flask(__name__)
# defining a route
@app.route("/", methods=['GET', 'POST', 'PUT']) # decorator
def home():
    Input = request.args.get('url')
    vectorizer, lgs = TL()
    X_predict = [Input]
    X_process = vectorizer.transform(X_predict)
    y_Predict = lgs.predict(X_process)
    y_string = (','.join(y_Predict))

#print(y_Predict)
#print(y_string)#printing predicted values
    print("success")
    return (y_string)
app.run(debug = True) 


# In[ ]:




