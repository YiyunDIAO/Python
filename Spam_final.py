# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:21:56 2020

@author: Administrator
"""

import pandas as pd 
import numpy as np 
import scipy
from scipy import stats
d1=pd.read_csv('Youtube01-Psy.csv')
d1['Video']='Psy'
d2=pd.read_csv('Youtube02-KatyPerry.csv')
d2['Video']='KatyPerry'
d3=pd.read_csv('Youtube03-LMFAO.csv')
d3['Video']='LMFAO'
d4=pd.read_csv('Youtube04-Eminem.csv')
d4['Video']='Eminem'
d5=pd.read_csv('Youtube05-Shakira.csv')
d5['Video']='Shakira'

Youtube=pd.concat([d1,d2,d3,d4,d5])
Youtube=Youtube.reset_index()
# check the null values 
Youtube.isnull().any()
# there are only null values in the "Date" column 

##### Pre-processing #####

# Step 1: Tokenization
# Tokenize the word and remove the stop word
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop=set(stopwords.words('english'))
Youtube['text']=[nltk.word_tokenize(w) for w in Youtube['CONTENT']]
for i in range(1956):
    Youtube['text'][i]=[w.lower() for w in Youtube['text'][i] if not w in stop]

# Step 2: lemmatization
# Try with lemmatization 
nltk.download('wordnet')
WNlemma=nltk.WordNetLemmatizer()
Youtube['lem']=0
for i in range(1956):
    Youtube['lem'][i]=[WNlemma.lemmatize(w) for w in Youtube['text'][i]]

Youtube['Content']=[' '.join(w) for w in Youtube['lem']]


# EDA 
# Q1: is there any length difference regarding spam and not spam 
# we also want to create an attribute to see the length of comment 

Youtube['len']=[len(w) for w in Youtube['text']]

spam=Youtube[Youtube['CLASS']==1]
nonspam=Youtube[Youtube['CLASS']==0]

print(np.mean(spam['len']))
print(np.mean(nonspam['len']))
print(scipy.stats.ttest_ind(spam['len'], nonspam['len']))
# we used t-test and the difference in length is significant enough 
# Conclusion: Spam comment is actually much longer than nonspam comment

# Q2: What are the most frequent words used in general, in spam and nonspam comments 
from nltk.probability import FreqDist
FreqDist=nltk.FreqDist()
for word in Youtube['text']:
    for w in list(word): 
        FreqDist[w]+=1

    
FreqDist1=nltk.FreqDist()
for word in spam['text']:
    for w in list(word):
        FreqDist1[w]+=1

FreqDist0=nltk.FreqDist()
for word in nonspam['text']:
    for w in list(word):
        FreqDist0[w]+=1

# Q2: will the spam text contain more digits? 
# define a function to count the number of digits
def count_digits(string):
    return sum(item.isdigit() for item in string)

Youtube['num_digit']=Youtube['CONTENT'].apply(count_digits)

spam=Youtube[Youtube['CLASS']==1]
nonspam=Youtube[Youtube['CLASS']==0]

print(np.mean(spam['num_digit']))
print(np.mean(nonspam['num_digit']))
print(scipy.stats.ttest_ind(spam['num_digit'], nonspam['num_digit']))

# Conclusion: spam information has significantly more digits than non-spam information 

# Q4: will the spam text contain more non-word characters? 
import re 
Youtube['non-word']=Youtube['CONTENT'].str.count(r'\W')

spam=Youtube[Youtube['CLASS']==1]
nonspam=Youtube[Youtube['CLASS']==0]

print(np.mean(spam['non-word']))
print(np.mean(nonspam['non-word']))
print(scipy.stats.ttest_ind(spam['non-word'], nonspam['non-word']))

# Conclusion: spam information has significantly more non-word characters than non-spam

# Modelling with CountVectorizer
# First we split the data 
from sklearn.model_selection import train_test_split   
X_train, X_test, y_train, y_test = train_test_split(Youtube['Content'], Youtube['CLASS'])     
        
        
# we want to first try with words 
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(min_df=2).fit(X_train)
X_train_vec=vect.transform(X_train)
X_test_vec=vect.transform(X_test)
# Feature Engineering: Extract useful features from the dataset 
# Define the function to add new features into the sparse matrix 
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

# Feature1: Length
X_train_text=[nltk.word_tokenize(w) for w in X_train]
X_train_len=[len(w) for w in X_train_text]
X_train_vec1=add_feature(X_train_vec, X_train_len)

X_test_text=[nltk.word_tokenize(w) for w in X_test]
X_test_len=[len(w) for w in X_test_text]
X_test_vec1=add_feature(X_test_vec, X_test_len)

# Feature2: Number of digits 
X_train_vec2=add_feature(X_train_vec1, X_train.apply(count_digits))
X_test_vec2=add_feature(X_test_vec1, X_test.apply(count_digits))

# Feature 3: Number of non-word characters 
X_train_vec3=add_feature(X_train_vec2, X_train.str.count(r'\W'))
X_test_vec3=add_feature(X_test_vec2, X_test.str.count(r'\W'))

# Model 0: Naive Bayes Classifier (CountVectorizer)
from sklearn.naive_bayes import MultinomialNB 
model0=MultinomialNB().fit(X_train_vec3, y_train)
prediction0=model0.predict(X_test_vec3)
print(model0.score(X_train_vec3, y_train))
print(model0.score(X_test_vec3, y_test))
# Training accuracy: 0.96, testing accuracy: 0.92

# Model 1: SVM (CountVectorizer)
from sklearn.svm import SVC 
model1=SVC().fit(X_train_vec3, y_train)
prediction1=model1.predict(X_test_vec3)
print(model1.score(X_train_vec3, y_train))
print(model1.score(X_test_vec3, y_test))
# Training Accuracy: 0.60, testing accuracy: 0.55

# Model 2: Logistic Regression (CountVectorizer)
from sklearn.linear_model import LogisticRegression 
model2=LogisticRegression(C=2).fit(X_train_vec3, y_train)
prediction2=model2.predict(X_test_vec3)
print(model2.score(X_train_vec3, y_train))
print(model2.score(X_test_vec3, y_test))
# Training accuracy: 0.99, testing accuracy: 0.96

# Now we want to tune the logistic Regression 
C_list=[1,2,3,4,5,6,7,8,9,10]
train_score=[]
test_score=[]
for value in C_list: 
    model2=LogisticRegression(C=value).fit(X_train_vec3, y_train)
    print(value)
    train_score.append(model2.score(X_train_vec3, y_train))
    test_score.append(model2.score(X_test_vec3, y_test))

log=pd.DataFrame(data={'C_value':C_list, 'train_accuracy': train_score, 'test_accuracy': test_score})
log.to_csv(r'G:\Duke MQM\Data Competition\YouTube Spam Detection\log.csv')
# Best Choice is still C=1

# Now based on the result, we want to know which tokens are most important for prediction 
feature_names=np.array(vect.get_feature_names()+['length', 'digit count', 'non-word count'])
sorted_coef_index=model2.coef_[0].argsort()

nonspam_words=pd.DataFrame(feature_names[sorted_coef_index[:20]])
nonspam_words.columns=['word']
nonspam_words['Spam Freq']=[FreqDist1[w] for w in nonspam_words['word']]
nonspam_words['Nonspam Freq']=[FreqDist0[w] for w in nonspam_words['word']]
nonspam_words['type']='nonspam'

spam_words=pd.DataFrame(feature_names[sorted_coef_index[-21:-1]])
spam_words.columns=['word']
spam_words['Spam Freq']=[FreqDist1[w] for w in spam_words['word']]
spam_words['Nonspam Freq']=[FreqDist0[w] for w in spam_words['word']]
spam_words['type']='spam'

top_words=pd.concat([spam_words, nonspam_words])

# Model 3: Decision Tree Classifier (CountVectorizer)
from sklearn.tree import DecisionTreeClassifier 
model3=DecisionTreeClassifier().fit(X_train_vec3, y_train)
print(model3.score(X_train_vec3, y_train))
print(model3.score(X_test_vec3, y_test))
# Training accuracy: 1.0, testing accuracy: 0.95 

# Model 4: Random Forest 
from sklearn.ensemble import RandomForestClassifier 
model4=RandomForestClassifier(random_state=0).fit(X_train_vec3, y_train)
print(model4.score(X_train_vec3, y_train))
print(model4.score(X_test_vec3, y_test))

# Model 5: Gradient Boosting Classifier 
from sklearn.ensemble import GradientBoostingClassifier 
model5=GradientBoostingClassifier(random_state=0, learning_rate=0.1, max_depth=10, n_estimators=100).fit(X_train_vec3, y_train)
print(model5.score(X_train_vec3, y_train))
print(model5.score(X_test_vec3, y_test))

Youtube.to_csv('G:\Duke MQM\Data Competition\YouTube Spam Detection\Youtube.csv')
top_words.to_csv(r'G:\Duke MQM\Data Competition\YouTube Spam Detection\topword.csv')
