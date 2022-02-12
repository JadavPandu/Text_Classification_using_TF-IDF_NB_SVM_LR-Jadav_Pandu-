#!/usr/bin/env python
# coding: utf-8

#    # Text classification using TF-IDF measurement and prediciting through Naive Bayes ,SVM and Logistic Regression models
# 

# # Importing required libraries and Dataset

# In[2]:


import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegressionCV


np.random.seed(500)
#df=pd.read_excel(r'G:\TROP ICSU INTERNSHIP\tweet_global_warming.xlsx',engine='openpyxl')
df = pd.read_excel("tweet_global_warming.xlsx")


# # Exploratory Data Analysis 

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


#cleaning the existence column i.e classifing all the similar meaning words as one word

import warnings
warnings.filterwarnings('ignore')

df['existence'].fillna("neutral",inplace = True)
df.existence
for i in range(0,5960):
    if df['existence'][i]=='Y':
        df['existence'][i]='yes'
    if df['existence'][i]=='Yes':
        df['existence'][i]='yes'
    if df['existence'][i]=='No':
        df['existence'][i]='no'
    if df['existence'][i]=='N':
        df['existence'][i]='no'
        


# # Pie chart visualization of Positive Negative and Neutral tweets

# In[6]:



fig=plt.figure(figsize=(5,5))
colors=["skyblue",'pink','red']
pos=df[df['existence']=='yes']
neg=df[df['existence']=='no']
neutral=df[df['existence']=='neutral']
ck=[pos['existence'].count(),neg['existence'].count(),neutral['existence'].count()]
legpie=plt.pie(ck,labels=["Positive","Negative","Neutral"],
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors = colors,
                 startangle = 45,
                 explode=(0, 0.1,0.1))


# # Data pre-processing

# In[7]:


#remove_url
import string
def remove_url(thestring):
    URLless_string = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', thestring)
    return URLless_string
df['tweet']=df['tweet'].apply(lambda x:remove_url(x))

#remove_html
def preprocessor(text):
             text=re.sub('<[^>]*>','',text)
             emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
             text=re.sub('[\W]+',' ',text.lower()) +                ' '.join(emojis).replace('-','')
             return text   
df['tweet']=df['tweet'].apply(preprocessor)

#-------------
result=string.punctuation  
print(result)
#-------------

#remove punctuations
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s
df['tweet'] = df['tweet'].apply(remove_punctuation)
df.head()


# In[8]:


#tokenizing the words in sentences
tokenizer=RegexpTokenizer('\w+')
df['after_tokenizer']=df['tweet'].apply(lambda x: tokenizer.tokenize(x.lower()))
df['after_tokenizer'][0]


# In[9]:


#removing unwanted words by stopwords
stoplist=stopwords.words('english')
stoplist.append('link')  #adding unwanted words to stoplist 

def remove_stopwords(text):
    words=[w for w in text if w not in stoplist and w.isalpha()]
    return words
df['stopword']=df['after_tokenizer'].apply(lambda x:remove_stopwords(x))
df.head() 


# In[10]:


#nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text=" ".join([lemmatizer.lemmatize(i) for i in text])
    return lem_text
df['lemmatize']=df['stopword'].apply(word_lemmatizer)
df['lemmatize']


# In[11]:


#combining the words in a sentence after lemmatization
stemmer =PorterStemmer()
def word_stemmer(text):
    stem_text=" ".join([stemmer.stem(i) for i in text])
    return stem_text
df['stem']=df['stopword'].apply(lambda x:word_stemmer(x))
df.head()
df['stem']


# # Analysing text after pre-processing
# 
# **Wordcloud for  Positive and Negative Tweets**

# In[12]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from nltk.corpus import stopwords

comment_words = '' 
stoplist = set(STOPWORDS) 
stoplist.add('rt')
stoplist.add('via')
stoplist.add('並')

positivedata = df[ df['existence'] == 'yes']
positivedata =positivedata['lemmatize']
negdata = df[df['existence'] == 'no']
negdata= negdata['lemmatize']

def wordcloud_draw(data, color = 'white'):
    words = ' '.join(data)
    wordcloud = WordCloud(stopwords=stoplist,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(words)
    plt.figure(1,figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words are as follows")
wordcloud_draw(positivedata,'white')
print("Negative words are as follows")
wordcloud_draw(negdata,'white')


# **Bar Graphs for most used words in Postive and Negative Tweets**

# In[13]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import collections
import matplotlib.cm as cm
from matplotlib import rcParams

def bar_graph(data):
    words = ' '.join(data)
    filtered_words = [word for word in words.split() if word not in stoplist]
    counted_words = collections.Counter(filtered_words)
    words = []
    counts = []
    for letter, count in counted_words.most_common(20):
        words.append(letter)
        counts.append(count)
        colors = cm.rainbow(np.linspace(0, 1, 10))
    rcParams['figure.figsize'] = 7, 7

    plt.title('Top words in the tweets vs their count')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.barh(words, counts, color=colors)
    
print("Positive words are as follows")
bar_graph(positivedata)
     


# In[14]:


print("Negative words are as follows")
bar_graph(negdata)


# # Dividing data set as Train and Test sets for predictions

# In[15]:


from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

Train_X, Test_X, Train_Y1, Test_Y = model_selection.train_test_split(df['stem'],df['existence'],test_size=0.3, random_state =42)
#print(Train_Y)


# In[16]:


#encoding the classfying column

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y1)
Test_Y = Encoder.fit_transform(Test_Y)
print(Train_Y)
print(Train_Y1)


# # Vectorizing the words in sentences

# In[17]:



Tfidf_vect = TfidfVectorizer(max_features=20000)
Tfidf_vect.fit(df['stem'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
print(Tfidf_vect.vocabulary_)

#we get the unique values for every word in everysentence


# In[18]:


print(Train_X_Tfidf)
 #getting the TF-IDF  score for all words.


# # Prediction using Naive Bayes model

# In[19]:


Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# **visualization of prediction_after_using_NB**

# In[20]:


import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


labels = ['yes', 'no','neutral']
label=[2,1,0]

print(classification_report(Test_Y,predictions_NB,target_names=labels))
cm = confusion_matrix(Test_Y, predictions_NB,labels=label)


sns.heatmap(cm, linewidths=1, annot=True, fmt='g')
plt.xlabel("predicted positive ,negative and neutral")
plt.ylabel("Actual positive negative and neutral")
plt.show()


# # Prediction using SVM 

# In[21]:


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# **visualization of prediction_after_using_SVM**

# In[22]:


print(classification_report(Test_Y,predictions_SVM,target_names=labels))
cm = confusion_matrix(Test_Y, predictions_SVM,labels=label)


sns.heatmap(cm, linewidths=1, annot=True, fmt='g')
plt.xlabel("predicted positive ,negative and neutral")
plt.ylabel("Actual positive negative and neutral")
plt.show()


# # Prediction using Logistic Regression

# In[23]:


clf=LogisticRegressionCV(scoring='accuracy',random_state=0,verbose=3,max_iter=500).fit(Train_X_Tfidf,Train_Y)
predictions_lr = clf.predict(Test_X_Tfidf)
from sklearn import metrics

print("Logistic Regression Accuracy score -> ",metrics.accuracy_score(Test_Y, predictions_lr))


# **visualization of prediction_after_using_Logistic_Regression**

# In[24]:


print(classification_report(Test_Y,predictions_lr,target_names=labels))
cm = confusion_matrix(Test_Y, predictions_lr,labels=label)


sns.heatmap(cm, linewidths=1, annot=True, fmt='g')
plt.xlabel("predicted positive ,negative and neutral")
plt.ylabel("Actual positive negative and neutral")
plt.show()


# # --------------------------------------------END------------------------------------------

# In[ ]:




