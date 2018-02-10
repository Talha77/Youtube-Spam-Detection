# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:27:05 2017
@author: hammadkhan
"""
#Loading Libraries.
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC,SVR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
eng_stopwords = set(stopwords.words("english"))

## Reading all the datasets ##
Psy_df = pd.read_csv("Youtube01-Psy.csv")
KatyPerry_df = pd.read_csv("Youtube02-KatyPerry.csv")
LMFAO_df = pd.read_csv("Youtube03-LMFAO.csv")
Eminem_df = pd.read_csv("Youtube04-Eminem.csv")
Shakira_df = pd.read_csv("Youtube05-Shakira.csv")

#Merging dataframes into single dataframe and making testing and training df
frames = [Psy_df,KatyPerry_df,LMFAO_df,Eminem_df,Shakira_df]
Dataset = pd.concat(frames)
train_df = pd.concat(frames)
test_df = Shakira_df
#test_df = test_df.drop('CLASS', axis=1)
Meta_features = pd.DataFrame()

## Creating metafeatures ##

## Number of words in the text ##
train_df["num_words"] = train_df["CONTENT"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["CONTENT"].apply(lambda x: len(str(x).split()))

## Number of unique words in the CONTENT ##
train_df["num_unique_words"] = train_df["CONTENT"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["CONTENT"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the CONTENT ##
train_df["num_chars"] = train_df["CONTENT"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["CONTENT"].apply(lambda x: len(str(x)))

## Number of stopwords in the CONTENT ##
train_df["num_stopwords"] = train_df["CONTENT"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["CONTENT"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the CONTENT ##
train_df["num_punctuations"] =train_df['CONTENT'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['CONTENT'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the CONTENT ##
train_df["num_words_upper"] = train_df["CONTENT"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["CONTENT"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the CONTENT ##
train_df["num_words_title"] = train_df["CONTENT"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["CONTENT"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the CONTENT ##
train_df["mean_word_len"] = train_df["CONTENT"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["CONTENT"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

### Prepare the data for modeling ###
train_y = train_df['CLASS']


cols_to_drop = ['COMMENT_ID','AUTHOR','DATE','CONTENT', 'CLASS']
train_X = train_df.drop(cols_to_drop, axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

#Tfidf features
tfidf_vect = TfidfVectorizer(ngram_range = (1,3),norm='l2',smooth_idf = False , analyzer='word', max_df= 0.30,min_df = 12, stop_words = 'english')
tfidf_features = tfidf_vect.fit_transform(Dataset['CONTENT'].values.tolist())

#Splitting the dataset into Training set and Testing set.
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(tfidf_features, Dataset.CLASS, train_size=0.7,test_size = 0.3, random_state = 10)

#CountVectorized features
count_vect = CountVectorizer(ngram_range=(1,3), analyzer='word', max_df= 0.25, min_df= 12, stop_words= 'english')
countvectorized_features = count_vect.fit_transform(Dataset['CONTENT'].values.tolist())

#Splitting the dataset into Training set and Testing set.
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(countvectorized_features, Dataset.CLASS, train_size=0.7,test_size = 0.3, random_state = 10)


#Fitting the Model and predicting.
model_count_NB = MultinomialNB(alpha=0.05)
model_count_NB.fit(X_train_count, y_train_count)
predictions_count = model_count_NB.predict(X_test_count)


model_count_LR = LogisticRegression()
model_count_LR.fit(X_train_count, y_train_count)
predictions_count = model_count_LR.predict(X_test_count)

model_count_SVC = SVC(kernel = 'linear', random_state = 0)
model_count_SVC.fit(X_train_count, y_train_count)
predictions_count = model_count_SVC.predict(X_test_count)

#n_estimator = 8 max_features = 30 acc = 94.3%
model_count_RF = RandomForestClassifier(n_estimators = 8, max_features = 30 ,criterion = 'entropy', random_state = 0)
model_count_RF.fit(X_train_count, y_train_count)
predictions_count = model_count_RF.predict(X_test_count)

#n_estimator = 8 max_features = 30 acc = 94.3%
model_count_AB = AdaBoostClassifier(n_estimators = 1000,random_state = 0)
model_count_AB.fit(X_train_count, y_train_count)
predictions_count = model_count_AB.predict(X_test_count)

#n_estimator = 1000 random_state = 10 acc = 94.54%
model_count_GB = GradientBoostingClassifier(n_estimators = 1000, random_state = 10)
model_count_GB.fit(X_train_count, y_train_count)
predictions_count = model_count_GB.predict(X_test_count)

#Ensembling
clf1 = LogisticRegression(random_state = 10)
clf2 = RandomForestClassifier(n_estimators = 8, max_features = 30 ,criterion = 'entropy', random_state = 0)
clf3 = GradientBoostingClassifier(n_estimators = 1000, random_state = 10)
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
                        voting='hard')
#Ensembling predictions
eclf.fit(X_train_count, y_train_count)
predictions_count = eclf.predict(X_test_count)

# Calculating accuracy now
from sklearn.metrics import accuracy_score
accuracy_count = accuracy_score(y_test_count, predictions_count)
print('Count Vectorized Words Accuracy:', accuracy_count)


### Function to create confusion matrix ###
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Calculating Confusion Matrix
cnf_matrix = confusion_matrix(y_test_count, predictions_count)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix, classes=['SPAM', 'HAM'],
                      title='Confusion matrix')
plt.show()


