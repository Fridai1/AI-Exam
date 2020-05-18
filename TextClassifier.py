# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd


# Modelling
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn import decomposition, ensemble

# Data preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

# validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

##### TEST REMOVE THIS
from sklearn.datasets import fetch_20newsgroups
categories = ['sci.crypt', 'talk.politics.misc','sci.space', 'comp.graphics', 'talk.politics.guns', 'sci.med' ]
remove_these = ('headers', 'footers', 'quotes')
train_set = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, remove=remove_these)
test_set = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, remove=remove_these)

dora_X_train, dora_y_train = train_set.data, train_set.target
dora_X_test, dora_y_test = test_set.data, test_set.target

target_names = train_set.target_names


#### END TEST

# HELPER METHODS


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    
    classifier.fit(feature_vector_train,label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, y_test)

def ConvertColumnArrayToNormalArray(array):
    newArray = []
    for i in array:
        newArray.append(i)
    return newArray


# [[[[DATASET PREPARATION]]]]

moviesDF = pd.read_csv('.\potential Datasets\Movie-stuff\movies.csv') 


moviesDF.groupby('genre').size() # prints how many of each genre there exists
moviesDF.describe()
# Removing NaN values from the dataframe
moviesDF = moviesDF.dropna()

y = moviesDF[['genre']] # our dependent variable

X = moviesDF[['summary']] # our independant variable


# Code to split the dataframe of summarys into a dataframe of individual words
X_list = X.values.tolist()
y_list = y.values.tolist()

# PLEASE RENAME ME
proper_X_list = []
proper_y_list = []

for i in X_list:
    proper_X_list = proper_X_list + i
    
for i in y_list:
    proper_y_list = proper_y_list + i



# we are splitting our dataset up here for training and later validation.
X_train, X_test, y_train, y_test = train_test_split(proper_X_list,proper_y_list) 

# Encoder to encode our dependant variable
encoder = preprocessing.LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


# [[[[FEATURE ENGINEERING]]]]

# Count Vector is a matrix notation of the dataset in 
# which every row represents a document from the corpus,
# every column represents a term from the corpus, 
# and every cell represents the frequency count of a particular term in a particular document.

# Here we create the count vectorized object
# analyzer=word means that we chose to create an n-gram over words compared to chars or char_wb which is a special n-gram
# when using analyzer=word we can choose a token pattern which is decided in a regular expression 
# in this case the '\w{1,}' means that it will match a word with at least 1 character length.
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english')

count_vect.fit(X_train)



# Now we are gonna transform the training and test data with our vectorized object

X_train_count = count_vect.transform(X_train)
X_test_count = count_vect.transform(X_test)

# Now we are gonna use Term Frequency - Inverted Document Frequence (TF-IDF) vectors as features
# The score generate by the TF-IDF represents the relatuve importance of a term in a document and the entire corpus.
# We generate this score in two steps:
# The first computes the normalizeds term frequency (Tf) --- TF(x) = Number of times x appears in the document / total bynber if terms in the document. 
# the second computes the inverse document frequency  (IDF) --- IDF(x) = log_e(total number of documents / number of documents with term x in it)
# as mentioned earlier we could have chosen to use an n-gram composed of words, which we have implemented in line 61.
# now we are creating the TF-IDF score based on that n-gram.

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', max_features=5000)
tfidf_vect.fit(X_train)
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)


# [[[NAIVE BAYES]]]

label = ConvertColumnArrayToNormalArray(y_train)
accuracy = train_model(classifier = naive_bayes.MultinomialNB(),
                       feature_vector_train = X_train_count,
                       label = label,
                       feature_vector_valid = X_test_count)
print(f'NB, Count vectors: {accuracy}')

accuracy = train_model(classifier = naive_bayes.MultinomialNB(),
                       feature_vector_train = X_train_tfidf,
                       label = label,
                       feature_vector_valid = X_test_tfidf)
print(f'NB, tf-idf vectors: {accuracy}')

accuracy = train_model(linear_model.LogisticRegression(), X_train_count, label, X_test_count)
print ("LR, Count Vectors: ", accuracy)

accuracy = train_model(linear_model.LogisticRegression(), X_train_tfidf, label, X_test_tfidf)
print ("LR, tf-idf Vectors: ", accuracy)

accuracy = train_model(classifier = svm.SVC(),
                       feature_vector_train = X_train_count,
                       label = label,
                       feature_vector_valid = X_test_count)
print(f'SVM, Count vectors: {accuracy}')

accuracy = train_model(classifier = svm.SVC(),
                       feature_vector_train = X_train_tfidf,
                       label = label,
                       feature_vector_valid = X_test_tfidf)
print(f'SVM, tf-idf vectors: {accuracy}')

accuracy = train_model(classifier = ensemble.RandomForestClassifier(),
                       feature_vector_train = X_train_count,
                       label = label,
                       feature_vector_valid = X_test_count)
print(f'RF, Count vectors: {accuracy}')

accuracy = train_model(classifier = ensemble.RandomForestClassifier(),
                       feature_vector_train = X_train_tfidf,
                       label = label,
                       feature_vector_valid = X_test_tfidf)
print(f'RF, tf-idf vectors: {accuracy}')




























