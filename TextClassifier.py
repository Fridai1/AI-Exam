import pandas as pd
import re
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# Modelling
from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
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


# HELPER METHODS


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    
    classifier.fit(feature_vector_train,label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)


    return metrics.accuracy_score(y_test, predictions)

def ConvertColumnArrayToNormalArray(array):
    newArray = []
    for i in array:
        newArray.append(i)
    return newArray

def TransformGenre(array):
    genreArray = [[],[]]
    failedIndexes = []
    counter = 0
    while counter < len(array):
        try:
            genreObject = array[counter]
            genreStr = genreObject[0]            
            genreStr = genreStr.replace('[', '')
            genreStr = genreStr.replace(']','')
            splitStr = genreStr.split('}, {')
            genreArray.append(counter)
            splitStr[0] = splitStr[0].replace('{', '')
            length = len(splitStr)
            splitStr[length-1] = splitStr[length-1].replace('}', '')
            for i in splitStr:
                i = "{"+i+"}"
                jsonobj = json.loads(i)
                genre = jsonobj["name"]
                genreArray[counter].append(genre)                      
            counter += 1 
        except:
            failedIndexes.sort()
            failedIndexes = failedIndexes[::-1] # reversing the list
            counter += 1
    return genreArray, failedIndexes
            
     
    

# [[[[DATASET PREPARATION]]]]

moviesDF = pd.read_csv('tmdb_movies.csv') 

# RETURN TO THIS LATER
# moviesDF.groupby('genres').size() # prints how many of each genre there exists


y = moviesDF[['genres']] # our dependent variable

X = moviesDF[['overview']] # our independant variable



X_list = X.values.tolist()
y_list = y.values.tolist()

#result, failed = TransformGenre(y_list)

y_ = [] # temp, array for genres that arent empty
failedIndexes = [] # array to keep track of the index we had an empty genre, to be used later in deletion.

counter = 0
# we iterate over y_list to find all genres that arent empty, if one is empty its gonna trigger an exception
# which triggers our except clause. The except clause saved the index the error empty genre was located and progresses the counter
while counter < len(y_list):
    try:
        genreObject = y_list[counter]
        genreStr = genreObject[0]
        strstr = genreStr.split()
        indexedstr = strstr[3]
        test = re.findall(r'\w+', indexedstr)    
        y_.append(test[0])
        counter += 1 
    except:
        failedIndexes.append(counter)
        counter += 1
        
failedIndexes.sort()
failedIndexes = failedIndexes[::-1] # reversing the list
# as it turns out when you delete an index from a python list, it collapses the list, so we had to delete the highest index first to circumvent this.
for index in failedIndexes:
    del X_list[index]
    
y_list = y_ 

X_ =  [] # temp list to contain all strings
failedIndexes = []
counter = 0
# Currently X_list contains a collection of collections, this kinda og list-ception is incompatible with the AI algorithm
# So we create this small loop to extract the string.
while counter < len(X_list):
    listLine = X_list[counter]
    listString = listLine[0]
    if not listString:
        failedIndexes.append(counter)
        counter += 1
        continue
    X_.append(listString)
    counter += 1
    
failedIndexes = failedIndexes[::-1]
for i in failedIndexes:
    del X_[i]
    
X_list = X_

# We had nan values in our X_list, we decided to convert the X_list back to a dataframe
# in order to run isnull(), which returns a list of false/true wether an entry is nan or not
# with this we simply saved the index of which the nan occured and deleted
# it from both our dependant and independant variable.

tempDf = pd.DataFrame(X_list)
boolList = tempDf.isnull().values

badIndexes = []
counter = 0
while counter < len(boolList):
    if boolList[counter][0]:
        badIndexes.append(counter)               
    counter += 1

badIndexes = badIndexes[::-1]

for i in badIndexes:
    del X_list[i]
    del y_list[i]
    
        

# we are splitting our dataset up here for training and later validation.
X_train, X_test, y_train, y_test = train_test_split(X_list,y_list) 

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

stop_words = set(stopwords.words('english'))

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stop_words, max_features=5000)

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

#tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', max_features=5000)
tfidf_vect = TfidfVectorizer(encoding='utf-8',lowercase=True, stop_words=stop_words, sublinear_tf=True, use_idf=True,max_features=5000)
tfidf_vect.fit(X_train)
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3),stop_words=stop_words,max_features=5000,lowercase=True)
tfidf_vect_ngram.fit(X_train)
X_train_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)



# [[[Training and testing]]]

nb = naive_bayes.MultinomialNB()

label = ConvertColumnArrayToNormalArray(y_train)
accuracy = train_model(classifier = nb,
                       feature_vector_train = X_train_count,
                       label = label,
                       feature_vector_valid = X_test_count)
print(f'NB, Count vectors: {accuracy}')

accuracy = train_model(classifier = naive_bayes.MultinomialNB(),
                       feature_vector_train = X_train_tfidf,
                       label = label,
                       feature_vector_valid = X_test_tfidf)
print(f'NB, tf-idf vectors: {accuracy}')


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

# =============================================================================
# y_arr = np.asarray(label)
# y_arr.flatten()
# yvonne = y_arr.astype(np.int32)
# classifier = create_model_architecture(X_train_tfidf_ngram.shape[1])
# accuracy = train_model(classifier = classifier,
#                        feature_vector_train = X_train_tfidf_ngram,
#                        label = yvonne,
#                        feature_vector_valid = X_test_tfidf_ngram,
#                        is_neural_net=True)
# print ("NN, Ngram Level TF IDF Vectors",  accuracy)
# =============================================================================

X_train_tokens = count_vect.get_feature_names()
print(X_train_tokens[0:50])
print(X_train_tokens[-50:])
nb.feature_count_
nb.feature_count_.shape



Action_token_count = nb.feature_count_[0, :]
Adventure_token_count = nb.feature_count_[1, :]
Animation_token_count = nb.feature_count_[2, :]
Comedy_token_count = nb.feature_count_[3, :]
Crime_token_count = nb.feature_count_[4, :]
Documentary_token_count = nb.feature_count_[5, :]
Drama_token_count = nb.feature_count_[6, :]
Family_token_count = nb.feature_count_[7, :]
Fantasy_token_count = nb.feature_count_[8, :]
Foreign_token_count = nb.feature_count_[9, :]
History_token_count = nb.feature_count_[10, :]
Horror_token_count = nb.feature_count_[11, :]
Music_token_count = nb.feature_count_[12, :]
Mystery_token_count = nb.feature_count_[13, :]
Romance_token_count = nb.feature_count_[14, :]
Science_token_count = nb.feature_count_[15, :]
TV_token_count = nb.feature_count_[16, :]
Thriller_token_count = nb.feature_count_[17, :]
War_token_count = nb.feature_count_[18, :]
Western_token_count = nb.feature_count_[19, :]


tokens = pd.DataFrame({'token':X_train_tokens, 'Action':Action_token_count,
                       'Adventure':Adventure_token_count,
                       'Animation':Animation_token_count,
                       'Comedy':Comedy_token_count,
                       'Crime':Crime_token_count,
                       'Documentary':Documentary_token_count,
                       'Drama':Drama_token_count,
                       'Family':Family_token_count,
                       'Fantasy':Fantasy_token_count,
                       'Foreign':Foreign_token_count,
                       'History':History_token_count,
                       'Horror':Horror_token_count,
                       'Music':Music_token_count,
                       'Mystery':Mystery_token_count,
                       'Romance':Romance_token_count,
                       'Science':Science_token_count,
                       'TV':TV_token_count,
                       'Thriller':Thriller_token_count,
                       'War':War_token_count,
                       'Western':Western_token_count
                       }).set_index('token')
tokens.head()
inversed = encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

classcount = nb.class_count_