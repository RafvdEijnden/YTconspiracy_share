### Code for the research of the masters thesis of Raf van den Eijnden

# this code has a few big stages:
#
# 1. loading in data and stratify-shuffle it, then cleaning the data
#
# 2. Create two keyword lists using my keywordextractor function
#
# 3. TF-IDF Vectorization of the dats using scikit learns TfidfVectorizer function.
#
# 4. Calculate four bias-matrices based on similarity between each word in the TF-IDF vocab and the keywords in the keywordlist
#    (word2vec & keyword list 1, word2vec & Keyword list 2, GloVe & Keyword list 1, and finally GloVe & Keyword list 2)
#
# 5. Apply bias matrix vector to baseline TF-IDF vector to create 4 new biased vectors
#
# 6. Training en testing with a SVM Classifier


from my_FunctionsModule_ForThesis import keywordextractor
from my_FunctionsModule_ForThesis import cleaner
from my_FunctionsModule_ForThesis import dataset_shuffler
import numpy as np

# use my function dataset_shuffler to get a test_data and a train_dataset which are shuffled and stratified
# # when using te combined JSONS, please replace this function with a shuffler of your own
test_data, training_data = dataset_shuffler('D:/School/CSAI/Thesis/Data Exploration Project','D:/School/CSAI/Thesis/Data Exploration Project','D:/School/CSAI/Thesis/Dataset','D:/School/CSAI/Thesis/Dataset/Transcripts')


# load the nlp here so you don't have to do that later
import en_core_web_sm

nlp = en_core_web_sm.load(disable=["parser", "tagger", "ner"])

# clean all data and pickle them
training_data_cleaned = cleaner(training_data,nlp,'D:/School/CSAI/Thesis/Data Exploration Project','training_data_cleaned')
test_data_cleaned = cleaner(test_data,nlp,'D:/School/CSAI/Thesis/Data Exploration Project','test_data_cleaned')

print(len(training_data_cleaned))
print(len(test_data_cleaned))

# we determine some hyperparameters here for the rest of the research, here we can change them to see the influence of certain model parameters
# as wel as set some key variables
keyword_number = 50
filepath = 'D:/School/CSAI/Thesis/Data Exploration Project'
max_feats = 7000
SVM_C = 1

print('Data loaded and cleaned')
print()

# # # # KEYWORD EXTRACTION creates two pickle files of keyword top keywords
keywordextractor('D:/School/CSAI/Thesis/Data Exploration Project','training_data_cleaned.pickle',keyword_number)

import os
os.chdir(filepath)
import pickle
with open(str(keyword_number) + 'keyword_list1.pickle', 'rb') as f:
    keyword_list1 = pickle.load(f)
with open(str(keyword_number) + 'keyword_list2.pickle', 'rb') as f:
    keyword_list2 = pickle.load(f)

print("keywords extracted")

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# for the train set I create X and y
X_trainingdata = []
y_trainingdata = []
for item in training_data_cleaned:
    X_trainingdata.append(item[0])
    y_trainingdata.append(item[1])

# Here I transform X to fit the input of the tfidf vectorizer
list_document_tokens = []
for i, document in enumerate(X_trainingdata):
    list_document_tokens.append(training_data_cleaned[i][0])
tfidf_input = []
for document in list_document_tokens:
    tfidf_input.append(" ".join(document))


# I do the same for my test data
X_testdata = []
y_testdata = []
for item in test_data_cleaned:
    X_testdata.append(item[0])
    y_testdata.append(item[1])

# once again making them fit
list_document_tokens_test = []
for i, document in enumerate(X_testdata):
    list_document_tokens_test.append(test_data_cleaned[i][0])
tfidf_input_test = []
for document in list_document_tokens_test:
    tfidf_input_test.append(" ".join(document))

### TF IDF VECTORIZATION


#for training I fit and transform the train data:
tv_training = TfidfVectorizer(stop_words=None, max_features=max_feats)
tf_idf_prel_training = tv_training.fit_transform(tfidf_input)
tf_idf_vector_training = tf_idf_prel_training.toarray()
tf_idf_feature_names_training = tv_training.get_feature_names()

#for test, using the same vectorizer I only transform the test data
tf_idf_prel_test = tv_training.transform(tfidf_input_test)
tf_idf_vector_test = tf_idf_prel_test.toarray()


### We route this back in X and Y variables (X_val & y_val for the validation stage, X_ts0 & y_tst for the final test set)
#
X_val = tf_idf_vector_training
y_val = []

# Here we condense class 2 & 3 into one class, leaving is with 0 (non conspiratorial) or 1 (conspiratorial)
for document in y_trainingdata:
    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_val.append(class_made)

y_val = np.array(y_val)

X_tst = tf_idf_vector_test
y_tst = []
for document in y_testdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_tst.append(class_made)

y_tst = np.array(y_tst)

print()
print('Features extracted')

### Here we set a baseline for the rest of the research

# Support Vector Machine Classifier Baseline
print()

# train test split for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, test_size=0.2, random_state=0)

# Here we train our clsasifier
from sklearn import svm
classifier1 = svm.SVC(C=SVM_C, kernel='linear', degree=7, gamma='auto')
classifier1.fit(X_train, y_train)

# We validate here
print("SVM validation baseline")
y_pred1 = classifier1.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
print(accuracy_score(y_test, y_pred1))

# here we test the trained model on our Test set
print("SVM test baseline")
y_pred1_test = classifier1.predict(X_tst)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_tst, y_pred1_test))
print(classification_report(y_tst, y_pred1_test))
print(accuracy_score(y_tst, y_pred1_test))

### biasing

# here I start by calculating our bias multiplier matrix
X_bias1 = []

# we need gensims word2vec here
import gensim
import logging

# initialise and train the Word2Vec model
model = gensim.models.Word2Vec(list_document_tokens, size=150, window=10, min_count=0, workers=10, iter=10)

# create a list with bias multipliers for each of the words in the TF-IDF vocab

for w1 in tf_idf_feature_names_training:
    if w1 != 'pron' and w1 != 'to':
        counter = 0
        averager = 0
        for w2 in keyword_list1:
            try:
                counter += model.wv.similarity(w1, w2)
                averager += 1
            except:
                averager += 0
                pass

        counter = counter/averager
        if counter > 0:
            counter += 1
        else:
            counter = counter
        X_bias1.append(counter)
    else:
        counter = 0
        X_bias1.append(counter)

# I now have a matrix containing a multiplier score for each word in the TF-IDF Vocabulary, based on similarity to our keyword lists
# now I multiply that matrix with each row of our TF-IDF vectors in both training and test sets

X_val_biased_1 = []
for i,matrixrow in enumerate(X_val):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias1[j]*item)

    X_val_biased_1.append(temp_num)
import numpy as np
X_val_biased_1 = np.array(X_val_biased_1)

X_test_biased_1 = []
for i,matrixrow in enumerate(X_tst):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias1[j]*item)

    X_test_biased_1.append(temp_num)
import numpy as np
X_test_biased_1 = np.array(X_test_biased_1)


# Now I route this back in X and Y variables
#for training:


X_val_bias1 = X_val_biased_1
y_val_bias1 = []
for document in y_trainingdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_val_bias1.append(class_made)

y_val_bias1 = np.array(y_val_bias1)

# for testing
X_tst_bias1 = X_test_biased_1
y_tst_bias1 = []
for document in y_testdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_tst_bias1.append(class_made)

y_tst_bias1 = np.array(y_tst_bias1)


# train test split for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_val_bias1, y_val_bias1, test_size=0.2, random_state=0)

# Train classifier using bias 1
from sklearn import svm
classifier2 = svm.SVC(C=SVM_C, kernel='linear', degree=7, gamma='auto')
classifier2.fit(X_train, y_train)

# validation
print("SVM validation bias1")
y_pred2 = classifier2.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))

# testing
print("SVM test bias1")
y_pred2_test = classifier2.predict(X_tst_bias1)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_tst_bias1, y_pred2_test))
print(classification_report(y_tst_bias1, y_pred2_test))
print(accuracy_score(y_tst_bias1, y_pred2_test))



# Support Vector Machine Classifier
print()

# now to bias the X for bias 2
X_bias2 = []

import gensim
import logging

# initialise and train the model
model = gensim.models.Word2Vec(list_document_tokens, size=150, window=10, min_count=0, workers=10, iter=10)

# create a list with bias multipliers for each of the words in the TF-IDF vocab, now with wordlist 2

for w1 in tf_idf_feature_names_training:
    if w1 != 'pron' and w1 != 'to':
        counter = 0
        averager = 0
        for w2 in keyword_list2:
            try:
                counter += model.wv.similarity(w1, w2)
                averager += 1
            except:
                averager += 0
                pass

        counter = counter/averager
        if counter > 0:
            counter += 1
        else:
            counter = counter
        X_bias2.append(counter)
    else:
        counter = 0
        X_bias2.append(counter)

X_val_biased_2 = []
for i,matrixrow in enumerate(X_val):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias2[j]*item)

    X_val_biased_2.append(temp_num)
import numpy as np
X_val_biased_2 = np.array(X_val_biased_2)

X_test_biased_2 = []
for i,matrixrow in enumerate(X_tst):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias2[j]*item)

    X_test_biased_2.append(temp_num)
import numpy as np
X_test_biased_2 = np.array(X_test_biased_2)

# create X and Y for training and testing:


X_val_bias2 = X_val_biased_2
y_val_bias2 = []
for document in y_trainingdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_val_bias2.append(class_made)

y_val_bias2 = np.array(y_val_bias2)

X_tst_bias2 = X_test_biased_2
y_tst_bias2 = []
for document in y_testdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_tst_bias2.append(class_made)

y_tst_bias2 = np.array(y_tst_bias2)


# train test split for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_val_bias2, y_val_bias2, test_size=0.2, random_state=0)

# train classifier on biased set 2
from sklearn import svm
classifier3 = svm.SVC(C=SVM_C, kernel='linear', degree=7, gamma='auto')
classifier3.fit(X_train, y_train)

# validation
print("SVM validation bias2")
y_pred3 = classifier3.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred3))
print(classification_report(y_test, y_pred3))
print(accuracy_score(y_test, y_pred3))
print("SVM test bias2")

# testing
y_pred3_test = classifier3.predict(X_tst_bias2)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_tst_bias2, y_pred3_test))
print(classification_report(y_tst_bias2, y_pred3_test))
print(accuracy_score(y_tst_bias2, y_pred3_test))




# Support Vector Machine Classifier
print()

# bias 3

import gensim
import logging
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# load the pretrained file GLove for word similairty
glove_file = datapath('D:/School/CSAI/Thesis/Data Exploration Project/glove.6B.50d/glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile('glove.6B.100d.txt')

glove2word2vec(glove_file,word2vec_glove_file)
# initialise and train the model
model2 = KeyedVectors.load_word2vec_format(word2vec_glove_file)

print('Glove model loaded')

# create biases in the same fashion as before
X_bias3 = []
for w1 in tf_idf_feature_names_training:
    if (w1 != 'pron') and (w1 != 'swinge') and (w1 != 'abaa') and (w1 != 'agendum') and (w1 != 'almubarak')and (w1 != 'arkaim') and (w1 != 'calory') and (w1 != 'froning') and(w1 != 'khalipa') and (w1 != 'spealler')and (w1 != '1000000') and w1 != ('1000000000') and w1 != ('aaaaaa') and (w1 != 'aaagh'):
        counter = 0
        averager = 0
        for w2 in keyword_list1:
            try:
                counter += model2.wv.similarity(w1, w2)
                averager += 1
            except:
                averager += 0
                pass

        try:
            counter = counter/averager
        except:
            counter = counter
        if counter > 0:
            counter += 1
        else:
            counter = counter
        X_bias3.append(counter)
    else:
        counter = 0
        X_bias3.append(counter)

X_val_biased_3 = []
for i,matrixrow in enumerate(X_val):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias3[j]*item)

    X_val_biased_3.append(temp_num)
import numpy as np
X_val_biased_3 = np.array(X_val_biased_3)

X_test_biased_3 = []
for i,matrixrow in enumerate(X_tst):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias3[j]*item)

    X_test_biased_3.append(temp_num)
import numpy as np
X_test_biased_3 = np.array(X_test_biased_3)

# X and y for training and test sets:


X_val_bias3 = X_val_biased_3
y_val_bias3 = []
for document in y_trainingdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_val_bias3.append(class_made)

y_val_bias3 = np.array(y_val_bias3)

X_tst_bias3 = X_test_biased_3
y_tst_bias3 = []
for document in y_testdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_tst_bias3.append(class_made)

y_tst_bias3 = np.array(y_tst_bias3)

# train test split for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_val_bias3, y_val_bias3, test_size=0.2, random_state=0)

# classifier training, validation and testing
from sklearn import svm
classifier4 = svm.SVC(C=SVM_C, kernel='linear', degree=7, gamma='auto')
classifier4.fit(X_train, y_train)
print("SVM validation bias3")
y_pred4 = classifier4.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred4))
print(classification_report(y_test, y_pred4))
print(accuracy_score(y_test, y_pred4))
print("SVM test bias3")
y_pred4_test = classifier4.predict(X_tst_bias3)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_tst_bias3, y_pred4_test))
print(classification_report(y_tst_bias3, y_pred4_test))
print(accuracy_score(y_tst_bias3, y_pred4_test))

# now for the last bias
X_bias4 = []
for w1 in tf_idf_feature_names_training:
    if (w1 != 'pron') and (w1 != 'swinge') and (w1 != 'calory') and (w1 != 'froning') and(w1 != 'khalipa') and (w1 != 'spealler'):
        counter = 0
        averager = 0
        for w2 in keyword_list2:
            try:
                counter += model2.wv.similarity(w1, w2)
                averager += 1
            except:
                averager += 0
                pass


        try:
            counter = counter/averager
        except:
            counter = counter
        if counter > 0:
            counter += 1
        else:
            counter = counter
        X_bias4.append(counter)
    else:
        counter = 0
        X_bias4.append(counter)

X_val_biased_4 = []
for i,matrixrow in enumerate(X_val):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias4[j]*item)

    X_val_biased_4.append(temp_num)
import numpy as np
X_val_biased_4 = np.array(X_val_biased_4)

X_test_biased_4 = []
for i,matrixrow in enumerate(X_tst):
    temp_num = []
    for j, item in enumerate(matrixrow):

        temp_num.append(X_bias4[j]*item)

    X_test_biased_4.append(temp_num)
import numpy as np
X_test_biased_4 = np.array(X_test_biased_4)

#for training:


X_val_bias4 = X_val_biased_4
y_val_bias4 = []
for document in y_trainingdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_val_bias4.append(class_made)

y_val_bias4 = np.array(y_val_bias4)

X_tst_bias4 = X_test_biased_4
y_tst_bias4 = []
for document in y_testdata:

    class_made = 0
    if document == 1:
        class_made = 0
    else:
        class_made = 1
    y_tst_bias4.append(class_made)

y_tst_bias4 = np.array(y_tst_bias4)

# train test split for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_val_bias4, y_val_bias4, test_size=0.2, random_state=0)

from sklearn import svm
classifier5 = svm.SVC(C=SVM_C, kernel='linear', degree=7, gamma='auto')
classifier5.fit(X_train, y_train)
print("SVM validation bias4")
y_pred5 = classifier5.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred5))
print(classification_report(y_test, y_pred5))
print(accuracy_score(y_test, y_pred5))
print("SVM test bias4")
y_pred5_test = classifier5.predict(X_tst_bias4)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_tst_bias4, y_pred5_test))
print(classification_report(y_tst_bias4, y_pred5_test))
print(accuracy_score(y_tst_bias4, y_pred5_test))


# this gives us and overview of performance of various classifiers. Validation scores will be used for hyperparameter tweaking
# then test scores to rapport differences between the biases and keywords versus the baseline





#
#
# print()
# print("NB validation baseline")
#
# # train test split for validation
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, test_size=0.2, random_state=0)
#
# # naive Bayes classifier
# from sklearn import naive_bayes
# classifier1_NB = naive_bayes.GaussianNB()
# classifier1_NB.fit(X_train, y_train)
#
# y_pred1_NB = classifier1_NB.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred1_NB))
# print(classification_report(y_test,y_pred1_NB))
# print(accuracy_score(y_test, y_pred1_NB))
#
# print()
# print("NB Test baseline")
#
# y_pred1_NB_test = classifier1_NB.predict(X_tst)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_tst,y_pred1_NB_test))
# print(classification_report(y_tst,y_pred1_NB_test))
# print(accuracy_score(y_tst, y_pred1_NB_test))
#
# print()
# print("NB validation bias1")
#
# # train test split for validation
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_val_bias1, y_val_bias1, test_size=0.2, random_state=0)
#
# # naive Bayes classifier
# from sklearn import naive_bayes
# classifier2_NB = naive_bayes.GaussianNB()
# classifier2_NB.fit(X_train, y_train)
#
# y_pred2_NB = classifier2_NB.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred1_NB))
# print(classification_report(y_test,y_pred1_NB))
# print(accuracy_score(y_test, y_pred1_NB))
#
# print()
# print("NB Test bias1")
#
# y_pred2_NB_test = classifier1_NB.predict(X_tst_bias1)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_tst_bias1,y_pred2_NB_test))
# print(classification_report(y_tst_bias1,y_pred2_NB_test))
# print(accuracy_score(y_tst_bias1, y_pred2_NB_test))
#
# print()
# print("NB validation bias2")
#
# # train test split for validation
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_val_bias2, y_val_bias2, test_size=0.2, random_state=0)
#
# # naive Bayes classifier
# from sklearn import naive_bayes
# classifier3_NB = naive_bayes.GaussianNB()
# classifier3_NB.fit(X_train, y_train)
#
# y_pred3_NB = classifier3_NB.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred3_NB))
# print(classification_report(y_test,y_pred3_NB))
# print(accuracy_score(y_test, y_pred3_NB))
#
# print()
# print("NB Test bias2")
#
# y_pred3_NB_test = classifier1_NB.predict(X_tst_bias2)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_tst_bias2,y_pred3_NB_test))
# print(classification_report(y_tst_bias2,y_pred3_NB_test))
# print(accuracy_score(y_tst_bias2, y_pred3_NB_test))
#
# print()
# print("NB validation bias3")
#
# # train test split for validation
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_val_bias3, y_val_bias3, test_size=0.2, random_state=0)
#
# # naive Bayes classifier
# from sklearn import naive_bayes
# classifier4_NB = naive_bayes.GaussianNB()
# classifier4_NB.fit(X_train, y_train)
#
# y_pred4_NB = classifier4_NB.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred4_NB))
# print(classification_report(y_test,y_pred4_NB))
# print(accuracy_score(y_test, y_pred4_NB))
#
# print()
# print("NB Test bias3")
#
# y_pred4_NB_test = classifier4_NB.predict(X_tst_bias3)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_tst_bias3,y_pred4_NB_test))
# print(classification_report(y_tst_bias3,y_pred4_NB_test))
# print(accuracy_score(y_tst_bias3, y_pred4_NB_test))
#
# print()
# print("NB validation bias4")
#
# # train test split for validation
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_val_bias4, y_val_bias4, test_size=0.2, random_state=0)
#
# # naive Bayes classifier
# from sklearn import naive_bayes
# classifier5_NB = naive_bayes.GaussianNB()
# classifier5_NB.fit(X_train, y_train)
#
# y_pred5_NB = classifier5_NB.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred5_NB))
# print(classification_report(y_test,y_pred5_NB))
# print(accuracy_score(y_test, y_pred5_NB))
#
# print()
# print("NB Test bias4")
#
# y_pred5_NB_test = classifier5_NB.predict(X_tst_bias4)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_tst_bias4,y_pred5_NB_test))
# print(classification_report(y_tst_bias4,y_pred5_NB_test))
# print(accuracy_score(y_tst_bias4, y_pred5_NB_test))