def TranscriptExtractor(directory_youtubecsv, directory_transcripts):
    "This Function extracts a list from the dataset with the following string contents: [Video_ID, Video_Category, Video_Transcript, Video_Rating]"
    import os
    import csv
    import json
    # first we load in all info from the youtube.csv we need
    os.chdir(directory_youtubecsv)
    with open('YouTube.csv', mode='r') as infile:
        reader = csv.reader(infile)
        youtube_ranked = []
        for rows in reader:
            youtube_ranked.append(rows)
        youtube_ranked_data = {}

    for listed in youtube_ranked:
        youtube_ranked_data.update({listed[1][32:]: listed[5]})

    ## now for the transcript data
    id_cat_transcripts = []

    os.chdir(directory_transcripts)

    # firearms
    with open('firearms_transcript.json') as f:
        data_transcript_firearms = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for firearms
    id_transcripts_firearms = list(data_transcript_firearms.keys())
    for id in id_transcripts_firearms:
        id_cat_transcripts.append([id, "firearms", data_transcript_firearms[id], youtube_ranked_data[id]])

    # fitness
    with open('fitness_transcript.json') as f:
        data_transcript_fitness = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_fitness = list(data_transcript_fitness.keys())
    for id in id_transcripts_fitness:
        id_cat_transcripts.append([id, "fitness", data_transcript_fitness[id], youtube_ranked_data[id]])

    # gurus
    with open('gurus_transcript.json') as f:
        data_transcript_gurus = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_gurus = list(data_transcript_gurus.keys())
    for id in id_transcripts_gurus:
        id_cat_transcripts.append([id, "gurus", data_transcript_gurus[id], youtube_ranked_data[id]])

    # martial_arts
    with open('martial_arts_transcript.json') as f:
        data_transcript_martial_arts = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_martial_arts = list(data_transcript_martial_arts.keys())
    for id in id_transcripts_martial_arts:
        id_cat_transcripts.append([id, "martial_arts", data_transcript_martial_arts[id], youtube_ranked_data[id]])

    # natural foods
    with open('natural_foods_transcript.json') as f:
        data_transcript_natural_foods = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_natural_foods = list(data_transcript_natural_foods.keys())
    for id in id_transcripts_natural_foods:
        id_cat_transcripts.append([id, "natural_foods", data_transcript_natural_foods[id], youtube_ranked_data[id]])

    # tiny houses
    with open('tiny_houses_transcrip.json') as f:
        data_transcript_tiny_houses = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_tiny_houses = list(data_transcript_tiny_houses.keys())
    for id in id_transcripts_tiny_houses:
        id_cat_transcripts.append([id, "tiny_houses", data_transcript_tiny_houses[id], youtube_ranked_data[id]])

    ## we now have a full set of 600 entries containing all video ID's, categories, transcripts and rankings
    # print(id_cat_transcripts)

    # NOW! let's remove all empty transcripts
    id_cat_transcripts_emptycleaned = []

    for unit in id_cat_transcripts:
        if unit[2] == '':
            continue
        else:
            id_cat_transcripts_emptycleaned.append(unit)

    return id_cat_transcripts_emptycleaned

def keywordextractor(filepath,filename,keywords_num):

    """ this function creates two picklefiles, each has a keyword list
    first parameter; filepath = the path where you want the pickles to go, and the path where your transcript data is
    filename: name of the pickle: for example; 'cleaned_transcript_data.pickle'
    second parameter; keywords_num = amount of keywords you want in your list"""

    import os
    import pickle
    import numpy as np

    os.chdir(filepath)
    with open(filename, 'rb') as f:
        transcripts_cleaned = pickle.load(f)


    keywords_number = keywords_num
    ### taking only transcripts
    list_document_tokens = []
    for i, document in enumerate(transcripts_cleaned):
        list_document_tokens.append(transcripts_cleaned[i][0])

    # create a list of documents to input to tfidfvectorizer
    tfidf_input = []
    for document in list_document_tokens:
        tfidf_input.append(" ".join(document))

    ### split it by class
    list_document_tokens_consp = []
    list_document_tokens_nonconsp = []
    for i, document in enumerate(transcripts_cleaned):
        if document[1] == '3':

            list_document_tokens_consp.append(transcripts_cleaned[i][0])
        else:
            list_document_tokens_nonconsp.append(transcripts_cleaned[i][0])

    tfidf_input_consp = []
    tfidf_input_nonconsp = []
    for document in list_document_tokens_consp:
        tfidf_input_consp.append(" ".join(document))
    for document in list_document_tokens_nonconsp:
        tfidf_input_nonconsp.append(" ".join(document))

    # now for keyword extraction method 1

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    tv = TfidfVectorizer(stop_words=None, max_features=10000)

    vocab = tv.fit(tfidf_input)
    feature_names = vocab.get_feature_names()

    word_count_vector = tv.fit_transform(tfidf_input).toarray()

    word_count_vector_transposed = word_count_vector.T

    total_word_info = []
    for i, word in enumerate(word_count_vector_transposed):
        tempword = word.tolist()
        word_info = []

        for j, document in enumerate(tempword):
            word_info.append([j, document, transcripts_cleaned[j][1]])
        total_word_info.append(word_info)

    tf_sum_consp = 0
    tf_sum_nonconsp = 0
    sum_array = []

    for i, word_info in enumerate(total_word_info):
        tf_sum_consp = 0
        tf_sum_nonconsp = 0
        tf_sum_delta = 0

        for array in word_info:
            boolchecker = array[2]

            if boolchecker == 1:

                value = array[1]
                tf_sum_nonconsp += value

            else:
                value = array[1]

                tf_sum_consp += value

        tf_sum_delta = tf_sum_nonconsp - tf_sum_consp

        sum_array.append([feature_names[i], tf_sum_delta])

    deltas = []
    for item in sum_array:
        deltas.append(item[1])

    deltas = np.array(deltas)
    indices = deltas.argsort()[:keywords_number]

    keywords_1 = [sum_array[i] for i in indices]

    keyword_list1 = []
    for i in keywords_1:
        keyword_list1.append(i[0])

    print("there are this many keywords in list1: ", len(keyword_list1))
    # we pickle it for posterity
    os.chdir(filepath)

    with open(str(keywords_num)+'keyword_list1.pickle', 'wb') as f:
        pickle.dump(keyword_list1, f)

    ### now for keyword extraction method 2

    # method two runs a basic pipeline with a SVM then finds most distinguishing features

    os.chdir(filepath)
    with open(filename, 'rb') as f:
        transcripts_cleaned = pickle.load(f)
    print("fully loaded")

    list_document_tokens = []
    for i, document in enumerate(transcripts_cleaned):
        list_document_tokens.append(transcripts_cleaned[i][0])

    tfidf_input = []
    for document in list_document_tokens:
        tfidf_input.append(" ".join(document))

    # now for feature extraction

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    tv = TfidfVectorizer(stop_words=None, max_features=10000)
    word_count_vector = tv.fit_transform(tfidf_input)

    tf_idf_vector = tv.fit_transform(tfidf_input).toarray()

    # create X and y

    X = tf_idf_vector
    y = []

    # merge categories
    for document in transcripts_cleaned:

        class_made = 0
        if document[1] == 3:
            class_made = 1
        else:
            class_made = 0
        y.append(class_made)

    # train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Support Vector Machine Classifier
    from sklearn import svm
    classifier3 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    classifier3.fit(X_train, y_train)

    y_pred3 = classifier3.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


    coef = classifier3.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-int((keywords_number)):]
    top_negative_coefficients = np.argsort(coef)[:int(keywords_number)]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    feature_names = np.array(tv.get_feature_names())
    keyword_list2 = feature_names[top_positive_coefficients]

    print(keyword_list1)
    print(keyword_list2)

    # we pickle it for posterity
    os.chdir(filepath)

    with open(str(keywords_num)+'keyword_list2.pickle', 'wb') as f:
        pickle.dump(keyword_list2, f)

    print("finished extracting keywords")

def bias_implementer(filepath,keyword_number,max_feats):
    """
    This function creates 4 pickle files containing biased data:
    bias1: first keywords list with word2vec
    bias2: second keywords list with word2vec
    bias3: first keywords list with Glove
    bias4: second keywords list with Glove
    :param filepath: path to your files
    :param keywords_num: number of keywords used
    :param max_feats: maximum features of tfidf vectorizer

    """
    print("starting bias implementation")
    import pickle
    import os
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    # specify the amount of keywords in the list made before:
    keywords_num = keyword_number

    # first we import our keyword lists

    os.chdir(filepath)
    with open(str(keywords_num) + 'keyword_list1.pickle', 'rb') as f:
        keyword_list1 = pickle.load(f)

    with open(str(keywords_num) + 'keyword_list2.pickle', 'rb') as f:
        keyword_list2 = pickle.load(f)

    # we also load our data (ID, Category, transcripts, rating)

    with open('training_data_cleaned.pickle', 'rb') as f:
        training_data_cleaned = pickle.load(f)

    with open('test_data_cleaned.pickle', 'rb') as f:
        test_data_cleaned = pickle.load(f)

    # we now need to get TF-IDF vectors from each transcript so we can later manipulate those with a bias

    # for train set
    list_document_tokens = []
    for i, document in enumerate(training_data_cleaned):
        list_document_tokens.append(training_data_cleaned[i][0])

    tfidf_input = []
    for document in list_document_tokens:
        tfidf_input.append(" ".join(document))

    # for test set
    list_document_tokens_test = []
    for i, document in enumerate(test_data_cleaned):
        list_document_tokens_test.append(test_data_cleaned[i][0])

    tfidf_input_test = []
    for document in list_document_tokens_test:
        tfidf_input_test.append(" ".join(document))

    # for Mark's dataset
    tv = TfidfVectorizer(stop_words=None, max_features=max_feats)
    word_count_vector = tv.fit_transform(tfidf_input)
    tf_idf_prel = tv.fit_transform(tfidf_input)
    tf_idf_vector = tf_idf_prel.toarray()
    tf_idf_feature_names = tv.get_feature_names()

    # for ours
    tv_test = TfidfVectorizer(stop_words=None, max_features=max_feats)
    word_count_vector_test = tv_test.fit_transform(tfidf_input_test)
    tf_idf_prel_test = tv_test.fit_transform(tfidf_input_test)
    tf_idf_vector_test = tf_idf_prel_test.toarray()
    tf_idf_feature_names_test = tv_test.get_feature_names()

    #
    with open('tf_idf_vector_training.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector, f)

    with open('tf_idf_vector_test.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_test, f)
    #
    print(len(tf_idf_vector_test))
    # so we have a list with the 10000 feature names which we need to compare in word similarity to the two keyword lists seperately
    # This shall be done using word2vec first

    import gensim
    import logging

    # initialise and train the model
    model = gensim.models.Word2Vec(list_document_tokens, size=150, window=10, min_count=0, workers=10, iter=10)

    # create a list with bias multipliers for each of the words in the TF-IDF vocab
    tf_idf_bias1 = []
    for w1 in tf_idf_feature_names:
        counter = 0
        averager = 0
        for w2 in keyword_list1:
            try:
                counter += model.wv.similarity(w1, w2)
                averager += 1
            except:
                averager -= 1
                pass
        counter = counter/averager
        tf_idf_bias1.append(counter)

    # now calculate the bias for each of the tfidf vectors

    tf_idf_vector_biased1 = np.zeros(shape=(339, max_feats))
    tf_idf_vector_biased1_test = np.zeros(shape=(43, max_feats))

    for i, document in enumerate(tf_idf_vector):
        tf_idf_vector_biased1[i] = document * tf_idf_bias1

    for i, document in enumerate(tf_idf_vector_test):
        tf_idf_vector_biased1_test[i] = document * tf_idf_bias1

    with open(str(keyword_number) + 'biasedvector1.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased1, f)
    with open(str(keyword_number) + 'biasedvector1test.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased1_test, f)


    print("Pickled bias 1")

    # create a list with bias multipliers for each of the 10000 words in the TF-IDF vocab
    tf_idf_bias2 = []
    for w1 in tf_idf_feature_names:
        counter = 0
        averager = 0
        for w2 in keyword_list2:
            try:
                counter += model.wv.similarity(w1, w2)
                averager += 1
            except:
                averager -= 1
                pass
        counter = counter
        tf_idf_bias2.append(counter)

    # now calculate the bias for each of the tfidf vectors

    tf_idf_vector_biased2 = np.zeros(shape=(339, max_feats))
    tf_idf_vector_biased2_test = np.zeros(shape=(43, max_feats))

    for i, document in enumerate(tf_idf_vector):
        tf_idf_vector_biased2[i] = document * tf_idf_bias2

    for i, document in enumerate(tf_idf_vector_test):
        tf_idf_vector_biased2_test[i] = document * tf_idf_bias2

    with open(str(keyword_number) + 'biasedvector2.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased2, f)
    with open(str(keyword_number) + 'biasedvector2test.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased2_test, f)
    print("Pickled bias 2")
    ## now we do this with GloVe, which is pretrained

    import gensim
    import logging
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

    # load the pretrained file
    glove_file = datapath(filepath + '/glove.6B.50d/glove.6B.100d.txt')
    word2vec_glove_file = get_tmpfile('glove.6B.100d.txt')
    glove2word2vec(glove_file, word2vec_glove_file)
    # initialise and train the model
    model2 = KeyedVectors.load_word2vec_format(word2vec_glove_file)

    # create a list with bias multipliers for each of the 10000 words in the TF-IDF vocab
    tf_idf_bias3 = []
    for w1 in tf_idf_feature_names:
        counter = 0
        averager = 0
        for w2 in keyword_list1:
            try:
                counter += model2.wv.similarity(w1, w2)
                averager += 1
            except:
                averager -= 1
                pass
        counter = counter
        tf_idf_bias3.append(counter)

    # now calculate the bias for each of the tfidf vectors

    tf_idf_vector_biased3 = np.zeros(shape=(339, max_feats))
    tf_idf_vector_biased3_test = np.zeros(shape=(43, max_feats))
    for i, document in enumerate(tf_idf_vector):
        tf_idf_vector_biased3[i] = document * tf_idf_bias3
    for i, document in enumerate(tf_idf_vector_test):
        tf_idf_vector_biased3_test[i] = document * tf_idf_bias3

    with open(str(keyword_number) + 'biasedvector3.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased3, f)
    with open(str(keyword_number) + 'biasedvector3test.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased3_test, f)
    print("Pickled bias 3")
    # create a list with bias multipliers for each of the 10000 words in the TF-IDF vocab
    tf_idf_bias4 = []
    for w1 in tf_idf_feature_names:
        counter = 0
        averager = 0
        for w2 in keyword_list2:
            try:
                counter += model2.wv.similarity(w1, w2)
                averager += 1
            except:
                averager -= 1
                pass
        counter = counter
        tf_idf_bias4.append(counter)

    # now calculate the bias for each of the tfidf vectors

    tf_idf_vector_biased4 = np.zeros(shape=(339, max_feats))
    tf_idf_vector_biased4_test = np.zeros(shape=(43, max_feats))

    for i, document in enumerate(tf_idf_vector):
        tf_idf_vector_biased4[i] = document * tf_idf_bias4

    for i, document in enumerate(tf_idf_vector_test):
        tf_idf_vector_biased4_test[i] = document * tf_idf_bias4

    with open(str(keyword_number) + 'biasedvector4.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased4, f)
    with open(str(keyword_number) + 'biasedvector4test.pickle', 'wb') as f:
        pickle.dump(tf_idf_vector_biased4_test, f)

    print("Pickled bias 4")
    print()
    print("finished biasing")


def model_testing(filepath, keywords_num):
    "starting model testing"
    import os
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn import naive_bayes

    keywords_number = keywords_num

    os.chdir(filepath)
    with open('training_data_cleaned.pickle', 'rb') as f:
        transcripts_cleaned = pickle.load(f)

    with open('test_data_cleaned.pickle', 'rb') as f:
        ourwatchedset_cleaned = pickle.load(f)

    with open('tf_idf_vector_training.pickle', 'rb') as f:
        tf_idf_vector = pickle.load(f)

    with open('tf_idf_vector_test.pickle', 'rb') as f:
        tf_idf_vector_test = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector1.pickle', 'rb') as f:
        tf_idf_vector_biased1 = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector2.pickle', 'rb') as f:
        tf_idf_vector_biased2 = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector3.pickle', 'rb') as f:
        tf_idf_vector_biased3 = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector4.pickle', 'rb') as f:
        tf_idf_vector_biased4 = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector1test.pickle', 'rb') as f:
        tf_idf_vector_biased1_test = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector2test.pickle', 'rb') as f:
        tf_idf_vector_biased2_test = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector3test.pickle', 'rb') as f:
        tf_idf_vector_biased3_test = pickle.load(f)

    with open(str(keywords_number) + 'biasedvector4test.pickle', 'rb') as f:
        tf_idf_vector_biased4_test = pickle.load(f)

    print("fully loaded")

    # create X and y

    X = tf_idf_vector
    y = []

    X_two = tf_idf_vector_test
    y_two = []

    # merge categories
    for document in transcripts_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y.append(class_made)

    for document in ourwatchedset_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_two.append(class_made)

    # train test split for validation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print()
    print("SVM validation baseline")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier1 = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier1.fit(X_train, y_train)

    y_pred1 = classifier1.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test, y_pred1))
    print(classification_report(y_test, y_pred1))
    print(accuracy_score(y_test, y_pred1))

    print()
    print("SVM test baseline")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier1_pred = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier1_pred.fit(X, y)

    y_pred1_test = classifier1_pred.predict(X_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_two, y_pred1_test))
    print(classification_report(y_two, y_pred1_test))
    print(accuracy_score(y_two, y_pred1_test))

    print()
    print("NB validation baseline")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier1_NB = naive_bayes.GaussianNB()
    classifier1_NB.fit(X_train, y_train)

    y_pred1_NB = classifier1_NB.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test, y_pred1_NB))
    print(classification_report(y_test, y_pred1_NB))
    print(accuracy_score(y_test, y_pred1_NB))

    print()
    print("NB Test baseline")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier1_NB_test = naive_bayes.GaussianNB()
    classifier1_NB_test.fit(X, y)

    y_pred1_NB_test = classifier1_NB_test.predict(X_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_two, y_pred1_NB_test))
    print(classification_report(y_two, y_pred1_NB_test))
    print(accuracy_score(y_two, y_pred1_NB_test))

    X_2 = tf_idf_vector_biased1
    y_2 = []

    X_2_two = tf_idf_vector_biased1_test
    y_2_two = []

    # merge categories
    for document in transcripts_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_2.append(class_made)

    for document in ourwatchedset_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_2_two.append(class_made)

    # train test split for validation
    from sklearn.model_selection import train_test_split
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=0)

    print()
    print("SVM validation keywords1 word2vec")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier2 = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier2.fit(X_train_2, y_train_2)

    y_pred2 = classifier2.predict(X_test_2)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_2, y_pred2))
    print(classification_report(y_test_2, y_pred2))
    print(accuracy_score(y_test_2, y_pred2))

    print()
    print("SVM test keywords 1 word2vec")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier2_pred = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier2_pred.fit(X_2, y_2)

    y_pred2_test = classifier2_pred.predict(X_2_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_2_two, y_pred2_test))
    print(classification_report(y_2_two, y_pred2_test))
    print(accuracy_score(y_2_two, y_pred2_test))

    print()
    print("NB validation keywords1 word2vec")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier2_NB = naive_bayes.GaussianNB()
    classifier2_NB.fit(X_train_2, y_train_2)

    y_pred2_NB = classifier2_NB.predict(X_test_2)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_2, y_pred2_NB))
    print(classification_report(y_test_2, y_pred2_NB))
    print(accuracy_score(y_test_2, y_pred2_NB))

    print()
    print("NB Test keywords1 word2vec")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier2_NB_test = naive_bayes.GaussianNB()
    classifier2_NB_test.fit(X_2, y_2)

    y_pred2_NB_test = classifier2_NB_test.predict(X_2_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_two, y_pred2_NB_test))
    print(classification_report(y_two, y_pred2_NB_test))
    print(accuracy_score(y_two, y_pred2_NB_test))

    X_3 = tf_idf_vector_biased2
    y_3 = []

    X_3_two = tf_idf_vector_biased2_test
    y_3_two = []

    # merge categories
    for document in transcripts_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_3.append(class_made)

    for document in ourwatchedset_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_3_two.append(class_made)

    # train test split for validation
    from sklearn.model_selection import train_test_split
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=0)

    print()
    print("SVM validation keywords2 word2vec")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier3 = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier3.fit(X_train_3, y_train_3)

    y_pred3 = classifier3.predict(X_test_3)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_3, y_pred3))
    print(classification_report(y_test_3, y_pred3))
    print(accuracy_score(y_test_3, y_pred3))

    print()
    print("SVM test keywords2 word2vec")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier3_pred = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier3_pred.fit(X_3, y_3)

    y_pred3_test = classifier3_pred.predict(X_3_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_3_two, y_pred3_test))
    print(classification_report(y_3_two, y_pred3_test))
    print(accuracy_score(y_3_two, y_pred3_test))

    print()
    print("NB validation keywords2 word2vec")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier3_NB = naive_bayes.GaussianNB()
    classifier3_NB.fit(X_train_3, y_train_3)

    y_pred3_NB = classifier3_NB.predict(X_test_3)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_3, y_pred3_NB))
    print(classification_report(y_test_3, y_pred3_NB))
    print(accuracy_score(y_test_3, y_pred3_NB))

    print()
    print("NB Test keywords2 word2vec")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier3_NB_test = naive_bayes.GaussianNB()
    classifier3_NB_test.fit(X_3, y_3)

    y_pred3_NB_test = classifier3_NB_test.predict(X_3_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_3_two, y_pred3_NB_test))
    print(classification_report(y_3_two, y_pred3_NB_test))
    print(accuracy_score(y_3_two, y_pred3_NB_test))

    X_4 = tf_idf_vector_biased3
    y_4 = []

    X_4_two = tf_idf_vector_biased3_test
    y_4_two = []

    # merge categories
    for document in transcripts_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_4.append(class_made)

    for document in ourwatchedset_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_4_two.append(class_made)

    # train test split for validation
    from sklearn.model_selection import train_test_split
    X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_4, y_4, test_size=0.2, random_state=0)

    print()
    print("SVM validation keywords1 Glove")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier4 = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier4.fit(X_train_4, y_train_4)

    y_pred4 = classifier4.predict(X_test_4)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_4, y_pred4))
    print(classification_report(y_test_4, y_pred4))
    print(accuracy_score(y_test_4, y_pred4))

    print()
    print("SVM test keywords1 Glove")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier4_pred = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier4_pred.fit(X_4, y_4)

    y_pred4_test = classifier4_pred.predict(X_4_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_4_two, y_pred4_test))
    print(classification_report(y_4_two, y_pred4_test))
    print(accuracy_score(y_4_two, y_pred4_test))

    print()
    print("NB validation keywords1 Glove")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier4_NB = naive_bayes.GaussianNB()
    classifier4_NB.fit(X_train_4, y_train_4)

    y_pred4_NB = classifier4_NB.predict(X_test_4)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_4, y_pred4_NB))
    print(classification_report(y_test_4, y_pred4_NB))
    print(accuracy_score(y_test_4, y_pred4_NB))

    print()
    print("NB Test keywords1 Glove")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier4_NB_test = naive_bayes.GaussianNB()
    classifier4_NB_test.fit(X_4, y_4)

    y_pred4_NB_test = classifier4_NB_test.predict(X_4_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_4_two, y_pred4_NB_test))
    print(classification_report(y_4_two, y_pred4_NB_test))
    print(accuracy_score(y_4_two, y_pred4_NB_test))

    X_5 = tf_idf_vector_biased4
    y_5 = []

    X_5_two = tf_idf_vector_biased4_test
    y_5_two = []

    # merge categories
    for document in transcripts_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_5.append(class_made)

    for document in ourwatchedset_cleaned:

        class_made = 0
        if document[1] == 1:
            class_made = 0
        else:
            class_made = 1
        y_5_two.append(class_made)

    # train test split for validation
    from sklearn.model_selection import train_test_split
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5, y_5, test_size=0.2, random_state=0)

    print()
    print("SVM validation keywords2 Glove")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier5 = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier5.fit(X_train_5, y_train_5)

    y_pred5 = classifier5.predict(X_test_5)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_5, y_pred5))
    print(classification_report(y_test_5, y_pred5))
    print(accuracy_score(y_test_5, y_pred5))

    print()
    print("SVM test keywords2 Glove")
    # Support Vector Machine Classifier
    from sklearn import svm
    classifier5_pred = svm.SVC(C=2.8, kernel='linear', degree=7, gamma='auto')
    classifier5_pred.fit(X_5, y_5)

    y_pred5_test = classifier5_pred.predict(X_5_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_5_two, y_pred5_test))
    print(classification_report(y_5_two, y_pred5_test))
    print(accuracy_score(y_5_two, y_pred5_test))

    print()
    print("NB validation keywords2 Glove")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier5_NB = naive_bayes.GaussianNB()
    classifier5_NB.fit(X_train_5, y_train_5)

    y_pred5_NB = classifier5_NB.predict(X_test_5)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test_5, y_pred5_NB))
    print(classification_report(y_test_5, y_pred5_NB))
    print(accuracy_score(y_test_5, y_pred5_NB))

    print()
    print("NB Test keywords2 Glove")
    # naive Bayes classifier
    from sklearn import naive_bayes
    classifier5_NB_test = naive_bayes.GaussianNB()
    classifier5_NB_test.fit(X_5, y_5)

    y_pred5_NB_test = classifier5_NB_test.predict(X_5_two)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_5_two, y_pred5_NB_test))
    print(classification_report(y_5_two, y_pred5_NB_test))
    print(accuracy_score(y_5_two, y_pred5_NB_test))

def dataset_shuffler(ourwatched_set_filepath, pickle_filepath, dataset_filepath, transcript_filepath):
        """
        This function collects our transcript data from Marks set and ours, combines it into one
        then stratified-splits it into test_data and training_data.
        :param ourwatched_set_filepath: input the folder where Transcripts2nddataset.csv is located
        :param pickle_filepath: input the folder path where you want to save the training and test set
        :param dataset_filepath: input the folder path where the YouTube.csv is located
        :param transcript_filepath: input the folder path where the JSONs of transcripts are located
        :return: returns two variables, one with test, one with training data, can be called like this: test_data, training_data = dataset_shuffler('D:/School/CSAI/Thesis/Data Exploration Project','D:/School/CSAI/Thesis/Data Exploration Project','D:/School/CSAI/Thesis/Dataset','D:/School/CSAI/Thesis/Dataset/Transcripts')
        also returns two picklefiles in the specified folder containing the same sets.
        """
        import os
        import csv
        import numpy as np
        import pickle
        from my_FunctionsModule_ForThesis import TranscriptExtractor
        Transcripts = TranscriptExtractor(dataset_filepath, transcript_filepath)
        blabla = Transcripts

        blabla.sort()

        bla = []
        for item in blabla:
            bla.append([item[2], item[3]])
        new_k = []
        for elem in bla:
            if elem not in new_k:
                new_k.append(elem)
        bla = new_k

        ourwatchedset = []
        os.chdir(ourwatched_set_filepath)
        with open(ourwatched_set_filepath + '/Transcripts2nddataset.csv', 'r', encoding='utf8') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                else:
                    ourwatchedset.append(row)

        os.chdir(ourwatched_set_filepath)

        combined = []
        for file in new_k:
            combined.append([file[0], file[1]])

        for transcript in ourwatchedset:
            combined.append([transcript[1], transcript[2]])
        X = []
        y = []
        for item in combined:
            if item[1] != 'x':
                X.append(item[0])
            if item[1] != 'x':
                y.append(int(item[1]))
        X = np.array(X)
        y = np.array(y)

        from sklearn.model_selection import StratifiedShuffleSplit
        indiced = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=4)

        for train_index, test_index in indiced.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        training_data = []

        test_data = []

        for i, item in enumerate(X_train):
            training_data.append([str(X_train[i]), y_train[i]])
        for i, item in enumerate(X_test):
            test_data.append([str(X_test[i]), y_test[i]])

        import pickle
        os.chdir(pickle_filepath)
        with open('training_data.pickle', 'wb') as f:
            pickle.dump(training_data, f)
        with open('test_data.pickle', 'wb') as f:
            pickle.dump(test_data, f)
        return test_data, training_data

def my_cleaner3(text,nlp):

    return[token.lemma_.lower() for token in nlp(text) if not (token.is_alpha==False or len(token.lemma_) <3) ]

def cleaner(training_data,nlp,filepath,filename):
    '''

    :param training_data: input here your list containing training or test dataset
    :param nlp: specify variable that holds nlp model
    :param filepath: input here your folder where you want the pickle
    :param filepath: input here your name for the pickle

    :return: returns cleaned version (tokenized and lowercased)
    '''

    training_data_cleaned = []
    for i, transcript in enumerate(training_data):
        doc = nlp(training_data[i][0])
        cleaned_tokens1 = my_cleaner3(doc.text, nlp=nlp)
        training_data_cleaned.append([cleaned_tokens1, training_data[i][1]])

    import os
    os.chdir(filepath)
    import pickle
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(training_data_cleaned, f)
    return training_data_cleaned

