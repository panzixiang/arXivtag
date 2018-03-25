from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
import math
import os
import time
import gensim
import cPickle as pickle
import numpy as np


def main():

    start = time.time()
    print("start ---------------------------------------------------")

    arxiv_15 = dictload(2015)
    intermediate_path = "../Data/Intermediate/"
    doc_set = pickle.load(open(intermediate_path + 'doc_set.p', "rb"))
    label_set = pickle.load(open(intermediate_path + 'label_set.p', "rb"))
    topic_superset = pickle.load(open(intermediate_path + 'topic_superset.p', "rb"))

    time_load = time.time()
    print("It took", time_load-start, "seconds to load")
    print("training ------------------------------------------------")

    doc_texts = tokenize(doc_set)

    print("tokenized")

    # build individual lda
    lda_superset = []
    num_topics_list = []
    dictionary_set = []

    for topic_set in topic_superset:
        topic_texts = tokenize(topic_set)

        # turn our tokenized documents into a id - term dictionary
        dictionary = corpora.Dictionary(topic_texts)
        dictionary_set.append(dictionary)

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in topic_texts]

        # generate LDA model
        num_topics = math.floor(len(topic_set)/100)
        num_topics_list.append(num_topics)
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
        lda_superset.append(ldamodel)

    print("all LDA built")

    # build training matrix
    prop_array_superset = []
    for i in range(len(num_topics_list)):
        num_topics = num_topics_list[i]
        topic_prop_array = np.zeros((len(doc_texts), num_topics))
        for j in range(len(doc_texts)):
            text = doc_texts[j]
            textProp = lda_superset[i][dictionary_set[i].doc2bow(text)]
            for pair in textProp:
                topicIdx = pair[0]
                weight = pair[1]
                topic_prop_array[j, topicIdx] = weight
        prop_array_superset.append(topic_prop_array)

    # concat full feature array
    trainingArray = prop_array_superset[0]
    for i in range(len(prop_array_superset)):
        if i != 0:
            trainingArray = np.concatenate((trainingArray, prop_array_superset[i]), axis=1)

    print("training matrix built")
    time_train = time.time()
    print("It took", time_train-time_load, "seconds to train")
    print("---------------------------------------------------------")
    print("testing")

    # test on new data
    test_set = arxiv_15['astro'][0:252] + arxiv_15['cond'][0:390] + \
        arxiv_15['cs'][0:386] + arxiv_15['hep'][0:408] + \
        arxiv_15['math'][0:866] + arxiv_15['physics'][0:380] + \
        arxiv_15['qbio'][0:40] + arxiv_15['qfin'][0:18] + \
        arxiv_15['quant'][0:131] + arxiv_15['stat'][0:47]
    test_label = [1]*253 + [2]*391 + [3]*387 + [4]*409 + [5]*867 + \
        [6]*381 + [7]*41 + [8]*18 + [9]*132 + [10]*48

    test_texts = tokenize(test_set)

    # build indiv test prop array
    test_prop_array_superset = []
    for i in range(len(num_topics_list)):
        num_topics = num_topics_list[i]
        test_prop_array = np.zeros((len(test_label), num_topics))
        for j in range(len(test_texts)):
            test = test_texts[j]
            testProp = lda_superset[i][dictionary_set[i].doc2bow(test)]
            for pair in testProp:
                topicIdx = pair[0]
                weight = pair[1]
                test_prop_array[j, topicIdx] = weight
        test_prop_array_superset.append(test_prop_array)

    # concat full test array
    test_array = test_prop_array_superset[0]
    for i in range(len(test_prop_array_superset)):
        if (i != 0):
            test_array = np.concatenate((test_array, test_prop_array_superset[i]), axis=1)

    pickle.dump(trainingArray, open(intermediate_path + 'train_array.p', "wb"))
    pickle.dump(test_array, open(intermediate_path + 'train_label.p', "wb"))

    x_train, x_test, y_train, y_test = trainingArray, test_array, label_set, test_label

    print("training_array length: " + str(len(topic_prop_array)))
    print("test_array length: " + str(len(test_prop_array)))
    print("training_label length: " + str(len(label_set)))
    print("test_label length: " + str(len(test_label)))
    print("---------------------------------------------------------")

    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = "../Results/" + now

    # knn3
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(x_train, y_train)
    predictions = knn3.predict(x_test)
    np.savetxt(save_path+'splitova_knn3pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print('knn3')
    print(accuracy_score(predictions, y_test))
    print('--------------------------------')

    # knn5
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(x_train, y_train)
    predictions = knn5.predict(x_test)
    np.savetxt(save_path+'splitova_knn5pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print('knn5')
    print(accuracy_score(predictions, y_test))
    print('--------------------------------')

    # svmlin
    svmlin = svm.SVC(kernel='linear')
    svmlin.fit(x_train, y_train)
    predictions = svmlin.predict(x_test)
    np.savetxt(save_path+'splitova_svmpred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print('svmlin')
    print(accuracy_score(predictions, y_test))
    print('--------------------------------')

    # gnb
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    predictions = gnb.predict(x_test)
    np.savetxt(save_path+'splitova_gnbpred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print('gnb')
    print(accuracy_score(predictions, y_test))
    print('--------------------------------')

    # rf50
    rf50 = RandomForestClassifier(n_estimators=50)
    rf50.fit(x_train, y_train)
    predictions = rf50.predict(x_test)
    np.savetxt(save_path+'splitova_rf50pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print('rf50')
    print(accuracy_score(predictions, y_test))
    print('--------------------------------')

    # dtree ada
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=400,
                             learning_rate=1,
                             algorithm="SAMME",
                             random_state=None)
    ada.fit(x_train, y_train)
    predictions = ada.predict(x_test)
    np.savetxt(save_path+'splitova_adapred.csv', predictions.astype(int), fmt='%i', delimiter=",")
    # print predictions
    print('ada')
    print(accuracy_score(predictions, y_test))
    print('--------------------------------')

    time_end = time.time()
    print("total time is ", time_end-start)


def tokenize(doc_set):
    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # create tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    doc_texts = []
    # loop through document list
    for i in doc_set:

        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if i not in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        doc_texts.append(stemmed_tokens)

    return doc_texts


def dictload(year):
    save_path = "../Data/dict"
    filename = str(year) + "_big_pop.p"
    filename = os.path.join(save_path, filename)

    # load pickle
    return pickle.load(open(filename, "rb"))


if __name__ == "__main__":
    main()
