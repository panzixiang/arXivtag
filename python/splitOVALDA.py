import nltk
import math
import os
import time
import gensim
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime


def main():

    start = time.time()
    print("start ---------------------------------------------------")

    # load test set
    test_year = dictload(2018)

    # load the rest
    intermediate_path = "../Data/Intermediate/"
    doc_set = pickle.load(open(os.path.join(intermediate_path + 'doc_set.p'), "rb"))
    label_set = pickle.load(open(os.path.join(intermediate_path + 'label_set.p'), "rb"))
    topic_superset = pickle.load(open(os.path.join(intermediate_path + 'topic_superset.p'), "rb"))

    time_load = time.time()
    print("It took", time_load-start, "seconds to load")
    print("training ------------------------------------------------")

    doc_texts = tokenize(doc_set)

    print("tokenized")

    # build individual lda
    lda_superset = []
    num_topics_list = []
    dictionary_set = []

    i = 0
    for topic_set in topic_superset:
        topic_texts = tokenize(topic_set)

        # turn our tokenized documents into a id - term dictionary
        dictionary = corpora.Dictionary(topic_texts)
        dictionary_set.append(dictionary)

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in topic_texts]

        # generate LDA model
        # number of topics is logarithmic
        num_topics = math.floor(math.log2(len(topic_set)))
        print(str(i) + ' ' + "number of topics: " + str(num_topics))
        num_topics_list.append(num_topics)
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
        lda_superset.append(ldamodel)
        i += 1

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
    training_array = prop_array_superset[0]
    for i in range(len(prop_array_superset)):
        if i != 0:
            training_array = np.concatenate((training_array, prop_array_superset[i]), axis=1)

    print("training matrix built")
    time_train = time.time()
    print("It took", time_train-time_load, "seconds to train")
    print("---------------------------------------------------------")
    print("testing")

    # test on new data 1000 documents split by proportion of training data
    test_set = test_year['astro'][0:144] + test_year['cond'][0:145] + \
        test_year['cs'][0:125] + test_year['hep'][0:113] + \
        test_year['math'][0:257] + test_year['physics'][0:134] + \
        test_year['qbio'][0:13] + test_year['qfin'][0:6] + \
        test_year['quant'][0:45] + test_year['stat'][0:17]
    test_label = [1]*144 + [2]*145 + [3]*125 + [4]*113 + [5]*257 + \
        [6]*134 + [7]*13 + [8]*6 + [9]*45 + [10]*17

    test_texts = tokenize(test_set)

    # build individual test prop array
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
        if i != 0:
            test_array = np.concatenate((test_array, test_prop_array_superset[i]), axis=1)

    arraydump('log2_topics_', training_array, test_array)

    x_train, x_test, y_train, y_test = training_array, test_array, label_set, test_label

    print("training_array length: " + str(len(topic_prop_array)))
    print("test_array length: " + str(len(test_prop_array)))
    print("training_label length: " + str(len(label_set)))
    print("test_label length: " + str(len(test_label)))
    print("---------------------------------------------------------")

    # choose model via a list
    model_names = ["knn3"]
    buildmodel(model_names, x_train, y_train, x_test, y_test)

    time_end = time.time()
    print("total time is ", time_end-start)


def tokenize(doc_set):
    # create English stop words list
    en_stop = stopwords.words('english')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # create tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    doc_texts = []
    # loop through document list
    for doc in doc_set:

        # doc is a tuple in the form (id, category, text)
        # clean and tokenize document string
        raw = doc[2].lower()
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


def arraydump(suffix, training_array, test_array):
    intermediate_path = "../Data/Intermediate/"
    pickle.dump(training_array, open(intermediate_path + suffix + 'train_array.p', "wb"), protocol=4)
    pickle.dump(test_array, open(intermediate_path + suffix + 'test_array.p', "wb"), protocol=4)


def buildmodel(model_names, x_train, y_train, x_test, y_test):
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = "../Results/" + now
    if "knn3" in model_names:
        # knn3
        knn3 = KNeighborsClassifier(n_neighbors=3)
        knn3.fit(x_train, y_train)
        predictions = knn3.predict(x_test)
        np.savetxt(save_path+'splitova_knn3pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
        # print predictions
        print('knn3')
        print(accuracy_score(predictions, y_test))
        print('--------------------------------')

    if "knn5" in model_names:
        # knn5
        knn5 = KNeighborsClassifier(n_neighbors=5)
        knn5.fit(x_train, y_train)
        predictions = knn5.predict(x_test)
        np.savetxt(save_path+'splitova_knn5pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
        # print predictions
        print('knn5')
        print(accuracy_score(predictions, y_test))
        print('--------------------------------')

    if "svmlin" in model_names:
        # svmlin
        svmlin = svm.SVC(kernel='linear')
        svmlin.fit(x_train, y_train)
        predictions = svmlin.predict(x_test)
        np.savetxt(save_path+'splitova_svmpred.csv', predictions.astype(int), fmt='%i', delimiter=",")
        # print predictions
        print('svmlin')
        print(accuracy_score(predictions, y_test))
        print('--------------------------------')

    if "gnb" in model_names:
        # gnb
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        predictions = gnb.predict(x_test)
        np.savetxt(save_path+'splitova_gnbpred.csv', predictions.astype(int), fmt='%i', delimiter=",")
        # print predictions
        print('gnb')
        print(accuracy_score(predictions, y_test))
        print('--------------------------------')

    if "rf50" in model_names:
        # rf50
        rf50 = RandomForestClassifier(n_estimators=50)
        rf50.fit(x_train, y_train)
        predictions = rf50.predict(x_test)
        np.savetxt(save_path+'splitova_rf50pred.csv', predictions.astype(int), fmt='%i', delimiter=",")
        # print predictions
        print('rf50')
        print(accuracy_score(predictions, y_test))
        print('--------------------------------')

    if "adatree" in model_names:
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


if __name__ == "__main__":
    main()
