import os
import pickle
import time
import matplotlib.pyplot as plt


def main():
    start = time.time()
    # load pickle
    arXiv_full = pickle.load(open("../Data/dict/big_pop.p", "rb"))

    arxiv_list = [arXiv_full]

    tr_set_names = ['astro', 'cond', 'cs', 'hep', 'math', 'physics', 'qbio', 'qfin', 'quant', 'stat']

    print("loaded pickles")

    # build doc set
    dict_doc_set = {}
    for name in tr_set_names:
        dict_doc_set[name] = build_docset(name, arxiv_list)

    # get sorted list of keys
    keys = sorted(list(dict_doc_set.keys()))

    doc_set = []
    label_set = []
    topic_superset = []
    frac = []
    i = 1
    for key in keys:
        doc_set = doc_set + dict_doc_set[key]
        # print doc_set
        label_set = label_set + [i] * len(dict_doc_set[key])
        # print label_set
        # topic based training sets
        topic_superset.append(dict_doc_set[key])
        # print topic_superset
        frac.append(float(len(dict_doc_set[key])))
        i = i + 1
        print(key + ": " + str(len(dict_doc_set[key])))
        assert (len(doc_set) == len(label_set)), "doc_set and label_set size mismatch!"

    frac = [x / len(label_set) for x in frac]

    intermediate_path = "../Data/intermediate/"
    pickle.dump(doc_set, open(intermediate_path + 'doc_set.p', "wb"))
    pickle.dump(label_set, open(intermediate_path + 'label_set.p', "wb"))
    pickle.dump(topic_superset, open(intermediate_path + 'topic_superset.p', "wb"))
    time_load = time.time()
    print("It took", time_load - start, "seconds to process")
    plot_metadata(frac)


def dictload(year):
    save_path = "../Data/dict"
    filename = str(year) + "_big_pop.p"
    filename = os.path.join(save_path, filename)

    # load pickle
    return pickle.load(open(filename, "rb"))


def build_docset(topic, arxiv_list):
    doc_set = []
    for doc_dict in arxiv_list:
        doc_set = doc_set + doc_dict[topic]
    return list(set(doc_set))


# sum pie chart
def plot_metadata(frac):
    # make a square figure and axes
    labels = ['astro', 'cond', 'cs', 'hep', 'math', 'physics', 'qbio', 'qfin', 'quant', 'stat']
    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('equal')
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'steelblue', 'seashell',
              'wheat', 'lightskyblue', 'lightcoral', 'steelblue']
    fig.patch.set_facecolor('white')

    # The slices will be ordered and plotted counter-clockwise.
    explode = [0] * 10

    plt.pie(frac, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)

    plt.title('Training set proportions', y=1.05)

    plt.show()

if __name__ == "__main__":
    main()

