import os
import _pickle as pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from operator import add


def main():
    start = time.time()
    # load pickle
    arxiv_11 = dictload(2011)
    arxiv_12 = dictload(2012)
    arxiv_13 = dictload(2013)
    arxiv_14 = dictload(2014)
    arxiv_15 = dictload(2015)
    arxiv_16 = dictload(2016)
    arxiv_17 = dictload(2017)

    arxiv_list = [arxiv_11,
                  arxiv_12,
                  arxiv_13,
                  arxiv_14,
                  arxiv_15,
                  arxiv_16,
                  arxiv_17
                  ]
    tr_set_names = ['astro', 'cond', 'cs', 'hep', 'math', 'physics', 'qbio', 'qfin', 'quant', 'stat']

    print("loaded pickles")

    # build proportion matrix
    list_of_lists = []
    for year in arxiv_list:
        count = []
        for name in tr_set_names:
            try:
                count.append(len(year[name]))
            except KeyError:
                count.append(0)
        list_of_lists.append(count)
    array = np.array(list_of_lists)
    print(array)

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
    # plot_metadata(frac)
    plot_bar(array)


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


# stacked bar plot
def plot_bar(motherlist):
    # position
    r = [0, 1, 2, 3, 4, 5, 6]
    # Names of group and bar width
    names = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'steelblue', 'seashell',
              'wheat', 'lightskyblue', 'lightcoral', 'steelblue']
    bar_width = 0.7
    # first bar
    motherlist = np.transpose(motherlist)
    bar0 = motherlist[0]
    height = [0] * len(bar0)
    plt.bar(r, bar0, color=colors[0], edgecolor='black', width=bar_width)
    for x in range(1, 10):
        bar = motherlist[x]
        height = np.add(motherlist[x-1], height)
        print(height)
        plt.bar(r, bar, bottom=height, color=colors[x], edgecolor='black', width=bar_width)

    # Custom X axis
    plt.xticks(r, names, fontweight='bold')
    plt.xlabel("years")

    # y-axis in bold
    plt.yticks(np.arange(0, 200000, step=50000), fontweight='bold')
    plt.ylabel("no.of articles")

    # Show graphic
    plt.show()


if __name__ == "__main__":
    main()

