from collections import OrderedDict
from collections import Counter
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():
    dictname = "../Data/dict/full_articleset.p"
    article_dic = pickle.load(open(dictname, "rb"))

    # keys that look like this oai:arXiv.org:adap-org/9806001
    dict9107 = {key: article_dic[key] for key in list(article_dic.keys()) if '/' in key}
    # dict9107 is currently unused

    # keys that look like this oai:arXiv.org:0704.0010
    dict0704 = {key: article_dic[key] for key in list(article_dic.keys()) if '/' not in key}

    # build individual lists
    astro = []
    cond = []
    cs = []
    hep = []
    math = []
    physics = []
    qbio = []
    qfin = []
    quant = []
    stat = []
    others = []
    for key, value in dict0704.items():
        if 'astro' in value[0]:
            astro.append((key, value[0], value[1]))
        elif 'cond' in value[0]:
            cond.append((key, value[0], value[1]))
        elif any(ext in value[0] for ext in ['chao', 'gr-qc', 'nlin', 'nucl', 'physics', 'phys']):
            physics.append((key, value[0], value[1]))
        elif 'cs' in value[0]:
            cs.append((key, value[0], value[1]))
        elif 'hep' in value[0]:
            hep.append((key, value[0], value[1]))
        elif 'math' in value[0]:
            math.append((key, value[0], value[1]))
        elif 'q-bio' in value[0]:
            qbio.append((key, value[0], value[1]))
        elif 'q-fin' in value[0]:
            qfin.append((key, value[0], value[1]))
        elif 'quant' in value[0]:
            quant.append((key, value[0], value[1]))
        elif 'stat' in value[0]:
            stat.append((key, value[0], value[1]))
        else:
            others.append((key, value[0], value[1]))

    # dict for pickle dump
    # this dictionary is in the form subject: (id, category, abstract)
    bigcat_dict = {'astro': astro, 'cond': cond, 'cs': cs, 'hep': hep, 'math': math, 'physics': physics,
                  'qbio': qbio, 'qfin': qfin, 'quant': quant, 'stat': stat, 'others': others}

    # for key in iter(bigcatDict.keys()):
    #    print(key + ": " + str(len(bigcatDict[key])))

    build_stats(bigcat_dict)
    dictname = "../Data/dict/big_pop.p"
    pickle.dump(bigcat_dict, open(dictname, "wb"))


# helper function that prints a small sample of a dictionary
def print_sample(in_dict, n):
    # see a sample of dictionary
    for key, value in list(islice(in_dict.items(), n)):
        print(key, value)


# helper to get per subject per year stats
def build_stats(in_dict):
    subject_count = {}
    # build a dictionary where the key is the subject and the value is dictionary of counts
    for subject in in_dict.keys():
        list_of_year = [tuple[0][:2]for tuple in in_dict[subject]]
        subject_count[subject] = OrderedDict(sorted(Counter(list_of_year).items()))

    #for subject in subject_count.keys():
    #    print(subject + ': ' + ' '.join(['{0}: {1}'.format(k, v) for k, v in subject_count[subject].items()]))

    # build proportion matrix
    list_of_lists = []
    for subject in subject_count.keys():
        count = []
        for year in subject_count['astro'].keys():
            count.append((subject_count[subject].get(year, 0)))
        list_of_lists.append(count)
    array = np.array(list_of_lists)
    print(np.matrix(list_of_lists))
    plot_bar(array, subject_count['astro'].keys())


# stacked bar plot
def plot_bar(in_array, list_of_year):
    # position
    y_pos = np.arange(len(list_of_year))
    # Names of group and bar width
    years = list_of_year
    names = ['astro', 'cond', 'cs', 'hep', 'math', 'physics', 'qbio', 'qfin', 'quant', 'stat', 'others']
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'steelblue', 'chocolate',
              'wheat', 'lightskyblue', 'orchid', 'steelblue']
    bar_width = 0.7
    opacity = 0.8
    # first bar
    bar0 = in_array[0]
    height = [0] * len(bar0)
    plt.bar(y_pos, bar0, color=colors[0], edgecolor='grey', alpha=opacity,
            width=bar_width, label=names[0])
    for x in range(1, 10):
        bar = in_array[x]
        height = np.add(in_array[x - 1], height)
        plt.bar(y_pos, bar, bottom=height, color=colors[x], edgecolor='grey', alpha=opacity,
                width=bar_width, label=names[x])

    # Custom X axis
    plt.xticks(y_pos, years, fontweight='bold')
    plt.xlabel("years")

    # y-axis in bold
    plt.yticks(np.arange(0, 150000, step=50000))
    plt.ylabel("no.of articles")

    plt.legend()

    # Show graphic
    plt.show()


if __name__ == "__main__":
        main()
