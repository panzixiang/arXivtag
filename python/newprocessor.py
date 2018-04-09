from collections import OrderedDict
from collections import Counter
from itertools import islice
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
            astro.append(value)
        elif 'cond' in value[0]:
            cond.append(value)
        elif any(ext in value[0] for ext in ['chao', 'gr-qc', 'nlin', 'nucl', 'physics', 'phys']):
            physics.append(value)
        elif 'cs' in value[0]:
            cs.append(value)
        elif 'hep' in value[0]:
            hep.append(value)
        elif 'math' in value[0]:
            math.append(value)
        elif 'q-bio' in value[0]:
            qbio.append(value)
        elif 'q-fin' in value[0]:
            qfin.append(value)
        elif 'quant' in value[0]:
            quant.append(value)
        elif 'stat' in value[0]:
            stat.append(value)
        else:
            others.append(value)

    # dict for pickle dump
    bigcatDict = {'astro': astro, 'cond': cond, 'cs': cs, 'hep': hep, 'math': math, 'physics': physics,
                  'qbio': qbio, 'qfin': qfin, 'quant': quant, 'stat': stat, 'others': others}

    for key in iter(bigcatDict.keys()):
        print(key + " " + str(len(bigcatDict[key])))

    build_stats(dict0704)
    #dictname = "../Data/dict/big_pop.p"
    #pickle.dump(bigcatDict, open(dictname, "wb"))


# helper function that prints a small sample of a dictionary
def print_sample(in_dict, n):
    # see a sample of dictionary
    for key, value in list(islice(in_dict.items(), n)):
        print(key, value)


# helper to get per year per subject stats
def build_stats(in_dict):
    list_of_years = sorted(list(set([key[:2] for key in in_dict.keys()])))
    topic_list = ['astro', 'cond', 'cs', 'hep', 'math', 'physics', 'q-bio', 'q-fin', 'quant', 'stat']
    # build a dictionary where the key is year and the value is dictionary of counts
    year_count = {}
    for year in list_of_years:
        list_of_setSpec = [value[0].split('-', 1)[0] for key, value in in_dict.items()
                           if key[:2] == year and any(ext in value[0] for ext in topic_list)]
        year_count[year] = OrderedDict(sorted(Counter(list_of_setSpec).items()))

        print("year: " + year)
        print(' '.join(['{0}: {1}'.format(k, v) for k, v in year_count[year].items()]))




if __name__ == "__main__":
        main()
