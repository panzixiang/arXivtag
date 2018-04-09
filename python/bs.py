from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import sys
import os
import io
import pickle

""" This script takes in xml and splits them into lists of category and their abstracts and saves them in a pickle"""


def main():
    print(sys.argv)
    year = sys.argv[1]
    process(year)


def process(year):
    save_path = "../Data/raw"
    filename = "arXiv" + str(year) + "full.xml"
    filename = os.path.join(save_path, filename)

    strainer_cat = SoupStrainer("categories")

    strainer_abs = SoupStrainer("abstract")

    soup_cat = BeautifulSoup(io.open(filename, encoding="utf-8"), "lxml", parse_only=strainer_cat)

    print("soup cat loaded, size: " + str(os.path.getsize(filename)) + " bytes")

    soup_abs = BeautifulSoup(io.open(filename, encoding="utf-8"), "lxml", parse_only=strainer_abs)

    print("soup abs loaded, size: " + str(os.path.getsize(filename)) + " bytes")

    # find all sub-categories
    allcats = [x.text for x in soup_cat.find_all('categories')]

    print("number of articles: " + str(len(allcats)))

    print()

    # split individual tags and get the first as the category
    pri_tagset = [(taglist.split())[0] for taglist in allcats]
    # see uniques
    uniq_tagset = sorted(set(pri_tagset))

    print("unique first tags")
    print(uniq_tagset)

    # list all abstracts
    allabs = [x.text.replace('\n', ' ').strip() for x in soup_abs.find_all('abstract')]

    print("number of abstracts: " + str(len(allabs)))

    print()

    # concat for future use
    withtag = map(lambda x, y: x + ': ' + y, allcats, allabs)
    sorted(withtag)

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
    for i in range(len(pri_tagset)):
        if 'astro' in pri_tagset[i]:
            astro.append(allabs[i])
        elif 'cond' in pri_tagset[i]:
            cond.append(allabs[i])
        elif any(ext in pri_tagset[i] for ext in ['chao', 'gr-qc', 'nlin', 'nucl', 'physics', 'phys']):
            physics.append(allabs[i])
        elif 'cs' in pri_tagset[i]:
            cs.append(allabs[i])
        elif 'hep' in pri_tagset[i]:
            hep.append(allabs[i])
        elif 'math' in pri_tagset[i]:
            math.append(allabs[i])
        elif 'q-bio' in pri_tagset[i]:
            qbio.append(allabs[i])
        elif 'q-fin' in pri_tagset[i]:
            qfin.append(allabs[i])
        elif 'quant' in pri_tagset[i]:
            quant.append(allabs[i])
        elif 'stat' in pri_tagset[i]:
            stat.append(allabs[i])
        else:
            others.append(allabs[i])

    # dict for pickle dump
    bigcatDict = {'astro': astro, 'cond': cond, 'cs': cs, 'hep': hep, 'math': math, 'physics': physics, 'qbio': qbio,
                  'qfin': qfin, 'quant': quant, 'stat': stat, 'others': others}

    for key in iter(bigcatDict.keys()):
        print(key + " " + str(len(bigcatDict[key])))

    dictname = "../Data/dict/" + str(year) + "2_big_pop.p"
    pickle.dump(bigcatDict, open(dictname, "wb"))
    return


if __name__ == "__main__":
    main()
