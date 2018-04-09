from __future__ import print_function
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import io
import pickle


def main():

    filename = "../Data/raw/arXivbulk.xml"

    strainer_id = SoupStrainer("identifier")
    soup_id = BeautifulSoup(io.open(filename, encoding="utf-8"), "xml", parse_only=strainer_id)

    # truncate just to get id
    id_list = [x[14:] for x in soup_id.strings]

    strainer_abs = SoupStrainer("abstract")
    soup_abs = BeautifulSoup(io.open(filename, encoding="utf-8"), "xml", parse_only=strainer_abs)

    # clean newline and whitespace from abs
    abs_list = [" ".join(x.text.replace('\n', ' ').strip().split()) for x in soup_abs.find_all('abstract')]

    # reduce categories to the first big category in the first word
    strainer_cat = SoupStrainer("categories")
    soup_set = BeautifulSoup(io.open(filename, encoding="utf-8"), "xml", parse_only=strainer_cat)
    set_list = [x.split(' ', 1)[0].split('.', 1)[0] for x in soup_set.strings]
    # print(set_list)

    # build a dictionary with key = id, value = tuple of other things
    keys = id_list
    values = list(zip(set_list, abs_list))
    print(values.__len__())
    article_dic = dict(set(zip(keys, values)))
    print(article_dic.keys().__len__())


    dictname = "../Data/dict/full_articleset.p"
    pickle.dump(article_dic, open(dictname, "wb"))



if __name__ == "__main__":
    main()

