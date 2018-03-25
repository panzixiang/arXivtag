import urllib.request
import time
from bs4 import BeautifulSoup
import sys
import os


def main():
    print(sys.argv)
    year = sys.argv[1]
    harvest_by_year(year)
    joinxml(year)


def harvest_by_year(year):
    save_path = "../Data/raw"
    filename = "arXiv" + str(year) + ".xml"
    filename = os.path.join(save_path, filename)
    f = open(filename, 'a')
    first_url = "http://export.arxiv.org/oai2?verb=ListRecords&from=" + \
        str(year) + "-01-01&until=" + \
        str(year) + "-12-31&metadataPrefix=arXiv"
    data = urllib.request.urlopen(first_url).read()
    soup = BeautifulSoup(data, 'lxml')
    f.write(soup.prettify())

    token = soup.find('resumptiontoken').text
    resume = True

    # loop over resumption tokens till the end
    while resume:
        # wait for server
        time.sleep(21)
        url = 'http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=' + token

        next_data = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(next_data, 'html.parser')
        f.write(soup.prettify())
        f.write(soup.prettify())
        if soup.find('resumptiontoken') is not None:
            token = soup.find('resumptiontoken').text
            if token is "":
                resume = False
                break
        else:
            resume = False
            break
    return


def joinxml(year):
    save_path = "../Data/raw"
    filename_in = "arXiv" + str(year) + ".xml"
    filename_in = os.path.join(save_path, filename_in)

    filename_out = "arXiv" + str(year) + "full.xml"
    filename_out = os.path.join(save_path, filename_out)

    with open(filename_in, "r") as inputfile:
        with open(filename_out, "w") as output:
            for inputline in inputfile:
                if "<?xml version" not in inputline:
                    output.write(inputline)
    return


if __name__ == "__main__":
    main()
