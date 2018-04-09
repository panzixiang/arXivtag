import urllib.request
import time
from bs4 import BeautifulSoup
import sys
import os
import io
import time


def main():
    harvest_routine(2015)


def harvest_routine(year):
    print("---------------" + str(year) + "---------------")
    #harvest_by_year(year)
    joinxml(year)
    time.sleep(60)


def harvest_by_year(year):
    save_path = "../Data/raw"
    filename = "arXiv" + str(year) + ".xml"
    filename = os.path.join(save_path, filename)
    f = io.open(filename, 'a', encoding="utf-8")
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

    skipone = False
    with io.open(filename_in, "r", encoding="utf-8") as inputfile:
        with io.open(filename_out, "w", encoding="utf-8") as output:
            for inputline in inputfile:
                if skipone is False:
                    skipone = True
                    continue
                if "<?xml version" not in inputline:
                    output.write(inputline)
            output.write("</html>")
    return


if __name__ == "__main__":
    main()
