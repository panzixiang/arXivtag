import os
import io

save_path = "../Data/raw"
filename_in = "out.xml"
filename_in = os.path.join(save_path, filename_in)

filename_out = "out2.xml"
filename_out = os.path.join(save_path, filename_out)

with io.open(filename_in, "r", encoding="utf-8") as inputfile:
    with io.open(filename_out, "w", encoding="utf-8") as output:
        for inputline in inputfile:
            if not inputline.startswith(("<?xml version",
                                    "<resumptiontoken", "</resumptiontoken",
                                         #"<listrecords>", "</listrecords>",
                                    "<oai-pmh", "</oai-pmh",
                                    #"<responsedate>", "</responsedate>"
                                         )):
                output.write(inputline)
