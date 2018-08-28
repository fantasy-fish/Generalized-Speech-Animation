import os
import json

#create the directory
dir = "../transcripts"
if(not os.path.exists(dir)):
    os.makedirs(dir)

#read sentences and create the json files
with open("sentences") as f:
    for line in f:
        n0 = line.find('(')
        sentence = line[:n0-1]
        id = line[n0+1:-2]
        filename = os.path.join(dir,id+".json")
        jsondict = dict()
        jsondict["speaker"] = "Unknown"
        jsondict["line"] = sentence
        jsonlist =[jsondict]
        with open(filename, 'w') as outfile:
            json.dump(jsonlist, outfile)

